"""Cliente de almacenamiento tipo S3.

Funciona igual contra MinIO, contra AWS S3 y contra el disco local: lo único que
cambia es el endpoint. Cuando ``ACH_S3_ENDPOINT`` viene vacío se usa el sistema de
archivos local, lo que permite correr los jobs y los tests sin levantar MinIO.

pandas y pyarrow se importan dentro de los métodos de parquet, no arriba: así el
backend reutiliza este mismo cliente para leer JSON del bucket sin tener que
instalar el stack de datos completo.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from typing import TYPE_CHECKING, Any

import fsspec

from .config import Settings, get_settings

if TYPE_CHECKING:  # pragma: no cover - solo para los verificadores de tipos
    import pandas as pd
    import pyarrow as pa
    import pyarrow.parquet as pq

log = logging.getLogger(__name__)


class Storage:
    """Lectura y escritura de parquet y JSON sobre S3/MinIO o disco local."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._fs: fsspec.AbstractFileSystem | None = None

    # -- Sistema de archivos -------------------------------------------------
    @property
    def fs(self) -> fsspec.AbstractFileSystem:
        if self._fs is None:
            if self.settings.usa_s3:
                self._fs = fsspec.filesystem(
                    "s3",
                    key=self.settings.s3_access_key,
                    secret=self.settings.s3_secret_key,
                    client_kwargs={
                        "endpoint_url": self.settings.s3_endpoint,
                        "region_name": self.settings.s3_region,
                    },
                )
            else:
                self._fs = fsspec.filesystem("file", auto_mkdir=True)
        return self._fs

    def ruta(self, bucket: str, *partes: str) -> str:
        """Construye la ruta completa de un objeto dentro de un bucket."""
        cola = "/".join(str(p).strip("/") for p in partes if str(p).strip("/"))
        if self.settings.usa_s3:
            return f"{bucket}/{cola}" if cola else bucket
        raiz = self.settings.local_root.rstrip("/")
        return f"{raiz}/{bucket}/{cola}" if cola else f"{raiz}/{bucket}"

    # -- Operaciones básicas -------------------------------------------------
    def existe(self, ruta: str) -> bool:
        return bool(self.fs.exists(ruta))

    def listar(self, ruta: str, sufijo: str | None = None) -> list[str]:
        """Lista objetos bajo un prefijo. Devuelve [] si el prefijo no existe."""
        if not self.existe(ruta):
            return []
        encontrados = [p for p in self.fs.find(ruta) if not p.rstrip("/").endswith("/")]
        if sufijo:
            encontrados = [p for p in encontrados if p.endswith(sufijo)]
        return sorted(encontrados)

    def crear_directorio(self, ruta: str) -> None:
        self.fs.makedirs(ruta, exist_ok=True)

    def borrar(self, ruta: str, recursivo: bool = False) -> None:
        if self.existe(ruta):
            self.fs.rm(ruta, recursive=recursivo)

    # -- JSON ----------------------------------------------------------------
    def escribir_json(self, objeto: Any, ruta: str) -> str:
        contenido = json.dumps(objeto, ensure_ascii=False, indent=2, default=str)
        padre = ruta.rsplit("/", 1)[0]
        if padre and padre != ruta:
            self.crear_directorio(padre)
        with self.fs.open(ruta, "w", encoding="utf-8") as fh:
            fh.write(contenido)
        log.info("JSON escrito: %s (%d bytes)", ruta, len(contenido.encode("utf-8")))
        return ruta

    def leer_json(self, ruta: str) -> Any:
        with self.fs.open(ruta, "r", encoding="utf-8") as fh:
            return json.load(fh)

    # -- Parquet -------------------------------------------------------------
    def escribir_parquet(
        self,
        df: pd.DataFrame,
        ruta: str,
        particiones: Iterable[str] | None = None,
        compresion: str = "snappy",
    ) -> str:
        """Escribe un DataFrame como parquet, opcionalmente particionado (estilo Hive)."""
        import pyarrow as pa
        import pyarrow.parquet as pq

        tabla = pa.Table.from_pandas(df, preserve_index=False)
        particiones = list(particiones or [])
        if particiones:
            self.crear_directorio(ruta)
            pq.write_to_dataset(
                tabla,
                root_path=ruta,
                partition_cols=particiones,
                filesystem=self.fs,
                compression=compresion,
                existing_data_behavior="overwrite_or_ignore",
            )
        else:
            padre = ruta.rsplit("/", 1)[0]
            if padre and padre != ruta:
                self.crear_directorio(padre)
            with self.fs.open(ruta, "wb") as fh:
                pq.write_table(tabla, fh, compression=compresion)
        log.info("Parquet escrito: %s (%d filas)", ruta, len(df))
        return ruta

    def escritor_parquet(self, ruta: str, esquema: pa.Schema, compresion: str = "snappy"):
        """Devuelve un ``ParquetWriter`` abierto, para escribir por lotes sin
        materializar el dataset completo en memoria."""
        import pyarrow.parquet as pq

        padre = ruta.rsplit("/", 1)[0]
        if padre and padre != ruta:
            self.crear_directorio(padre)
        handle = self.fs.open(ruta, "wb")
        escritor = pq.ParquetWriter(handle, esquema, compression=compresion)
        return _EscritorLotes(escritor, handle, ruta)

    def leer_parquet(
        self,
        ruta: str,
        columnas: list[str] | None = None,
        filtros: list | None = None,
    ) -> pd.DataFrame:
        """Lee un parquet o un dataset particionado completo."""
        import pyarrow.parquet as pq

        dataset = pq.ParquetDataset(ruta, filesystem=self.fs, filters=filtros)
        tabla = dataset.read(columns=columnas)
        return tabla.to_pandas()

    def contar_filas_parquet(self, ruta: str) -> int:
        import pyarrow.parquet as pq

        dataset = pq.ParquetDataset(ruta, filesystem=self.fs)
        return sum(fragmento.count_rows() for fragmento in dataset.fragments)


class _EscritorLotes:
    """Contexto que cierra el ``ParquetWriter`` y el handle del sistema de archivos."""

    def __init__(self, escritor: pq.ParquetWriter, handle, ruta: str) -> None:
        self._escritor = escritor
        self._handle = handle
        self.ruta = ruta
        self.filas = 0

    def escribir(self, lote: pa.RecordBatch | pa.Table) -> None:
        import pyarrow as pa

        tabla = pa.Table.from_batches([lote]) if isinstance(lote, pa.RecordBatch) else lote
        self._escritor.write_table(tabla)
        self.filas += tabla.num_rows

    def __enter__(self) -> _EscritorLotes:
        return self

    def __exit__(self, *excepcion) -> None:
        self.cerrar()

    def cerrar(self) -> None:
        self._escritor.close()
        self._handle.close()
        log.info("Parquet escrito por lotes: %s (%d filas)", self.ruta, self.filas)


def get_storage(settings: Settings | None = None) -> Storage:
    return Storage(settings)
