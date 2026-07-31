"""Job de procesamiento: XLSX crudos → dataset curado en parquet.

    python -m processing.main --fecha 2026-07-30

Qué hace:

1. Localiza los XLSX en ``raw/<fecha>/`` y los baja a disco temporal.
2. Los lee por lotes (streaming), valida contra el diccionario de datos y aplica tipos.
3. Elimina filas duplicadas exactas — las mismas que quitaban los notebooks originales,
   para no alterar las métricas.
4. Escribe ``curated/dataset.parquet`` particionado por fuente y periodo.
5. Deja un ``_manifest.json`` con filas, columnas, tipos y hash de contenido.

Falla temprano: si un archivo no cumple el diccionario de datos, el job aborta con el
detalle de qué columna falla y por qué, antes de escribir nada en curated.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import shutil
import sys
import tempfile
from datetime import UTC, datetime
from pathlib import Path

import pandas as pd

from common.config import Settings, get_settings
from common.features import ARCHIVO_MANIFIESTO, RUTA_CURADO
from common.logging_config import configurar_logging
from common.schema import (
    ARCHIVO_FUENTE,
    COLUMNA_PERIODO,
    DICCIONARIO,
    NOMBRE_FUENTE,
    ErrorDeEsquema,
    esquema_arrow,
    nombres_columnas,
    validar,
)
from common.storage import Storage, get_storage
from processing.lector_excel import TAMANO_LOTE, leer_encabezado, leer_por_lotes

log = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Localización de los archivos crudos                                          #
# --------------------------------------------------------------------------- #
def resolver_fecha(storage: Storage, fecha: str | None) -> str:
    """Fecha de la carga a procesar. Sin argumento, toma la más reciente de raw/."""
    if fecha:
        return fecha
    raiz = storage.ruta(storage.settings.bucket_raw)
    if not storage.existe(raiz):
        raise FileNotFoundError(
            f"No existe el bucket de entrada en {raiz}. Sube los XLSX con 'make seed'."
        )
    carpetas = sorted({
        objeto[len(raiz):].strip("/").split("/")[0]
        for objeto in storage.listar(raiz)
        if objeto.startswith(raiz)
    })
    carpetas = [c for c in carpetas if c]
    if not carpetas:
        raise FileNotFoundError(f"No hay cargas en {raiz}. Sube los XLSX con 'make seed'.")
    elegida = carpetas[-1]
    log.info("Fecha de carga resuelta automáticamente: %s", elegida)
    return elegida


def descargar(storage: Storage, fecha: str, fuente: str, destino: Path) -> Path:
    """Baja el XLSX de una fuente a disco local para poder leerlo en streaming."""
    nombre = ARCHIVO_FUENTE[fuente]
    remoto = storage.ruta(storage.settings.bucket_raw, fecha, nombre)
    if not storage.existe(remoto):
        raise FileNotFoundError(
            f"Falta el archivo de {NOMBRE_FUENTE[fuente]} en {remoto}. "
            f"Se esperaba el nombre exacto {nombre!r}."
        )
    local = destino / nombre
    log.info("Descargando %s", remoto)
    with storage.fs.open(remoto, "rb") as origen, open(local, "wb") as copia:
        shutil.copyfileobj(origen, copia, length=8 * 1024 * 1024)
    log.info("%s: %.1f MB en disco temporal", nombre, local.stat().st_size / 1e6)
    return local


# --------------------------------------------------------------------------- #
# Procesamiento de una fuente                                                  #
# --------------------------------------------------------------------------- #
def _hash_filas(df: pd.DataFrame) -> pd.Series:
    """Huella de contenido por fila. Dos filas iguales dan la misma huella, que es lo
    que permite deduplicar en streaming sin tener el dataset completo en memoria.
    La probabilidad de colisión con 64 bits y ~10^6 filas es del orden de 10^-8."""
    return pd.util.hash_pandas_object(df, index=False)


def procesar_fuente(
    fuente: str,
    ruta_local: Path,
    storage: Storage,
    settings: Settings,
    tamano_lote: int = TAMANO_LOTE,
) -> dict:
    """Lee, valida y escribe una fuente. Devuelve su bloque del manifiesto."""
    nombre_legible = NOMBRE_FUENTE[fuente]
    log.info("── %s ──", nombre_legible)

    # Chequeo de estructura antes de leer una sola fila de datos: si el archivo no es
    # el que esperamos, el job debe morir aquí y no a los diez minutos.
    encabezado = leer_encabezado(ruta_local)
    faltantes = [c for c in nombres_columnas(fuente) if c not in encabezado]
    if faltantes:
        raise ErrorDeEsquema(
            f"[{nombre_legible}] el archivo {ruta_local.name} no cumple el diccionario de datos: "
            f"faltan {len(faltantes)} columnas {faltantes}. Encabezado recibido: {encabezado}"
        )

    ruta_destino = storage.ruta(settings.bucket_curated, RUTA_CURADO)
    columnas = nombres_columnas(fuente)
    columna_periodo = COLUMNA_PERIODO[fuente]
    esquema = esquema_arrow(fuente)

    vistos: set[int] = set()
    filas_leidas = filas_duplicadas = filas_escritas = 0
    periodos: dict[str, int] = {}
    acumulado: list[pd.DataFrame] = []

    for lote in leer_por_lotes(ruta_local, tamano_lote):
        filas_leidas += len(lote)
        lote = lote[columnas]
        tipado = validar(lote, fuente)

        # Se descartan tanto los duplicados dentro del propio lote como los que ya
        # aparecieron en lotes anteriores.
        huellas = _hash_filas(tipado)
        nuevas = ~huellas.duplicated() & ~huellas.isin(vistos)
        filas_duplicadas += int((~nuevas).sum())
        vistos.update(huellas[nuevas].tolist())
        tipado = tipado.loc[nuevas.to_numpy()]
        if tipado.empty:
            continue

        tipado = tipado.assign(
            fuente=fuente,
            periodo=tipado[columna_periodo].astype("string").str.slice(0, 7),
        )
        for periodo, cuenta in tipado["periodo"].value_counts().items():
            periodos[str(periodo)] = periodos.get(str(periodo), 0) + int(cuenta)
        filas_escritas += len(tipado)
        acumulado.append(tipado)

        # Se descarga a parquet cada ~4 lotes para acotar el pico de memoria.
        if sum(len(p) for p in acumulado) >= tamano_lote * 4:
            storage.escribir_parquet(pd.concat(acumulado, ignore_index=True),
                                     ruta_destino, particiones=["fuente", "periodo"])
            acumulado = []

    if acumulado:
        storage.escribir_parquet(pd.concat(acumulado, ignore_index=True),
                                 ruta_destino, particiones=["fuente", "periodo"])

    if filas_escritas == 0:
        raise ErrorDeEsquema(f"[{nombre_legible}] no quedó ninguna fila después de validar.")

    contenido = hashlib.sha256()
    for huella in sorted(vistos):
        contenido.update(str(huella).encode("ascii"))

    log.info("%s: %s filas escritas, %s duplicadas exactas descartadas, %d periodos",
             nombre_legible, f"{filas_escritas:,}", f"{filas_duplicadas:,}", len(periodos))

    return {
        "fuente": fuente,
        "nombre": nombre_legible,
        "archivo_origen": ruta_local.name,
        "filas_leidas": filas_leidas,
        "filas_duplicadas_descartadas": filas_duplicadas,
        "filas": filas_escritas,
        "columnas": len(columnas),
        "tipos": {c.nombre: c.tipo for c in DICCIONARIO[fuente]},
        "periodos": dict(sorted(periodos.items())),
        "rango_periodos": [min(periodos), max(periodos)] if periodos else [],
        "truncado_en_excel": filas_leidas >= 1_048_573,
        "hash_contenido": contenido.hexdigest(),
        "esquema_arrow": str(esquema.types),
    }


# --------------------------------------------------------------------------- #
# Orquestación                                                                 #
# --------------------------------------------------------------------------- #
def ejecutar(
    fecha: str | None = None,
    fuentes: list[str] | None = None,
    tamano_lote: int = TAMANO_LOTE,
    settings: Settings | None = None,
) -> dict:
    """Corre el procesamiento completo y devuelve el manifiesto."""
    settings = settings or get_settings()
    storage = get_storage(settings)
    fuentes = fuentes or list(DICCIONARIO)
    inicio = datetime.now(UTC)

    fecha_resuelta = resolver_fecha(storage, fecha)
    destino = storage.ruta(settings.bucket_curated, RUTA_CURADO)

    # El curado se reescribe completo en cada corrida: es una foto del extracto,
    # no un histórico incremental.
    log.info("Limpiando %s", destino)
    storage.borrar(destino, recursivo=True)

    bloques = []
    with tempfile.TemporaryDirectory(prefix="ach_raw_") as temporal:
        carpeta = Path(temporal)
        for fuente in fuentes:
            local = descargar(storage, fecha_resuelta, fuente, carpeta)
            bloques.append(procesar_fuente(fuente, local, storage, settings, tamano_lote))
            local.unlink(missing_ok=True)

    global_hash = hashlib.sha256(
        "".join(b["hash_contenido"] for b in bloques).encode("ascii")
    ).hexdigest()

    manifiesto = {
        "schema_version": "1.0",
        "generado_en": datetime.now(UTC).isoformat(),
        "duracion_segundos": round((datetime.now(UTC) - inicio).total_seconds(), 2),
        "run_id": settings.run_id,
        "fecha_carga": fecha_resuelta,
        "uri": destino,
        "lineage": settings.lineage,
        "ventana_configurada": list(settings.ventana),
        "filas": sum(b["filas"] for b in bloques),
        "columnas": sum(b["columnas"] for b in bloques),
        "hash": global_hash,
        "fuentes": bloques,
    }

    ruta_manifiesto = storage.ruta(settings.bucket_curated, RUTA_CURADO, ARCHIVO_MANIFIESTO)
    storage.escribir_json(manifiesto, ruta_manifiesto)

    log.info("Curado listo: %s filas totales · hash %s", f"{manifiesto['filas']:,}", global_hash[:12])
    return manifiesto


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="processing", description="Convierte los XLSX crudos de ACH en el dataset curado en parquet.")
    parser.add_argument("--fecha", default=None,
                        help="Carpeta de raw/ a procesar (YYYY-MM-DD). Por defecto, la más reciente.")
    parser.add_argument("--fuentes", nargs="*", choices=sorted(DICCIONARIO), default=None,
                        help="Fuentes a procesar. Por defecto, las tres.")
    parser.add_argument("--tamano-lote", type=int, default=TAMANO_LOTE,
                        help="Filas por lote de lectura. Baja este número si la memoria aprieta.")
    args = parser.parse_args(argv)

    configurar_logging()
    try:
        ejecutar(fecha=args.fecha, fuentes=args.fuentes, tamano_lote=args.tamano_lote)
    except (ErrorDeEsquema, FileNotFoundError) as error:
        log.error("Procesamiento abortado: %s", error)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
