"""Tests del cliente de almacenamiento."""

from __future__ import annotations

import pandas as pd
import pytest

from common.config import Settings
from common.storage import Storage


def test_ruta_local_incluye_la_raiz(settings: Settings):
    almacen = Storage(settings)
    ruta = almacen.ruta("curated", "dataset.parquet", "_manifest.json")
    assert ruta.endswith("curated/dataset.parquet/_manifest.json")
    assert settings.local_root in ruta


def test_ruta_s3_no_incluye_raiz_local():
    ajustes = Settings(_env_file=None, s3_endpoint="http://minio:9000", local_root="/no/usar")
    ruta = Storage(ajustes).ruta("curated", "dataset.parquet")
    assert ruta == "curated/dataset.parquet"


def test_json_ida_y_vuelta(storage: Storage):
    ruta = storage.ruta("results", "modelo", "run.json")
    storage.escribir_json({"metricas": {"silueta": 0.27}, "acentos": "ñá"}, ruta)
    assert storage.existe(ruta)
    assert storage.leer_json(ruta)["metricas"]["silueta"] == 0.27
    assert storage.leer_json(ruta)["acentos"] == "ñá"


def test_parquet_ida_y_vuelta(storage: Storage):
    df = pd.DataFrame({"a": [1, 2, 3], "b": ["x", "y", "z"]})
    ruta = storage.ruta("curated", "simple.parquet")
    storage.escribir_parquet(df, ruta)
    assert storage.leer_parquet(ruta).equals(df)


def test_parquet_particionado_se_lee_filtrado(storage: Storage):
    df = pd.DataFrame({"fuente": ["ss", "ss", "pse"], "valor": [1, 2, 3]})
    ruta = storage.ruta("curated", "particionado.parquet")
    storage.escribir_parquet(df, ruta, particiones=["fuente"])

    completo = storage.leer_parquet(ruta)
    assert len(completo) == 3

    solo_ss = storage.leer_parquet(ruta, filtros=[("fuente", "==", "ss")])
    assert len(solo_ss) == 2
    assert set(solo_ss["valor"]) == {1, 2}


def test_listar_devuelve_vacio_si_no_existe(storage: Storage):
    assert storage.listar(storage.ruta("results", "inexistente")) == []


def test_escritura_por_lotes_acumula_filas(storage: Storage):
    import pyarrow as pa

    esquema = pa.schema([pa.field("n", pa.int64())])
    ruta = storage.ruta("curated", "lotes.parquet")
    with storage.escritor_parquet(ruta, esquema) as escritor:
        for inicio in range(0, 30, 10):
            escritor.escribir(pa.RecordBatch.from_pydict({"n": list(range(inicio, inicio + 10))}, esquema))
        assert escritor.filas == 30

    assert storage.contar_filas_parquet(ruta) == 30
    assert list(storage.leer_parquet(ruta)["n"]) == list(range(30))


def test_borrar_es_idempotente(storage: Storage):
    ruta = storage.ruta("results", "efimero.json")
    storage.escribir_json({"a": 1}, ruta)
    storage.borrar(ruta)
    storage.borrar(ruta)  # no debe fallar la segunda vez
    assert not storage.existe(ruta)


@pytest.mark.parametrize("bucket", ["raw", "curated", "results"])
def test_los_tres_buckets_son_direccionables(storage: Storage, bucket: str):
    ruta = storage.ruta(bucket, "prueba.json")
    storage.escribir_json({"bucket": bucket}, ruta)
    assert storage.leer_json(ruta)["bucket"] == bucket
