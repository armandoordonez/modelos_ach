"""Tests de la configuración por variables de entorno."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from common.config import Settings


def test_valores_por_defecto_apuntan_al_stack_local():
    ajustes = Settings(_env_file=None)
    assert ajustes.bucket_raw == "raw"
    assert ajustes.bucket_curated == "curated"
    assert ajustes.bucket_results == "results"
    assert ajustes.lineage == "cedula-v1", "producción debe cruzar por cédula"


def test_lee_variables_de_entorno(monkeypatch):
    monkeypatch.setenv("ACH_BUCKET_RAW", "crudo")
    monkeypatch.setenv("ACH_SEED", "7")
    monkeypatch.setenv("ACH_LINEAGE", "legacy")
    ajustes = Settings(_env_file=None)
    assert ajustes.bucket_raw == "crudo"
    assert ajustes.seed == 7
    assert ajustes.lineage == "legacy"


def test_endpoint_vacio_activa_modo_local():
    assert Settings(_env_file=None, s3_endpoint="").usa_s3 is False
    assert Settings(_env_file=None, s3_endpoint="http://minio:9000").usa_s3 is True


def test_endpoint_se_normaliza_sin_barra_final():
    assert Settings(_env_file=None, s3_endpoint="http://minio:9000/").s3_endpoint == "http://minio:9000"


@pytest.mark.parametrize("periodo", ["2025", "2025-1", "2025-13", "enero", "25-01"])
def test_periodo_invalido_falla(periodo):
    with pytest.raises(ValidationError):
        Settings(_env_file=None, window_start=periodo)


def test_conteo_de_meses_de_la_ventana():
    ajustes = Settings(_env_file=None, window_start="2025-01", window_end="2026-06")
    assert ajustes.meses_ventana() == 18
    assert ajustes.ventana == ("2025-01", "2026-06")


def test_linaje_invalido_falla():
    with pytest.raises(ValidationError):
        Settings(_env_file=None, lineage="lo-que-sea")


def test_no_hay_credenciales_por_defecto():
    ajustes = Settings(_env_file=None)
    assert ajustes.s3_access_key == ""
    assert ajustes.s3_secret_key == ""
