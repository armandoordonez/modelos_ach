"""Fixtures compartidos. Los tests corren contra el sistema de archivos local,
sin MinIO ni Docker: el cliente de storage se comporta igual en ambos casos."""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import pytest

RAIZ_JOBS = Path(__file__).resolve().parents[1] / "jobs"
if str(RAIZ_JOBS) not in sys.path:
    sys.path.insert(0, str(RAIZ_JOBS))

from common.config import Settings  # noqa: E402
from common.storage import Storage  # noqa: E402


@pytest.fixture
def settings(tmp_path: Path) -> Settings:
    """Configuración aislada apuntando a un directorio temporal."""
    return Settings(
        s3_endpoint="",
        local_root=str(tmp_path / "almacen"),
        bucket_raw="raw",
        bucket_curated="curated",
        bucket_results="results",
        lineage="cedula-v1",
        window_start="2025-01",
        window_end="2026-06",
        seed=42,
        run_id="test-run",
    )


@pytest.fixture
def storage(settings: Settings) -> Storage:
    return Storage(settings)


@pytest.fixture
def ss_df() -> pd.DataFrame:
    """Muestra mínima de Seguridad Social con todas las columnas del diccionario."""
    filas = [
        {
            "Nombre": "***ANA PEREZ", "Tipo de documento": "P", "Número de documento": "10000001****",
            "Promedio salario básico": 1500000, "Promedio ingreso base pensión": 0,
            "Promedio ingreso base salud": 1500000, "Promedio ingreso base caja compensación": 0,
            "Promedio ingreso base riesgos": 1500000, "Razón social": "EMPRESA********",
            "Tipo documento": "J", "Número documento": "90000001****",
            "Código actividad económica": 4771, "Actividad económica": "Comercio",
            "Clase aportante": "EMPRESAS CON MENOS DE 200 COTIZANTES", "Tipo aportante": "EMPLEADOR",
            "Fecha de pago": "2025-02-10", "Periodo cotización": "2025-01", "Empleador": "EMPRESA********",
            "Tipo planilla": "E", "Relación laboral": "DEPENDIENTE", "Salario básico": 1500000,
            "Tipo salario": "F", "Novedades": None, "Ingreso base pensión": 1500000, "Días pensión": 30,
            "Ingreso base salud": 1500000, "Días salud": 30, "Ingreso base caja compensación": 1500000,
            "Días caja compensación": 30, "Ingreso base riesgos": 1500000, "Días riesgos": 30,
        },
        {
            "Nombre": "***LUIS GOMEZ", "Tipo de documento": "P", "Número de documento": "10000002****",
            "Promedio salario básico": 0, "Promedio ingreso base pensión": 0,
            "Promedio ingreso base salud": 900000, "Promedio ingreso base caja compensación": 0,
            "Promedio ingreso base riesgos": 0, "Razón social": "ADMINISTRADORA COLOMBIANA DE PENSIONES COLP********",
            "Tipo documento": "J", "Número documento": "90000002****",
            "Código actividad económica": 6531, "Actividad económica": "Régimen de prima media",
            "Clase aportante": "EMPRESAS CON MAS DE 200 COTIZANTES", "Tipo aportante": "PAGADOR DE PENSIONES",
            "Fecha de pago": "2025-02-11", "Periodo cotización": "2025-01", "Empleador": "COLP********",
            "Tipo planilla": "P", "Relación laboral": "Pensionado de regimen de prima media con tope maximo 25 SMLMV",
            "Salario básico": 0, "Tipo salario": None, "Novedades": None, "Ingreso base pensión": 0,
            "Días pensión": 0, "Ingreso base salud": 900000, "Días salud": 30,
            "Ingreso base caja compensación": 0, "Días caja compensación": 0,
            "Ingreso base riesgos": 0, "Días riesgos": 0,
        },
    ]
    return pd.DataFrame(filas)


@pytest.fixture
def trf_df() -> pd.DataFrame:
    """Muestra de Transferencias con un recibido externo, un enviado y una cuenta propia."""
    base = {
        "Nombre": "***ANA PEREZ", "Tipo documento": "P", "Número documento": "10000001****",
        "Promedio ACH recibidas": 0, "Promedio Transfiya recibidas": 0,
        "Promedio ACH enviadas": 0, "Promedio Transfiya enviadas": 0,
        "Periodo": "2025-01", "Total recibidas periodo": 0.0, "Cantidad recibidas periodo": 0,
        "Valor promedio recibidas periodo": 0.0, "Total enviadas periodo": 0.0,
        "Cantidad enviadas periodo": 0, "Valor promedio enviadas periodo": 0.0,
        "Tipo de cuenta": "AHORROS", "Entidad origen": "BAN********", "Usuario originador": "9********",
        "Nombre usuario originador": "EMPRESA********", "Entidad destino": None,
        "Usuario receptor": None, "Nombre usuario receptor": None,
    }
    filas = [
        {**base, "Servicio": "ACH", "Tipo registro": "RECIBIDO", "Cuenta propia": "NO",
         "Cantidad de transferencias": 1, "Valor": 1_000_000.0},
        {**base, "Servicio": "TRANSFIYA", "Tipo registro": "RECIBIDO", "Cuenta propia": "NO",
         "Cantidad de transferencias": 2, "Valor": 200_000.0},
        {**base, "Servicio": "ACH", "Tipo registro": "ENVIADO", "Cuenta propia": "NO",
         "Cantidad de transferencias": 1, "Valor": 300_000.0},
        {**base, "Servicio": "ACH", "Tipo registro": "RECIBIDO", "Cuenta propia": "SI",
         "Cantidad de transferencias": 1, "Valor": 5_000_000.0},
    ]
    return pd.DataFrame(filas)


@pytest.fixture
def pse_df() -> pd.DataFrame:
    """Muestra de Pagos Digitales con comercios de categorías distintas."""
    base = {
        "Nombre": "***ANA PEREZ", "Tipo documento": "P", "Número documento": "10000001****",
        "Promedio pagos digitales": 0, "Periodo": "2025-01", "Total pagos periodo": 0.0,
        "Cantidad pagos periodo": 0, "Valor promedio periodo": 0.0,
        "Entidad autorizadora": "BANCOLOM********",
    }
    filas = [
        {**base, "Comercio": "BANCOLOM********", "Cantidad": 1, "Valor": 250_000.0},
        {**base, "Comercio": "CAJA COLOMBIANA DE SUBSIDIO FAMILIAR COL********", "Cantidad": 1, "Valor": 80_000.0},
        {**base, "Comercio": "COMUNICACION CELULAR COMC********", "Cantidad": 2, "Valor": 65_000.0},
        {**base, "Periodo": "2025-02", "Comercio": "ALMACENES EXITO-PRO********", "Cantidad": 1, "Valor": 120_000.0},
    ]
    return pd.DataFrame(filas)


@pytest.fixture
def curated(storage: Storage, ss_df, trf_df, pse_df) -> Storage:
    """Escribe un dataset curado mínimo con las tres fuentes particionadas."""
    from common.features import RUTA_CURADO
    from common.schema import COLUMNA_PERIODO, aplicar_tipos

    ruta = storage.ruta(storage.settings.bucket_curated, RUTA_CURADO)
    for fuente, df in (("ss", ss_df), ("trf", trf_df), ("pse", pse_df)):
        tipado = aplicar_tipos(df, fuente)
        tipado["fuente"] = fuente
        tipado["periodo"] = tipado[COLUMNA_PERIODO[fuente]].astype("string").str.slice(0, 7)
        storage.escribir_parquet(tipado, ruta, particiones=["fuente", "periodo"])
    return storage
