"""Configuración de la pipeline, leída exclusivamente de variables de entorno.

Ningún valor sensible vive en el código: las credenciales llegan por entorno y los
valores por defecto son los del stack local de desarrollo (MinIO en la red del compose).
"""

from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, field_validator
from pydantic_settings import BaseSettings, SettingsConfigDict

Lineage = Literal["cedula-v1", "legacy"]

# Estrategias de llave de persona. Vive aquí y no en features.py para que el backend
# pueda leer el registro de modelos sin arrastrar pandas ni scikit-learn.
EstrategiaLlave = Literal["cedula", "nombre_documento", "nombre_normalizado_documento_visible"]

FUENTES = ("ss", "trf", "pse")


class Settings(BaseSettings):
    """Configuración de la pipeline.

    Todas las variables usan el prefijo ``ACH_``. Ejemplo: ``ACH_S3_ENDPOINT``.
    """

    model_config = SettingsConfigDict(
        env_prefix="ACH_",
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # --- Almacenamiento ----------------------------------------------------
    s3_endpoint: str = Field(
        default="http://minio:9000",
        description="Endpoint S3. Vacío = modo sistema de archivos local (tests).",
    )
    s3_access_key: str = Field(default="", description="Access key de MinIO/AWS.")
    s3_secret_key: str = Field(default="", description="Secret key de MinIO/AWS.")
    s3_region: str = Field(default="us-east-1", description="Región S3.")
    local_root: str = Field(
        default=".localstore",
        description="Raíz en disco cuando no hay endpoint S3 (desarrollo y tests).",
    )

    bucket_raw: str = Field(default="raw", description="Bucket de los XLSX crudos.")
    bucket_curated: str = Field(default="curated", description="Bucket del parquet curado.")
    bucket_results: str = Field(default="results", description="Bucket de los JSON de métricas.")

    # --- Linaje y ventana de análisis --------------------------------------
    lineage: Lineage = Field(
        default="cedula-v1",
        description=(
            "Estrategia de llave de persona. 'cedula-v1' cruza por cédula ofuscada (producción); "
            "'legacy' reproduce las llaves compuestas de los scripts originales (solo paridad)."
        ),
    )
    window_start: str = Field(default="2025-01", description="Primer periodo YYYY-MM de la ventana común.")
    window_end: str = Field(default="2026-06", description="Último periodo YYYY-MM de la ventana común.")

    # --- Ejecución ---------------------------------------------------------
    seed: int = Field(default=42, description="Semilla global de reproducibilidad.")
    run_id: str = Field(default="local", description="Identificador de la corrida (lo inyecta Airflow).")
    log_level: str = Field(default="INFO", description="Nivel de logging.")

    @field_validator("window_start", "window_end")
    @classmethod
    def _validar_periodo(cls, valor: str) -> str:
        partes = valor.split("-")
        if len(partes) != 2 or len(partes[0]) != 4 or len(partes[1]) != 2:
            raise ValueError(f"El periodo debe tener formato YYYY-MM, se recibió {valor!r}")
        if not (partes[0].isdigit() and partes[1].isdigit() and 1 <= int(partes[1]) <= 12):
            raise ValueError(f"El periodo debe tener formato YYYY-MM válido, se recibió {valor!r}")
        return valor

    @field_validator("s3_endpoint")
    @classmethod
    def _normalizar_endpoint(cls, valor: str) -> str:
        return valor.strip().rstrip("/")

    @property
    def usa_s3(self) -> bool:
        """True cuando hay endpoint configurado; False = disco local."""
        return bool(self.s3_endpoint)

    @property
    def ventana(self) -> tuple[str, str]:
        return (self.window_start, self.window_end)

    def meses_ventana(self) -> int:
        """Cantidad de meses que cubre la ventana, ambos extremos incluidos."""
        anio_i, mes_i = (int(p) for p in self.window_start.split("-"))
        anio_f, mes_f = (int(p) for p in self.window_end.split("-"))
        return (anio_f - anio_i) * 12 + (mes_f - mes_i) + 1


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Configuración cacheada del proceso."""
    return Settings()


def reset_settings_cache() -> None:
    """Limpia la caché de configuración. Solo lo usan los tests."""
    get_settings.cache_clear()
