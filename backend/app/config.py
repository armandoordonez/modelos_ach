"""Configuración del backend.

Extiende la configuración común de la pipeline con lo propio de la API. Igual que en
los jobs: todo por variables de entorno, ningún secreto en el código.
"""

from __future__ import annotations

from functools import lru_cache

from pydantic import Field
from pydantic_settings import SettingsConfigDict

from common.config import Settings as SettingsPipeline


class Settings(SettingsPipeline):
    """Configuración de la API. Hereda buckets y credenciales de la pipeline."""

    model_config = SettingsConfigDict(
        env_prefix="ACH_", env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    cors_origins: str = Field(
        default="http://localhost:5173",
        description="Orígenes permitidos para el navegador, separados por coma.",
    )
    cache_ttl: float = Field(default=30.0, ge=0, description="Segundos de vida de la caché.")
    url_prefirmada_ttl: int = Field(
        default=300, ge=30,
        description="Vida de las URL prefirmadas de artefactos, en segundos.",
    )
    api_titulo: str = Field(default="ACH · API de modelos")

    @property
    def origenes_cors(self) -> list[str]:
        return [o.strip() for o in self.cors_origins.split(",") if o.strip()]


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
