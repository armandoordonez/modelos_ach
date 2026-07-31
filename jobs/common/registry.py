"""Registro de modelos: lee ``models_config.yml``, la única fuente de verdad.

Agregar un modelo es crear su módulo y añadir una entrada al YAML. Ni el DAG ni el
backend ni el frontend se enteran: todos leen de aquí.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel, Field

from .features import EstrategiaLlave
from .results import TaskType

NOMBRE_ARCHIVO = "models_config.yml"


class ModeloConfig(BaseModel):
    """Una entrada del registro de modelos."""

    id: str
    nombre: str
    catalogo: str = ""
    caso_uso: int = Field(ge=1, le=5)
    task_type: TaskType
    modulo: str
    legacy_key: EstrategiaLlave = "cedula"
    params: dict[str, Any] = Field(default_factory=dict)

    @property
    def comando(self) -> list[str]:
        """Comando que ejecuta este modelo dentro de la imagen de jobs."""
        return ["python", "-m", self.modulo, "--model-id", self.id]


class RegistroModelos(BaseModel):
    schema_version: str = "1.0"
    modelos: list[ModeloConfig]

    def ids(self) -> list[str]:
        return [m.id for m in self.modelos]

    def obtener(self, model_id: str) -> ModeloConfig:
        for modelo in self.modelos:
            if modelo.id == model_id:
                return modelo
        raise KeyError(
            f"El modelo {model_id!r} no está en {NOMBRE_ARCHIVO}. Registrados: {self.ids()}"
        )


def ruta_config() -> Path:
    """Ubicación del registro. ``ACH_MODELS_CONFIG`` la puede sobrescribir."""
    desde_entorno = os.environ.get("ACH_MODELS_CONFIG")
    if desde_entorno:
        return Path(desde_entorno)
    return Path(__file__).resolve().parents[1] / NOMBRE_ARCHIVO


def cargar_registro(ruta: Path | str | None = None) -> RegistroModelos:
    """Carga y valida el registro de modelos."""
    destino = Path(ruta) if ruta else ruta_config()
    if not destino.exists():
        raise FileNotFoundError(
            f"No se encontró el registro de modelos en {destino}. "
            "Define ACH_MODELS_CONFIG o deja el archivo junto al paquete."
        )
    contenido = yaml.safe_load(destino.read_text(encoding="utf-8"))
    registro = RegistroModelos.model_validate(contenido)

    duplicados = {i for i in registro.ids() if registro.ids().count(i) > 1}
    if duplicados:
        raise ValueError(f"Hay ids de modelo repetidos en {destino}: {sorted(duplicados)}")
    return registro


@lru_cache(maxsize=1)
def registro() -> RegistroModelos:
    return cargar_registro()


def obtener_modelo(model_id: str) -> ModeloConfig:
    return registro().obtener(model_id)
