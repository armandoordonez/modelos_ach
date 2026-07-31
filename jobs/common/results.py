"""Contrato del JSON de salida de los modelos.

Es la frontera entre la pipeline y el backend/frontend: mientras un modelo respete
este esquema, el tablero lo renderiza sin cambios de código. Por eso ``metrics`` es
un diccionario plano de número → valor y ``charts`` es opcional y por bloques: si un
modelo no reporta un bloque, el frontend simplemente no dibuja esa sección.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator

ESQUEMA_VERSION = "1.0"

TaskType = Literal["clustering", "regression", "classification", "scoring"]
Estado = Literal["success", "failed"]


def ahora_utc() -> datetime:
    return datetime.now(timezone.utc)


class DatasetInfo(BaseModel):
    """De qué datos salió este resultado."""

    uri: str = Field(description="Ruta del dataset curado que consumió el modelo")
    manifest_hash: str = Field(default="", description="Hash del dataset, tomado del _manifest.json")
    rows: int = Field(default=0, ge=0, description="Filas del dataset leído")
    window: tuple[str, str] = Field(description="Ventana de análisis (YYYY-MM, YYYY-MM)")
    lineage: str = Field(default="cedula-v1", description="Estrategia de llave de persona")


class Segmento(BaseModel):
    """Un grupo de una segmentación. Solo aplica a modelos de clustering."""

    id: int
    label: str
    n: int = Field(ge=0)
    share: float = Field(ge=0, le=1)
    profile: dict[str, float] = Field(default_factory=dict)


class CurvaROC(BaseModel):
    fpr: list[float]
    tpr: list[float]
    auc: float


class MatrizConfusion(BaseModel):
    labels: list[str]
    matrix: list[list[int]]


class ImportanciaVariable(BaseModel):
    feature: str
    importance: float


class PuntoScatter(BaseModel):
    x: float
    y: float
    label: str | None = None


class Scatter2D(BaseModel):
    points: list[PuntoScatter]
    explained_variance: float | None = None
    x_label: str = "PC1"
    y_label: str = "PC2"


class PuntoSeleccionK(BaseModel):
    k: int
    silhouette: float | None = None
    inertia: float | None = None
    davies_bouldin: float | None = None


class Charts(BaseModel):
    """Bloques opcionales de visualización.

    ``extra="allow"`` deja que un modelo nuevo publique un bloque que hoy no existe
    sin romper el contrato; el frontend ignora lo que no sabe dibujar.
    """

    model_config = ConfigDict(extra="allow")

    segment_distribution: list[Segmento] | None = None
    k_selection: list[PuntoSeleccionK] | None = None
    scatter_2d: Scatter2D | None = None
    feature_importance: list[ImportanciaVariable] | None = None
    roc_curve: CurvaROC | None = None
    confusion_matrix: MatrizConfusion | None = None
    residuals: dict[str, list[float]] | None = None
    distribution: dict[str, list[float]] | None = None


class ModelResult(BaseModel):
    """JSON que cada job de modelo escribe en ``results/<model_id>/<run_id>.json``."""

    model_config = ConfigDict(populate_by_name=True)

    schema_version: str = ESQUEMA_VERSION
    model_id: str = Field(description="Identificador estable del modelo")
    model_name: str = Field(description="Nombre legible para el tablero")
    catalog_ref: str = Field(default="", description="Número del modelo en el catálogo ACH, ej. '#101'")
    use_case: int = Field(ge=1, le=5, description="Caso de uso del proyecto")
    task_type: TaskType

    run_id: str = Field(description="Identificador de la corrida; es el dag_run_id de Airflow")
    status: Estado = "success"
    started_at: datetime
    finished_at: datetime
    duration_seconds: float = Field(ge=0)

    dataset: DatasetInfo
    params: dict[str, Any] = Field(default_factory=dict)
    metrics: dict[str, float] = Field(default_factory=dict)
    segments: list[Segmento] = Field(default_factory=list)
    charts: Charts = Field(default_factory=Charts)
    artifacts: dict[str, str] = Field(default_factory=dict)
    notes: list[str] = Field(default_factory=list)
    error: str | None = None

    @field_validator("metrics")
    @classmethod
    def _metricas_finitas(cls, valor: dict[str, float]) -> dict[str, float]:
        import math

        limpias: dict[str, float] = {}
        for nombre, dato in valor.items():
            numero = float(dato)
            if math.isnan(numero) or math.isinf(numero):
                raise ValueError(
                    f"La métrica {nombre!r} vale {dato!r}: el JSON de resultados no admite NaN ni infinitos. "
                    "Reemplázala por un valor real o quítala del reporte."
                )
            limpias[nombre] = numero
        return limpias

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


class EntradaIndice(BaseModel):
    """Resumen de un modelo dentro de ``results/index.json``."""

    model_id: str
    model_name: str
    catalog_ref: str = ""
    use_case: int
    task_type: TaskType
    status: Estado
    run_id: str
    finished_at: datetime
    duration_seconds: float
    metrics: dict[str, float] = Field(default_factory=dict)
    latest_uri: str = ""
    error: str | None = None


class IndiceResultados(BaseModel):
    """``results/index.json``: lo que el backend lee para listar modelos sin recorrer el bucket."""

    schema_version: str = ESQUEMA_VERSION
    run_id: str
    generated_at: datetime = Field(default_factory=ahora_utc)
    total_models: int = 0
    successful: int = 0
    failed: int = 0
    models: list[EntradaIndice] = Field(default_factory=list)

    def to_json_dict(self) -> dict[str, Any]:
        return self.model_dump(mode="json", exclude_none=True)


def construir_resultado(
    *,
    model_id: str,
    model_name: str,
    catalog_ref: str,
    use_case: int,
    task_type: TaskType,
    run_id: str,
    started_at: datetime,
    dataset: DatasetInfo,
    params: dict[str, Any] | None = None,
    metrics: dict[str, float] | None = None,
    segments: list[Segmento] | None = None,
    charts: Charts | None = None,
    artifacts: dict[str, str] | None = None,
    notes: list[str] | None = None,
) -> ModelResult:
    """Arma un resultado exitoso calculando la duración."""
    finished = ahora_utc()
    return ModelResult(
        model_id=model_id,
        model_name=model_name,
        catalog_ref=catalog_ref,
        use_case=use_case,
        task_type=task_type,
        run_id=run_id,
        status="success",
        started_at=started_at,
        finished_at=finished,
        duration_seconds=round((finished - started_at).total_seconds(), 3),
        dataset=dataset,
        params=params or {},
        metrics=metrics or {},
        segments=segments or [],
        charts=charts or Charts(),
        artifacts=artifacts or {},
        notes=notes or [],
    )


def construir_fallo(
    *,
    model_id: str,
    model_name: str,
    catalog_ref: str,
    use_case: int,
    task_type: TaskType,
    run_id: str,
    started_at: datetime,
    dataset: DatasetInfo,
    error: str,
    params: dict[str, Any] | None = None,
) -> ModelResult:
    """Arma un resultado fallido para que el tablero muestre el error, no un vacío."""
    finished = ahora_utc()
    return ModelResult(
        model_id=model_id,
        model_name=model_name,
        catalog_ref=catalog_ref,
        use_case=use_case,
        task_type=task_type,
        run_id=run_id,
        status="failed",
        started_at=started_at,
        finished_at=finished,
        duration_seconds=round((finished - started_at).total_seconds(), 3),
        dataset=dataset,
        params=params or {},
        error=error[:2000],
    )
