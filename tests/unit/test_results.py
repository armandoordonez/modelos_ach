"""Tests del contrato del JSON de resultados."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta

import pytest
from pydantic import ValidationError

from common.results import (
    Charts,
    CurvaROC,
    DatasetInfo,
    EntradaIndice,
    IndiceResultados,
    ModelResult,
    Segmento,
    construir_fallo,
    construir_resultado,
)


def _dataset() -> DatasetInfo:
    return DatasetInfo(uri="curated/dataset.parquet", manifest_hash="abc123", rows=100,
                       window=("2025-01", "2026-06"), lineage="cedula-v1")


def test_resultado_minimo_calcula_la_duracion():
    inicio = datetime.now(UTC) - timedelta(seconds=5)
    resultado = construir_resultado(
        model_id="caso05_clv", model_name="CLV", catalog_ref="#4", use_case=5,
        task_type="clustering", run_id="manual__2026-07-30T120000Z",
        started_at=inicio, dataset=_dataset(), metrics={"silhouette": 0.231},
    )
    assert resultado.status == "success"
    assert resultado.duration_seconds >= 5
    assert resultado.schema_version == "1.0"


def test_las_metricas_deben_ser_finitas():
    inicio = datetime.now(UTC)
    for valor in (float("nan"), float("inf")):
        with pytest.raises(ValidationError, match="no admite NaN"):
            construir_resultado(
                model_id="m", model_name="M", catalog_ref="#1", use_case=5,
                task_type="clustering", run_id="r", started_at=inicio,
                dataset=_dataset(), metrics={"silhouette": valor},
            )


def test_task_type_invalido_falla():
    with pytest.raises(ValidationError):
        ModelResult(
            model_id="m", model_name="M", use_case=5, task_type="magia", run_id="r",
            started_at=datetime.now(UTC), finished_at=datetime.now(UTC),
            duration_seconds=1.0, dataset=_dataset(),
        )


def test_caso_de_uso_fuera_de_rango_falla():
    with pytest.raises(ValidationError):
        ModelResult(
            model_id="m", model_name="M", use_case=9, task_type="clustering", run_id="r",
            started_at=datetime.now(UTC), finished_at=datetime.now(UTC),
            duration_seconds=1.0, dataset=_dataset(),
        )


def test_los_bloques_de_charts_son_opcionales():
    """El frontend debe poder omitir secciones: un modelo de clustering no reporta ROC."""
    charts = Charts(segment_distribution=[Segmento(id=0, label="Alto valor", n=44, share=0.12)])
    serializado = charts.model_dump(exclude_none=True)
    assert "segment_distribution" in serializado
    assert "roc_curve" not in serializado
    assert "confusion_matrix" not in serializado


def test_un_clasificador_si_reporta_roc():
    charts = Charts(roc_curve=CurvaROC(fpr=[0.0, 0.5, 1.0], tpr=[0.0, 0.8, 1.0], auc=0.741))
    serializado = charts.model_dump(exclude_none=True)
    assert serializado["roc_curve"]["auc"] == 0.741
    assert "segment_distribution" not in serializado


def test_charts_admite_bloques_nuevos_sin_romper_el_contrato():
    charts = Charts(bloque_del_futuro={"cosa": 1})
    assert charts.model_dump(exclude_none=True)["bloque_del_futuro"] == {"cosa": 1}


def test_share_de_segmento_debe_ser_proporcion():
    with pytest.raises(ValidationError):
        Segmento(id=0, label="x", n=10, share=1.5)


def test_resultado_fallido_conserva_el_error_y_no_inventa_metricas():
    resultado = construir_fallo(
        model_id="caso04_propension_salud", model_name="Propensión", catalog_ref="#17",
        use_case=4, task_type="classification", run_id="r",
        started_at=datetime.now(UTC), dataset=_dataset(),
        error="No hay suficientes positivos en la ventana objetivo",
    )
    assert resultado.status == "failed"
    assert resultado.metrics == {}
    assert "positivos" in resultado.error


def test_el_error_se_trunca_para_no_inflar_el_json():
    resultado = construir_fallo(
        model_id="m", model_name="M", catalog_ref="", use_case=5, task_type="clustering",
        run_id="r", started_at=datetime.now(UTC), dataset=_dataset(), error="x" * 5000,
    )
    assert len(resultado.error) == 2000


def test_el_json_es_serializable_y_omite_lo_vacio():
    inicio = datetime.now(UTC)
    resultado = construir_resultado(
        model_id="caso05_pensionados", model_name="Pensionados", catalog_ref="#101", use_case=5,
        task_type="clustering", run_id="r", started_at=inicio, dataset=_dataset(),
        metrics={"silhouette": 0.271, "davies_bouldin": 1.201},
        segments=[Segmento(id=2, label="Alto consumo", n=44, share=0.122)],
    )
    payload = resultado.to_json_dict()
    assert payload["metrics"]["silhouette"] == 0.271
    assert payload["segments"][0]["label"] == "Alto consumo"
    assert "error" not in payload

    import json

    assert json.loads(json.dumps(payload))["model_id"] == "caso05_pensionados"


def test_los_campos_opcionales_de_dashboard_no_rompen_resultados_anteriores():
    inicio = datetime.now(UTC)
    resultado = construir_resultado(
        model_id="caso04_propension_salud", model_name="Propensión", catalog_ref="#17", use_case=4,
        task_type="classification", run_id="r", started_at=inicio, dataset=_dataset(),
        metrics={"roc_auc": 0.71},
    )
    payload = resultado.to_json_dict()
    assert "availability" not in payload
    assert "category" not in payload
    assert "placeholder" not in payload


def test_indice_de_resultados_resume_la_corrida():
    indice = IndiceResultados(
        run_id="manual__2026-07-30T120000Z", total_models=2, successful=1, failed=1,
        models=[
            EntradaIndice(model_id="a", model_name="A", use_case=5, task_type="clustering",
                          status="success", run_id="r", finished_at=datetime.now(UTC),
                          duration_seconds=1.0, metrics={"silhouette": 0.3}),
            EntradaIndice(model_id="b", model_name="B", use_case=4, task_type="classification",
                          status="failed", run_id="r", finished_at=datetime.now(UTC),
                          duration_seconds=0.5, error="falló"),
        ],
    )
    payload = indice.to_json_dict()
    assert payload["total_models"] == 2
    assert len(payload["models"]) == 2
