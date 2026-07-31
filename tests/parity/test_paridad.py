"""Tests de paridad: comparan las métricas de la pipeline contra las de referencia.

Necesitan el dataset curado, así que se saltan solos si no existe. Se corren con:

    make parity

Los valores de referencia viven en ``referencias.json`` y su interpretación está en
``Docs/PARIDAD.md``. Para los tres modelos del Caso 05 la tolerancia es estrecha: deben
reproducir el notebook. Para los de Caso 02 y 04 el valor esperado es el del
re-baseline documentado, no el del script original.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from common.config import get_settings
from common.registry import cargar_registro
from common.storage import get_storage

pytestmark = pytest.mark.parity

REFERENCIAS = json.loads((Path(__file__).parent / "referencias.json").read_text(encoding="utf-8"))

# Tolerancia relativa por tipo de métrica. Los conteos deben ser exactos; las métricas
# continuas admiten el ruido de una versión distinta de BLAS o de scikit-learn.
TOLERANCIA = {
    "n_entities": 0.0,
    "k": 0.0,
    "pensionados_identificados": 0.0,
    "n_entrenamiento": 0.0,
    "n_positivos": 0.0,
    "silhouette": 0.02,
    "davies_bouldin": 0.02,
    "roc_auc": 0.03,
    "roc_auc_nuevos_adoptantes": 0.05,
    "r2_log": 0.05,
    "mape": 0.05,
    "pct_con_transaccional": 0.01,
}
TOLERANCIA_POR_DEFECTO = 0.05


def _resultado(model_id: str) -> dict | None:
    """Lee el último resultado de un modelo. None si no está o si no hay bucket."""
    ajustes = get_settings()
    storage = get_storage(ajustes)
    ruta = storage.ruta(ajustes.bucket_results, model_id, "latest.json")
    try:
        if not storage.existe(ruta):
            return None
        return storage.leer_json(ruta)
    except Exception:  # noqa: BLE001
        # No hay almacenamiento accesible (MinIO apagado, credenciales ausentes,
        # endpoint inalcanzable): los tests de paridad se saltan, no fallan.
        return None


@pytest.fixture(scope="module")
def resultados() -> dict[str, dict]:
    encontrados = {m.id: _resultado(m.id) for m in cargar_registro().modelos}
    disponibles = {k: v for k, v in encontrados.items() if v}
    if not disponibles:
        pytest.skip(
            "No hay resultados accesibles en el bucket. Levanta el stack y corre la "
            "pipeline, o exporta ACH_S3_ENDPOINT='' y ACH_LOCAL_ROOT para usar el "
            "almacén local, antes de 'make parity'."
        )
    return disponibles


@pytest.mark.parametrize("model_id", sorted(REFERENCIAS["modelos"]))
def test_metricas_dentro_de_tolerancia(resultados, model_id):
    if model_id not in resultados:
        pytest.skip(f"{model_id} no tiene resultado en este bucket")

    obtenido = resultados[model_id]
    assert obtenido["status"] == "success", f"{model_id} terminó en estado {obtenido['status']}"

    esperadas = REFERENCIAS["modelos"][model_id]["metricas_esperadas"]
    metricas = obtenido["metrics"]
    desviaciones = []
    for nombre, referencia in esperadas.items():
        assert nombre in metricas, f"{model_id} dejó de reportar la métrica {nombre!r}"
        tolerancia = TOLERANCIA.get(nombre, TOLERANCIA_POR_DEFECTO)
        limite = abs(referencia) * tolerancia
        diferencia = abs(metricas[nombre] - referencia)
        if diferencia > limite:
            desviaciones.append(
                f"  {nombre}: esperado {referencia}, obtenido {metricas[nombre]} "
                f"(diferencia {diferencia:.4f} > tolerancia {limite:.4f})")
    assert not desviaciones, (
        f"{model_id} se salió de la referencia:\n" + "\n".join(desviaciones)
        + "\nSi el cambio es intencional, actualiza referencias.json y Docs/PARIDAD.md."
    )


@pytest.mark.parametrize("model_id", sorted(REFERENCIAS["modelos"]))
def test_el_contrato_del_json_se_respeta(resultados, model_id):
    """Lo que consume el backend no puede romperse aunque cambie el modelo."""
    if model_id not in resultados:
        pytest.skip(f"{model_id} no tiene resultado en este bucket")

    from common.results import ModelResult

    resultado = ModelResult.model_validate(resultados[model_id])
    assert resultado.model_id == model_id
    assert resultado.metrics, "un modelo exitoso debe reportar métricas"
    assert resultado.dataset.rows > 0
    assert resultado.dataset.lineage in ("cedula-v1", "legacy")


def test_los_modelos_de_clustering_publican_sus_segmentos(resultados):
    for model_id, datos in resultados.items():
        if datos.get("task_type") != "clustering":
            continue
        assert datos["segments"], f"{model_id} es clustering pero no publicó segmentos"
        total = sum(s["n"] for s in datos["segments"])
        assert total == datos["dataset"]["rows"], (
            f"{model_id}: los segmentos suman {total} personas pero el dataset tiene "
            f"{datos['dataset']['rows']}")
        suma_share = sum(s["share"] for s in datos["segments"])
        assert abs(suma_share - 1.0) < 0.01, f"{model_id}: los share suman {suma_share}"


def test_los_clasificadores_publican_roc_y_matriz_de_confusion(resultados):
    """Es lo que hace que el tablero pueda dibujar esas secciones."""
    for model_id, datos in resultados.items():
        if datos.get("task_type") != "classification":
            continue
        charts = datos.get("charts", {})
        assert "roc_curve" in charts, f"{model_id} no publicó la curva ROC"
        assert "confusion_matrix" in charts, f"{model_id} no publicó la matriz de confusión"
        assert len(charts["roc_curve"]["fpr"]) == len(charts["roc_curve"]["tpr"])


def test_la_fuga_de_la_categoria_objetivo_esta_corregida(resultados):
    """Regresión del defecto más importante que traía el script del Caso 04."""
    for model_id in ("caso04_propension_salud", "caso04_propension_turismo"):
        if model_id not in resultados:
            continue
        params = resultados[model_id]["params"]
        excluidas = params.get("variables_excluidas_por_fuga", [])
        assert excluidas, (
            f"{model_id} no excluyó ninguna variable de la categoría objetivo: la fuga volvió")
        categoria = params["categoria_objetivo"]
        assert all(categoria.split()[0] in v for v in excluidas), (
            f"{model_id} excluyó {excluidas}, que no corresponden a {categoria!r}")


def test_los_modelos_del_caso05_mantienen_paridad_exacta(resultados):
    """El Caso 05 ya cruzaba por cédula: la unificación no debe haberlo movido."""
    for model_id, referencia in REFERENCIAS["modelos"].items():
        if referencia["paridad"] != "exacta" or model_id not in resultados:
            continue
        notebook = referencia["metricas_notebook_original"]
        metricas = resultados[model_id]["metrics"]
        for nombre, valor in notebook.items():
            if nombre not in metricas:
                continue
            tolerancia = max(abs(valor) * TOLERANCIA.get(nombre, TOLERANCIA_POR_DEFECTO), 1e-9)
            assert abs(metricas[nombre] - valor) <= tolerancia, (
                f"{model_id}.{nombre}: el notebook daba {valor} y la pipeline da "
                f"{metricas[nombre]}. La paridad exacta se rompió.")
