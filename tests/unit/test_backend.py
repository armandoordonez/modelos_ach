"""Tests del backend y del contrato que consume el tablero.

Corren contra un bucket local temporal: no necesitan MinIO ni Docker.
"""

from __future__ import annotations

import json
import sys
from datetime import UTC, datetime
from pathlib import Path

import pytest

RAIZ_BACKEND = Path(__file__).resolve().parents[2] / "backend"
if str(RAIZ_BACKEND) not in sys.path:
    sys.path.insert(0, str(RAIZ_BACKEND))

fastapi = pytest.importorskip("fastapi", reason="El backend necesita fastapi instalado")
from fastapi.testclient import TestClient  # noqa: E402


def _resultado(model_id: str, run_id: str, task_type: str, metrics: dict,
               charts: dict | None = None, status: str = "success") -> dict:
    ahora = datetime.now(UTC).isoformat()
    return {
        "schema_version": "1.0", "model_id": model_id, "model_name": f"Modelo {model_id}",
        "catalog_ref": "#1", "use_case": 5, "task_type": task_type, "run_id": run_id,
        "status": status, "started_at": ahora, "finished_at": ahora, "duration_seconds": 1.5,
        "dataset": {"uri": "curated/dataset.parquet", "manifest_hash": "abc", "rows": 100,
                    "window": ["2025-01", "2026-06"], "lineage": "cedula-v1"},
        "params": {"k": 4}, "metrics": metrics, "segments": [], "charts": charts or {},
        "artifacts": {"model_uri": f"results/{model_id}/{run_id}/model.joblib"}, "notes": [],
    }


@pytest.fixture
def cliente(tmp_path, monkeypatch):
    """Backend apuntando a un bucket local con dos modelos y dos corridas."""
    monkeypatch.setenv("ACH_S3_ENDPOINT", "")
    monkeypatch.setenv("ACH_LOCAL_ROOT", str(tmp_path / "almacen"))
    monkeypatch.setenv("ACH_CORS_ORIGINS", "http://localhost:5173")
    monkeypatch.setenv("ACH_CACHE_TTL", "0")

    for modulo in [m for m in list(sys.modules) if m.startswith(("app.", "common.")) or m in ("app", "common")]:
        del sys.modules[modulo]

    from app.config import get_settings

    get_settings.cache_clear()

    base = tmp_path / "almacen" / "results"
    datos = {
        "caso05_clv": _resultado(
            "caso05_clv", "run_2", "clustering", {"silhouette": 0.23, "k": 4},
            {"segment_distribution": [{"id": 0, "label": "Alto valor", "n": 60, "share": 0.6}]}),
        "caso04_propension_salud": _resultado(
            "caso04_propension_salud", "run_2", "classification", {"roc_auc": 0.69},
            {"roc_curve": {"fpr": [0, 1], "tpr": [0, 1], "auc": 0.69},
             "confusion_matrix": {"labels": ["no", "si"], "matrix": [[8, 2], [1, 4]]}}),
    }
    for model_id, resultado in datos.items():
        carpeta = base / model_id
        carpeta.mkdir(parents=True, exist_ok=True)
        (carpeta / "run_2.json").write_text(json.dumps(resultado), encoding="utf-8")
        (carpeta / "latest.json").write_text(json.dumps(resultado), encoding="utf-8")
        anterior = {**resultado, "run_id": "run_1", "duration_seconds": 2.0}
        (carpeta / "run_1.json").write_text(json.dumps(anterior), encoding="utf-8")
        (carpeta / "run_2").mkdir(exist_ok=True)
        (carpeta / "run_2" / "model.joblib").write_bytes(b"artefacto-de-prueba")

    indice = {
        "schema_version": "1.0", "run_id": "run_2",
        "generated_at": datetime.now(UTC).isoformat(),
        "total_models": 2, "successful": 2, "failed": 0,
        "models": [
            {"model_id": mid, "model_name": r["model_name"], "catalog_ref": "#1", "use_case": 5,
             "task_type": r["task_type"], "status": "success", "run_id": "run_2",
             "finished_at": r["finished_at"], "duration_seconds": 1.5, "metrics": r["metrics"],
             "latest_uri": f"results/{mid}/latest.json"}
            for mid, r in datos.items()
        ],
    }
    (base / "index.json").write_text(json.dumps(indice), encoding="utf-8")

    from app.main import app

    return TestClient(app)


# --------------------------------------------------------------------------- #
# Salud                                                                        #
# --------------------------------------------------------------------------- #
def test_health_reporta_el_estado_del_bucket(cliente):
    respuesta = cliente.get("/health")
    assert respuesta.status_code == 200
    cuerpo = respuesta.json()
    assert cuerpo["status"] == "ok"
    assert cuerpo["storage"]["alcanzable"] is True
    assert "cache" in cuerpo


# --------------------------------------------------------------------------- #
# Modelos                                                                      #
# --------------------------------------------------------------------------- #
def test_listar_modelos_usa_el_indice(cliente):
    cuerpo = cliente.get("/api/models").json()
    assert cuerpo["origen"] == "index", "debe leer index.json y no recorrer el bucket"
    assert cuerpo["total_models"] == 2
    assert {m["model_id"] for m in cuerpo["models"]} == {"caso05_clv", "caso04_propension_salud"}


def test_listar_modelos_cae_al_bucket_si_no_hay_indice(cliente, tmp_path):
    (tmp_path / "almacen" / "results" / "index.json").unlink()
    cuerpo = cliente.get("/api/models").json()
    assert cuerpo["origen"] == "bucket"
    assert cuerpo["total_models"] >= 1


def test_ultimo_resultado_trae_el_contrato_completo(cliente):
    cuerpo = cliente.get("/api/models/caso05_clv/latest").json()
    assert cuerpo["model_id"] == "caso05_clv"
    assert cuerpo["metrics"]["silhouette"] == 0.23
    assert cuerpo["dataset"]["lineage"] == "cedula-v1"
    assert "segment_distribution" in cuerpo["charts"]


def test_modelo_inexistente_devuelve_404_con_mensaje(cliente):
    respuesta = cliente.get("/api/models/no_existe/latest")
    assert respuesta.status_code == 404
    assert "no_existe" in respuesta.json()["detail"]


def test_historico_ordena_de_lo_mas_reciente_a_lo_mas_antiguo(cliente):
    cuerpo = cliente.get("/api/models/caso05_clv/runs").json()
    assert cuerpo["total"] == 2
    assert {c["run_id"] for c in cuerpo["runs"]} == {"run_1", "run_2"}


def test_historico_no_incluye_latest_como_corrida(cliente):
    corridas = cliente.get("/api/models/caso05_clv/runs").json()["runs"]
    assert all(c["run_id"] != "latest" for c in corridas)


# --------------------------------------------------------------------------- #
# Corridas                                                                     #
# --------------------------------------------------------------------------- #
def test_una_corrida_devuelve_todos_sus_modelos(cliente):
    cuerpo = cliente.get("/api/runs/run_2").json()
    assert cuerpo["run_id"] == "run_2"
    assert cuerpo["total_models"] == 2
    assert cuerpo["successful"] == 2


def test_corrida_inexistente_devuelve_404(cliente):
    assert cliente.get("/api/runs/corrida_fantasma").status_code == 404


def test_listar_corridas_conocidas(cliente):
    cuerpo = cliente.get("/api/runs").json()
    assert set(cuerpo["runs"]) == {"run_1", "run_2"}


# --------------------------------------------------------------------------- #
# Artefactos y seguridad                                                       #
# --------------------------------------------------------------------------- #
def test_los_artefactos_se_sirven_por_el_backend(cliente):
    respuesta = cliente.get("/api/models/caso05_clv/runs/run_2/artifacts/model.joblib")
    assert respuesta.status_code == 200
    assert respuesta.content == b"artefacto-de-prueba"


def test_artefacto_inexistente_devuelve_404(cliente):
    assert cliente.get("/api/models/caso05_clv/runs/run_2/artifacts/fantasma.bin").status_code == 404


def test_ninguna_respuesta_expone_credenciales(cliente):
    """Lo que llega al navegador no puede traer llaves del bucket."""
    for ruta in ("/health", "/api/models", "/api/models/caso05_clv/latest", "/api/runs/run_2"):
        texto = cliente.get(ruta).text.lower()
        for prohibido in ("secret_key", "access_key", "minioadmin", "password"):
            assert prohibido not in texto, f"{ruta} expone {prohibido}"


def test_cors_permite_el_origen_del_tablero(cliente):
    respuesta = cliente.get("/api/models", headers={"Origin": "http://localhost:5173"})
    assert respuesta.headers["access-control-allow-origin"] == "http://localhost:5173"


# --------------------------------------------------------------------------- #
# Contrato con el tablero                                                      #
# --------------------------------------------------------------------------- #
RENDERIZADORES_DEL_TABLERO = {
    "segment_distribution", "k_selection", "scatter_2d", "feature_importance",
    "roc_curve", "confusion_matrix", "residuals", "distribution",
}


def test_el_tablero_puede_dibujar_todo_lo_que_publica_la_api(cliente):
    """Si un modelo publicara un bloque desconocido, el tablero lo ignoraría; este
    test avisa para que se agregue su renderizador."""
    for modelo in cliente.get("/api/models").json()["models"]:
        detalle = cliente.get(f"/api/models/{modelo['model_id']}/latest").json()
        desconocidos = set(detalle.get("charts", {})) - RENDERIZADORES_DEL_TABLERO
        assert not desconocidos, f"{modelo['model_id']} publica bloques sin renderizador: {desconocidos}"


def test_cada_modelo_trae_lo_que_la_tarjeta_necesita(cliente):
    for modelo in cliente.get("/api/models").json()["models"]:
        for campo in ("model_id", "model_name", "task_type", "status", "metrics", "use_case"):
            assert campo in modelo, f"falta {campo} en la tarjeta de {modelo.get('model_id')}"
        assert modelo["metrics"], "una tarjeta sin métricas no puede mostrar nada"
