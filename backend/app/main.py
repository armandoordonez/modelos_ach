"""API de resultados de los modelos de ACH.

Sirve lo que la pipeline deja en el bucket. No calcula nada: si un número no está en
el JSON, el backend no se lo inventa.

    uvicorn app.main:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import logging

from fastapi import FastAPI, HTTPException, Query, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import RedirectResponse

from . import servicio
from .cache import get_cache
from .config import get_settings

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
log = logging.getLogger("ach.api")

ajustes = get_settings()

app = FastAPI(
    title=ajustes.api_titulo,
    version="1.0.0",
    description=(
        "Resultados de los modelos analíticos de ACH Colombia. "
        "El esquema de cada resultado está documentado en Docs/CONTRATO_RESULTADOS.md."
    ),
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=ajustes.origenes_cors,
    allow_credentials=False,
    allow_methods=["GET"],
    allow_headers=["*"],
)


# --------------------------------------------------------------------------- #
# Salud                                                                        #
# --------------------------------------------------------------------------- #
@app.get("/health", tags=["salud"])
def health() -> dict:
    """Liveness y comprobación de que el bucket responde."""
    almacenamiento = servicio.estado_almacenamiento()
    return {
        "status": "ok" if almacenamiento["alcanzable"] else "degradado",
        "storage": almacenamiento,
        "cache": get_cache().estadisticas,
    }


# --------------------------------------------------------------------------- #
# Modelos                                                                      #
# --------------------------------------------------------------------------- #
@app.get("/api/models", tags=["modelos"])
def listar_modelos() -> dict:
    """Modelos registrados con su último resultado.

    Lee `results/index.json` cuando existe; si no, recorre el bucket.
    """
    return servicio.listar_modelos()


@app.get("/api/models/{model_id}/latest", tags=["modelos"])
def ultimo_resultado(model_id: str) -> dict:
    """Último resultado completo de un modelo, con métricas y bloques de gráficos."""
    datos = servicio.resultado_mas_reciente(model_id)
    if datos is None:
        raise HTTPException(
            status_code=404,
            detail=f"El modelo '{model_id}' no tiene resultados en el bucket todavía.")
    return datos


@app.get("/api/models/{model_id}/runs", tags=["modelos"])
def historico_modelo(model_id: str) -> dict:
    """Histórico de corridas de un modelo, de la más reciente a la más antigua."""
    corridas = servicio.historico(model_id)
    return {"model_id": model_id, "total": len(corridas), "runs": corridas}


@app.get("/api/models/{model_id}/runs/{run_id}", tags=["modelos"])
def resultado_de_corrida(model_id: str, run_id: str) -> dict:
    """Resultado de un modelo en una corrida concreta."""
    datos = servicio.resultado_de_corrida(model_id, run_id)
    if datos is None:
        raise HTTPException(
            status_code=404, detail=f"No hay resultado de '{model_id}' en la corrida '{run_id}'.")
    return datos


# --------------------------------------------------------------------------- #
# Corridas                                                                     #
# --------------------------------------------------------------------------- #
@app.get("/api/runs", tags=["corridas"])
def listar_corridas() -> dict:
    """Corridas presentes en el bucket."""
    corridas = servicio.corridas_conocidas()
    return {"total": len(corridas), "runs": corridas}


@app.get("/api/runs/{run_id}", tags=["corridas"])
def corrida(run_id: str) -> dict:
    """Todos los modelos de una misma corrida."""
    datos = servicio.corrida_completa(run_id)
    if not datos["models"]:
        raise HTTPException(status_code=404, detail=f"La corrida '{run_id}' no tiene resultados.")
    return datos


# --------------------------------------------------------------------------- #
# Casos de uso                                                                 #
# --------------------------------------------------------------------------- #
@app.get("/api/use-cases/4/dashboard", tags=["casos-de-uso"])
def dashboard_caso_04(run_id: str | None = Query(default=None)) -> dict:
    """Vista agregada del caso de uso 04 para el dashboard interactivo."""
    return servicio.dashboard_caso_04(run_id=run_id)


# --------------------------------------------------------------------------- #
# Artefactos                                                                   #
# --------------------------------------------------------------------------- #
@app.get("/api/models/{model_id}/runs/{run_id}/artifacts", tags=["artefactos"])
def artefactos(model_id: str, run_id: str) -> dict:
    """Artefactos disponibles de una corrida (modelo entrenado, asignaciones)."""
    nombres = servicio.listar_artefactos(model_id, run_id)
    if not nombres:
        raise HTTPException(status_code=404, detail="No hay artefactos para esa corrida.")
    return {"model_id": model_id, "run_id": run_id, "artifacts": nombres}


@app.get("/api/models/{model_id}/runs/{run_id}/artifacts/{nombre}", tags=["artefactos"])
def descargar_artefacto(model_id: str, run_id: str, nombre: str):
    """Descarga un artefacto.

    Con MinIO o S3 se responde con una redirección a una URL prefirmada de vida corta;
    en desarrollo con disco local se sirve por streaming. En ningún caso salen
    credenciales hacia el navegador.
    """
    url = servicio.url_artefacto(model_id, run_id, nombre)
    if url:
        return RedirectResponse(url, status_code=307)

    contenido = servicio.leer_artefacto(model_id, run_id, nombre)
    if contenido is None:
        raise HTTPException(status_code=404, detail=f"No existe el artefacto '{nombre}'.")
    return Response(
        content=contenido,
        media_type="application/octet-stream",
        headers={"Content-Disposition": f'attachment; filename="{nombre}"'},
    )


# --------------------------------------------------------------------------- #
# Utilidades                                                                   #
# --------------------------------------------------------------------------- #
@app.post("/api/cache/flush", tags=["salud"])
def limpiar_cache() -> dict:
    """Invalida la caché. Útil justo después de una corrida del DAG."""
    get_cache().limpiar()
    return {"status": "ok", "cache": get_cache().estadisticas}


@app.get("/", include_in_schema=False)
def raiz() -> dict:
    return {
        "servicio": ajustes.api_titulo,
        "documentacion": "/docs",
        "salud": "/health",
        "modelos": "/api/models",
    }
