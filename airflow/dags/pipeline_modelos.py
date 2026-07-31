"""DAG ``pipeline_modelos`` — procesa los datos y corre todos los modelos.

    procesamiento ──▶ modelo[caso02_ingresos]      ──┐
                 ├──▶ modelo[caso04_rfm]            ──┤
                 ├──▶ modelo[caso04_propension_...] ──┼──▶ consolidacion
                 └──▶ ...                            ──┘

Los modelos se expanden con *dynamic task mapping* desde ``models_config.yml``, así que
**agregar un modelo es agregar una entrada a ese archivo**: el DAG no cambia.

Airflow solo orquesta. Cada tarea lanza un contenedor con la imagen de jobs, que es
donde viven pandas y scikit-learn. Dos detalles que hay que respetar para que eso
funcione:

* El contenedor de Airflow monta ``/var/run/docker.sock`` y corre con un grupo que
  tiene permiso sobre el socket (ver ``docker-compose.yml``).
* Los contenedores lanzados usan la red del compose, no la de por defecto: si no,
  no resuelven ``minio`` por nombre de servicio.
"""

from __future__ import annotations

import logging
import os
import re
from datetime import timedelta
from pathlib import Path

import yaml
from airflow.providers.docker.operators.docker import DockerOperator
from airflow.sdk import DAG

log = logging.getLogger(__name__)

# --------------------------------------------------------------------------- #
# Configuración desde el entorno: nada de rutas ni credenciales en el código    #
# --------------------------------------------------------------------------- #
IMAGEN_JOBS = os.environ.get("ACH_JOBS_IMAGE", "ach-jobs:latest")
RED_COMPOSE = os.environ.get("ACH_DOCKER_NETWORK", "ach_net")
SOCKET_DOCKER = os.environ.get("ACH_DOCKER_URL", "unix://var/run/docker.sock")
POOL_MODELOS = os.environ.get("ACH_POOL_MODELOS", "modelos")
RUTA_CONFIG = Path(os.environ.get("ACH_MODELS_CONFIG", "/opt/airflow/config/models_config.yml"))

# Variables que reciben todos los contenedores de jobs.
ENTORNO_JOBS = {
    "ACH_S3_ENDPOINT": os.environ.get("ACH_S3_ENDPOINT", "http://minio:9000"),
    "ACH_S3_ACCESS_KEY": os.environ.get("ACH_S3_ACCESS_KEY", ""),
    "ACH_S3_SECRET_KEY": os.environ.get("ACH_S3_SECRET_KEY", ""),
    "ACH_BUCKET_RAW": os.environ.get("ACH_BUCKET_RAW", "raw"),
    "ACH_BUCKET_CURATED": os.environ.get("ACH_BUCKET_CURATED", "curated"),
    "ACH_BUCKET_RESULTS": os.environ.get("ACH_BUCKET_RESULTS", "results"),
    "ACH_LINEAGE": os.environ.get("ACH_LINEAGE", "cedula-v1"),
    "ACH_WINDOW_START": os.environ.get("ACH_WINDOW_START", "2025-01"),
    "ACH_WINDOW_END": os.environ.get("ACH_WINDOW_END", "2026-06"),
    "ACH_SEED": os.environ.get("ACH_SEED", "42"),
    "ACH_LOG_LEVEL": os.environ.get("ACH_LOG_LEVEL", "INFO"),
}


def sanear_run_id(run_id: str) -> str:
    """Convierte el run_id de Airflow en una key de S3 limpia.

    ``manual__2026-07-30T12:00:00+00:00`` trae ``:`` y ``+``, que sirven como key pero
    obligan a escapar en URLs y rutas. Se normaliza una sola vez, aquí, para que el
    backend pueda cruzar ``GET /api/runs/<run_id>`` con lo que hay en el bucket.
    """
    return re.sub(r"[^A-Za-z0-9._-]", "", run_id.replace(":", "").replace("+", ""))


def cargar_modelos() -> list[dict]:
    """Lee el registro de modelos. Es lo único que define cuántas tareas se crean."""
    if not RUTA_CONFIG.exists():
        log.error("No se encontró el registro de modelos en %s", RUTA_CONFIG)
        return []
    contenido = yaml.safe_load(RUTA_CONFIG.read_text(encoding="utf-8")) or {}
    return contenido.get("modelos", [])


MODELOS = cargar_modelos()

ARGUMENTOS_BASE = {
    "owner": "ach-data",
    "depends_on_past": False,
    "email_on_failure": False,
}

OPCIONES_DOCKER = {
    "image": IMAGEN_JOBS,
    "docker_url": SOCKET_DOCKER,
    "network_mode": RED_COMPOSE,
    "auto_remove": "success",
    # En hosts Windows el montaje del directorio temporal falla; los jobs no lo usan.
    "mount_tmp_dir": False,
    "tty": False,
    "environment": ENTORNO_JOBS,
}


with DAG(
    dag_id="pipeline_modelos",
    description="Procesa los extractos de ACH y ejecuta todos los modelos del registro",
    schedule=None,          # ejecución manual desde la UI
    catchup=False,
    max_active_runs=1,
    default_args=ARGUMENTOS_BASE,
    tags=["ach", "modelos", "crisp-dm"],
    user_defined_macros={"sanear": sanear_run_id},
    doc_md=__doc__,
) as dag:

    procesamiento = DockerOperator(
        task_id="procesamiento",
        command=["processing.main"],
        retries=1,
        retry_delay=timedelta(minutes=2),
        execution_timeout=timedelta(hours=1),
        doc_md=(
            "Lee los XLSX de `raw/`, valida contra el diccionario de datos y escribe "
            "`curated/dataset.parquet` particionado, con su `_manifest.json`."
        ),
        **OPCIONES_DOCKER,
    )

    # Dynamic task mapping: una tarea por entrada del registro, todas en paralelo
    # (limitadas por el pool a 2 simultáneas para no reventar la memoria del host).
    modelos = DockerOperator.partial(
        task_id="modelo",
        retries=2,
        retry_delay=timedelta(minutes=1),
        execution_timeout=timedelta(hours=2),
        pool=POOL_MODELOS,
        map_index_template="{{ task.container_name }}",
        doc_md=(
            "Ejecuta un modelo del registro y deja su JSON en "
            "`results/<model_id>/<run_id>.json`. Los modelos no se conocen entre sí."
        ),
        **OPCIONES_DOCKER,
    ).expand_kwargs([
        {
            "command": [
                "models.runner",
                "--model-id", modelo["id"],
                "--run-id", "{{ sanear(dag_run.run_id) }}",
            ],
            "container_name": f"ach-modelo-{modelo['id']}",
        }
        for modelo in MODELOS
    ])

    consolidacion = DockerOperator(
        task_id="consolidacion",
        command=["processing.consolidar", "--run-id", "{{ sanear(dag_run.run_id) }}"],
        retries=1,
        retry_delay=timedelta(seconds=30),
        # Corre aunque algún modelo falle: el índice debe reflejar lo que pasó,
        # incluidos los fallos.
        trigger_rule="all_done",
        doc_md="Genera `results/index.json` con el resumen de todos los modelos de la corrida.",
        **OPCIONES_DOCKER,
    )

    procesamiento >> modelos >> consolidacion
