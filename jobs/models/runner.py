"""Ejecutor común de los jobs de modelo.

    python -m models.runner --model-id caso05_pensionados --run-id manual__2026-07-30T120000Z

Es la interfaz que usa el DAG para todos los modelos por igual. Resuelve el módulo
desde ``models_config.yml``, lo ejecuta y escribe el JSON en
``results/<model_id>/<run_id>.json`` más una copia en ``latest.json``.

Si el modelo falla, el JSON se escribe igual con ``status: failed`` y el error —para
que el tablero muestre el fallo en vez de un vacío— y el proceso termina con código
distinto de cero para que Airflow reintente.
"""

from __future__ import annotations

import argparse
import importlib
import logging
import sys
import traceback
from datetime import UTC, datetime

from common.config import get_settings
from common.logging_config import configurar_logging
from common.registry import obtener_modelo
from common.results import ModelResult, construir_fallo
from models.base import ContextoModelo

log = logging.getLogger(__name__)

ARCHIVO_ULTIMO = "latest.json"


def guardar_resultado(ctx: ContextoModelo, resultado: ModelResult) -> tuple[str, str]:
    """Escribe el JSON de la corrida y actualiza ``latest.json``."""
    carpeta = ctx.storage.ruta(ctx.settings.bucket_results, ctx.config.id)
    ruta_corrida = f"{carpeta}/{ctx.run_id}.json"
    ruta_ultimo = f"{carpeta}/{ARCHIVO_ULTIMO}"
    payload = resultado.to_json_dict()
    ctx.storage.escribir_json(payload, ruta_corrida)
    ctx.storage.escribir_json(payload, ruta_ultimo)
    return ruta_corrida, ruta_ultimo


def ejecutar_modelo(model_id: str, run_id: str | None = None) -> ModelResult:
    """Corre un modelo del registro y persiste su resultado."""
    configuracion = obtener_modelo(model_id)
    settings = get_settings()
    ctx = ContextoModelo.crear(configuracion, settings=settings, run_id=run_id)

    log.info("Modelo %s (%s) · run %s · linaje %s",
             configuracion.id, configuracion.catalogo, ctx.run_id, ctx.estrategia)

    modulo = importlib.import_module(configuracion.modulo)
    if not hasattr(modulo, "ejecutar"):
        raise AttributeError(
            f"El módulo {configuracion.modulo} debe exponer una función ejecutar(ctx) -> ModelResult"
        )

    try:
        resultado = modulo.ejecutar(ctx)
    except Exception as error:  # noqa: BLE001 - se reporta y se relanza
        log.exception("El modelo %s falló", configuracion.id)
        fallo = construir_fallo(
            model_id=configuracion.id, model_name=configuracion.nombre,
            catalog_ref=configuracion.catalogo, use_case=configuracion.caso_uso,
            task_type=configuracion.task_type, run_id=ctx.run_id, started_at=ctx.started_at,
            dataset=ctx.dataset_info(0),
            error=f"{type(error).__name__}: {error}\n{traceback.format_exc(limit=5)}",
            params=ctx.params_reportados(),
        )
        guardar_resultado(ctx, fallo)
        raise

    ruta, _ = guardar_resultado(ctx, resultado)
    log.info("Modelo %s listo en %.1f s · %s", configuracion.id, resultado.duration_seconds, ruta)
    for nombre, valor in resultado.metrics.items():
        log.info("  %-24s %s", nombre, round(valor, 4))
    return resultado


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="models.runner", description="Ejecuta un modelo de la pipeline.")
    parser.add_argument("--model-id", required=True, help="Identificador del modelo en models_config.yml")
    parser.add_argument("--run-id", default=None, help="Identificador de la corrida (dag_run_id de Airflow)")
    args = parser.parse_args(argv)

    configurar_logging()
    inicio = datetime.now(UTC)
    try:
        ejecutar_modelo(args.model_id, args.run_id)
    except KeyError as error:
        log.error("%s", error)
        return 2
    except Exception:  # noqa: BLE001 - el detalle ya quedó en el JSON y en el log
        log.error("El modelo %s terminó con error tras %.1f s",
                  args.model_id, (datetime.now(UTC) - inicio).total_seconds())
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
