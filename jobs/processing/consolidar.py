"""Consolidación: genera ``results/index.json`` con el resumen de una corrida.

    python -m processing.consolidar --run-id manual__2026-07-30T120000Z

Es la última tarea del DAG. El backend lee este archivo para listar modelos sin tener
que recorrer el bucket, así que ``GET /api/models`` es una lectura y no N.

Un modelo que falló también aparece en el índice, con su estado y su error: el
tablero debe mostrar el fallo, no un hueco.
"""

from __future__ import annotations

import argparse
import logging
import sys
from datetime import UTC, datetime

from common.config import get_settings
from common.logging_config import configurar_logging
from common.registry import registro
from common.results import EntradaIndice, IndiceResultados
from common.storage import get_storage

log = logging.getLogger(__name__)

ARCHIVO_INDICE = "index.json"


def consolidar(run_id: str | None = None) -> IndiceResultados:
    """Recorre los modelos del registro y arma el índice de la corrida."""
    settings = get_settings()
    storage = get_storage(settings)
    run_id = run_id or settings.run_id

    entradas: list[EntradaIndice] = []
    for modelo in registro().modelos:
        carpeta = storage.ruta(settings.bucket_results, modelo.id)
        ruta_corrida = f"{carpeta}/{run_id}.json"
        ruta_ultima = f"{carpeta}/latest.json"

        origen = ruta_corrida if storage.existe(ruta_corrida) else ruta_ultima
        if not storage.existe(origen):
            log.warning("%s no dejó resultado en esta corrida ni tiene uno previo", modelo.id)
            entradas.append(EntradaIndice(
                model_id=modelo.id, model_name=modelo.nombre, catalog_ref=modelo.catalogo,
                use_case=modelo.caso_uso, task_type=modelo.task_type, status="failed",
                run_id=run_id, finished_at=datetime.now(UTC), duration_seconds=0.0,
                error="El modelo no produjo ningún resultado en esta corrida."))
            continue

        datos = storage.leer_json(origen)
        entradas.append(EntradaIndice(
            model_id=datos.get("model_id", modelo.id),
            model_name=datos.get("model_name", modelo.nombre),
            catalog_ref=datos.get("catalog_ref", modelo.catalogo),
            use_case=datos.get("use_case", modelo.caso_uso),
            task_type=datos.get("task_type", modelo.task_type),
            status=datos.get("status", "failed"),
            run_id=datos.get("run_id", run_id),
            finished_at=datos.get("finished_at", datetime.now(UTC)),
            duration_seconds=datos.get("duration_seconds", 0.0),
            metrics=datos.get("metrics", {}),
            latest_uri=ruta_ultima,
            error=datos.get("error"),
        ))

    exitosos = sum(1 for e in entradas if e.status == "success")
    indice = IndiceResultados(
        run_id=run_id, total_models=len(entradas),
        successful=exitosos, failed=len(entradas) - exitosos, models=entradas)

    ruta = storage.ruta(settings.bucket_results, ARCHIVO_INDICE)
    storage.escribir_json(indice.to_json_dict(), ruta)
    log.info("Índice consolidado en %s · %d/%d modelos con resultado",
             ruta, exitosos, len(entradas))
    for entrada in entradas:
        estado = "OK " if entrada.status == "success" else "FALLÓ"
        log.info("  %-6s %-32s %s", estado, entrada.model_id,
                 ", ".join(f"{k}={round(v, 4)}" for k, v in list(entrada.metrics.items())[:3]))
    return indice


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="processing.consolidar", description="Genera results/index.json con el resumen de la corrida.")
    parser.add_argument("--run-id", default=None, help="Identificador de la corrida (dag_run_id).")
    args = parser.parse_args(argv)

    configurar_logging()
    indice = consolidar(args.run_id)
    # La consolidación no falla porque un modelo haya fallado: su trabajo es dejar
    # constancia de lo que pasó, y el estado de cada modelo ya lo reporta su tarea.
    return 0 if indice.total_models else 1


if __name__ == "__main__":
    sys.exit(main())
