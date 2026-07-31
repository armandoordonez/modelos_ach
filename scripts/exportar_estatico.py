"""Exporta los resultados del bucket como JSON estáticos para publicar el tablero.

    python scripts/exportar_estatico.py

Deja en ``frontend/public/datos/`` un archivo por cada ruta que el tablero consulta y
un ``config.js`` en modo estático. Después basta con compilar el frontend y subir la
carpeta ``dist`` a cualquier hosting de estáticos: no hace falta backend, ni MinIO, ni
Airflow, ni credenciales.

Es la forma rápida de enseñarle el avance a un cliente. Dos límites que conviene tener
presentes:

* Los datos quedan **congelados** en el momento de la exportación. Para actualizarlos
  hay que volver a exportar y volver a publicar.
* Los artefactos binarios (``.joblib``, ``.parquet``) **no se exportan**: en estático no
  hay quién los sirva, y además no deberían salir del entorno controlado.

**Antes de publicar:** lo que se exporta son métricas reales de ACH sobre datos
ofuscados. Publicarlas en una URL abierta es una decisión de gobierno del dato, no
técnica. Si la demo es pública, considera exportar desde una corrida con
``scripts/seed_minio.py`` (datos sintéticos) en vez de la corrida real.
"""

from __future__ import annotations

import argparse
import json
import logging
import shutil
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ / "jobs") not in sys.path:
    sys.path.insert(0, str(RAIZ / "jobs"))

from common.config import get_settings  # noqa: E402
from common.logging_config import configurar_logging  # noqa: E402
from common.registry import cargar_registro  # noqa: E402
from common.storage import get_storage  # noqa: E402

log = logging.getLogger(__name__)

DESTINO_POR_DEFECTO = RAIZ / "frontend" / "public" / "datos"


def _escribir(destino: Path, contenido) -> None:
    destino.parent.mkdir(parents=True, exist_ok=True)
    destino.write_text(json.dumps(contenido, ensure_ascii=False, indent=1), encoding="utf-8")


def exportar(destino: Path = DESTINO_POR_DEFECTO) -> dict:
    ajustes = get_settings()
    storage = get_storage(ajustes)

    def leer(*partes: str):
        ruta = storage.ruta(ajustes.bucket_results, *partes)
        return storage.leer_json(ruta) if storage.existe(ruta) else None

    if destino.exists():
        shutil.rmtree(destino)
    destino.mkdir(parents=True)

    indice = leer("index.json")
    if not indice:
        raise FileNotFoundError(
            "No hay results/index.json en el bucket. Corre la pipeline antes de exportar."
        )
    _escribir(destino / "models.json", {**indice, "origen": "index"})
    _escribir(destino / "health.json", {
        "status": "ok",
        "storage": {"alcanzable": True, "endpoint": "exportación estática"},
        "cache": {"entradas": 0, "ttl_segundos": 0},
    })

    exportados, corridas = 0, set()
    for modelo in cargar_registro().modelos:
        ultimo = leer(modelo.id, "latest.json")
        if not ultimo:
            log.warning("%s no tiene resultado; se omite", modelo.id)
            continue
        # Los artefactos no viajan: no hay backend que los sirva ni deben salir del
        # entorno controlado.
        ultimo = {**ultimo, "artifacts": {}}
        _escribir(destino / "modelos" / f"{modelo.id}.json", ultimo)
        exportados += 1

        historico = []
        carpeta = storage.ruta(ajustes.bucket_results, modelo.id)
        for ruta in storage.listar(carpeta, sufijo=".json"):
            nombre = ruta.replace("\\", "/").rsplit("/", 1)[-1]
            if nombre == "latest.json":
                continue
            datos = storage.leer_json(ruta)
            corridas.add(datos.get("run_id", ""))
            historico.append({
                "run_id": datos.get("run_id"), "status": datos.get("status"),
                "started_at": datos.get("started_at"), "finished_at": datos.get("finished_at"),
                "duration_seconds": datos.get("duration_seconds"),
                "metrics": datos.get("metrics", {}), "dataset": datos.get("dataset", {}),
            })
        historico.sort(key=lambda c: c.get("finished_at") or "", reverse=True)
        _escribir(destino / "modelos" / f"{modelo.id}-runs.json",
                  {"model_id": modelo.id, "total": len(historico), "runs": historico})

    config = RAIZ / "frontend" / "public" / "config.js"
    config.write_text(
        "// Generado por scripts/exportar_estatico.py — el tablero lee JSON exportados,\n"
        "// no llama a ningún backend.\n"
        'window.__ACH_CONFIG__ = { modo: "estatico" };\n',
        encoding="utf-8")

    resumen = {"modelos": exportados, "corridas": len(corridas), "destino": str(destino)}
    log.info("Exportados %d modelos y %d corridas a %s", exportados, len(corridas), destino)
    log.info("Siguiente paso: make demo-estatico (compila) o 'npm run build' en frontend/")
    return resumen


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="exportar_estatico",
        description="Exporta los resultados como JSON estáticos para publicar el tablero.")
    parser.add_argument("--destino", type=Path, default=DESTINO_POR_DEFECTO)
    args = parser.parse_args(argv)

    configurar_logging()
    log.warning("Lo que se exporta son métricas reales sobre datos ofuscados de ACH. "
                "Publicarlas en una URL abierta es una decisión de gobierno del dato.")
    exportar(args.destino)
    return 0


if __name__ == "__main__":
    sys.exit(main())
