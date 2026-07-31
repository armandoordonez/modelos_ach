"""Sube los XLSX locales al bucket ``raw``. Es lo que corre ``make seed``.

No forma parte de la pipeline: es el puente entre la entrega de ACH (tres archivos
en una carpeta) y el bucket. La ruta de origen llega por argumento o por la variable
``ACH_DATA_DIR``; nunca está escrita en el código.
"""

from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from datetime import date
from pathlib import Path

from common.logging_config import configurar_logging
from common.schema import ARCHIVO_FUENTE, NOMBRE_FUENTE
from common.storage import get_storage

log = logging.getLogger(__name__)


def subir(origen: Path, fecha: str) -> list[str]:
    """Sube los tres XLSX a ``raw/<fecha>/``. Devuelve las rutas remotas."""
    if not origen.is_dir():
        raise FileNotFoundError(f"No existe el directorio de origen {origen}")

    storage = get_storage()
    faltantes = [n for n in ARCHIVO_FUENTE.values() if not (origen / n).exists()]
    if faltantes:
        raise FileNotFoundError(
            f"Faltan archivos en {origen}: {faltantes}. "
            f"Se esperan exactamente estos nombres: {sorted(ARCHIVO_FUENTE.values())}"
        )

    subidos = []
    for fuente, nombre in ARCHIVO_FUENTE.items():
        local = origen / nombre
        remoto = storage.ruta(storage.settings.bucket_raw, fecha, nombre)
        log.info("Subiendo %s (%.1f MB) → %s", NOMBRE_FUENTE[fuente], local.stat().st_size / 1e6, remoto)
        storage.crear_directorio(remoto.rsplit("/", 1)[0])
        with open(local, "rb") as entrada, storage.fs.open(remoto, "wb") as salida:
            shutil.copyfileobj(entrada, salida, length=8 * 1024 * 1024)
        subidos.append(remoto)
    log.info("Carga %s lista: %d archivos", fecha, len(subidos))
    return subidos


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="seed", description="Sube los XLSX de ACH al bucket raw.")
    parser.add_argument("--origen", default=os.environ.get("ACH_DATA_DIR"),
                        help="Directorio con los 3 XLSX. Por defecto, la variable ACH_DATA_DIR.")
    parser.add_argument("--fecha", default=date.today().isoformat(),
                        help="Carpeta destino dentro de raw/ (YYYY-MM-DD). Por defecto, hoy.")
    args = parser.parse_args(argv)

    configurar_logging()
    if not args.origen:
        log.error("Indica el directorio de los XLSX con --origen o la variable ACH_DATA_DIR.")
        return 2
    try:
        subir(Path(args.origen), args.fecha)
    except FileNotFoundError as error:
        log.error("%s", error)
        return 2
    return 0


if __name__ == "__main__":
    sys.exit(main())
