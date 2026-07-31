"""Logging uniforme para todos los jobs, para que los logs de Airflow se lean igual."""

from __future__ import annotations

import logging
import sys

from .config import get_settings

_FORMATO = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"


def configurar_logging(nivel: str | None = None) -> logging.Logger:
    """Configura el logging del proceso y devuelve el logger raíz de la pipeline."""
    resuelto = (nivel or get_settings().log_level).upper()
    logging.basicConfig(
        level=getattr(logging, resuelto, logging.INFO),
        format=_FORMATO,
        stream=sys.stdout,
        force=True,
    )
    # Estas librerías son ruidosas en DEBUG y no aportan al diagnóstico de los jobs.
    for ruidoso in ("botocore", "boto3", "s3fs", "fsspec", "urllib3", "aiobotocore"):
        logging.getLogger(ruidoso).setLevel(logging.WARNING)
    return logging.getLogger("ach")
