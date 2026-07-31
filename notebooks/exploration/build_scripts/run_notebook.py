# -*- coding: utf-8 -*-
"""Ejecuta un notebook in-place (mismo patrón que el resto del proyecto)."""
import sys
from pathlib import Path

import nbformat as nbf
from nbclient import NotebookClient

ROOT = Path(__file__).resolve().parents[1]
destino = ROOT / sys.argv[1]
nb = nbf.read(destino, as_version=4)
cliente = NotebookClient(nb, timeout=1800, kernel_name="python3",
                         resources={"metadata": {"path": str(ROOT)}})
cliente.execute()
nbf.write(nb, destino)
print(f"Ejecutado: {destino}")
