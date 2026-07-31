"""Acceso al bucket de resultados.

Usa el mismo cliente que la pipeline (``common/storage.py``), así que funciona igual
contra MinIO y contra AWS cambiando solo el endpoint. El backend nunca reimplementa
rutas ni formatos: los toma del contrato de ``common/results.py``.
"""

from __future__ import annotations

import logging
from typing import Any

from common.registry import cargar_registro
from common.storage import Storage

from .cache import get_cache
from .config import get_settings

log = logging.getLogger(__name__)

ARCHIVO_INDICE = "index.json"
ARCHIVO_ULTIMO = "latest.json"


class ErrorDeAlmacenamiento(RuntimeError):
    """El bucket no está accesible."""


def _storage() -> Storage:
    return Storage(get_settings())


def _ruta(*partes: str) -> str:
    ajustes = get_settings()
    return _storage().ruta(ajustes.bucket_results, *partes)


def _leer_json(ruta: str) -> Any | None:
    almacen = _storage()
    try:
        if not almacen.existe(ruta):
            return None
        return almacen.leer_json(ruta)
    except Exception as error:  # noqa: BLE001
        log.warning("No se pudo leer %s: %s", ruta, error)
        return None


# --------------------------------------------------------------------------- #
# Salud                                                                        #
# --------------------------------------------------------------------------- #
def estado_almacenamiento() -> dict[str, Any]:
    """Comprueba que el bucket de resultados responde."""
    ajustes = get_settings()
    try:
        alcanzable = _storage().existe(_storage().ruta(ajustes.bucket_results))
        return {
            "alcanzable": bool(alcanzable),
            "endpoint": ajustes.s3_endpoint or "sistema de archivos local",
            "bucket": ajustes.bucket_results,
        }
    except Exception as error:  # noqa: BLE001
        return {"alcanzable": False, "endpoint": ajustes.s3_endpoint, "error": str(error)[:200]}


# --------------------------------------------------------------------------- #
# Modelos                                                                      #
# --------------------------------------------------------------------------- #
def _resumen_desde_resultado(datos: dict, model_id: str, latest_uri: str) -> dict:
    return {
        "model_id": datos.get("model_id", model_id),
        "model_name": datos.get("model_name", model_id),
        "catalog_ref": datos.get("catalog_ref", ""),
        "use_case": datos.get("use_case"),
        "task_type": datos.get("task_type"),
        "status": datos.get("status", "failed"),
        "run_id": datos.get("run_id", ""),
        "finished_at": datos.get("finished_at"),
        "duration_seconds": datos.get("duration_seconds", 0.0),
        "metrics": datos.get("metrics", {}),
        "latest_uri": latest_uri,
        "error": datos.get("error"),
    }


def _listar_desde_bucket() -> list[dict]:
    """Reconstruye la lista recorriendo el bucket. Es el plan B cuando todavía no
    existe el índice (por ejemplo, si la consolidación no ha corrido)."""
    resumenes = []
    for modelo in cargar_registro().modelos:
        ruta = _ruta(modelo.id, ARCHIVO_ULTIMO)
        datos = _leer_json(ruta)
        if datos:
            resumenes.append(_resumen_desde_resultado(datos, modelo.id, ruta))
    return resumenes


def listar_modelos() -> dict[str, Any]:
    """Modelos con su último resultado.

    Lee ``results/index.json``, que genera la tarea de consolidación: así el endpoint
    es una lectura y no una por modelo.
    """
    def calcular() -> dict[str, Any]:
        indice = _leer_json(_ruta(ARCHIVO_INDICE))
        if indice and indice.get("models"):
            return {
                "run_id": indice.get("run_id", ""),
                "generated_at": indice.get("generated_at"),
                "total_models": indice.get("total_models", len(indice["models"])),
                "successful": indice.get("successful", 0),
                "failed": indice.get("failed", 0),
                "origen": "index",
                "models": indice["models"],
            }

        modelos = _listar_desde_bucket()
        exitosos = sum(1 for m in modelos if m["status"] == "success")
        return {
            "run_id": modelos[0]["run_id"] if modelos else "",
            "generated_at": None,
            "total_models": len(modelos),
            "successful": exitosos,
            "failed": len(modelos) - exitosos,
            "origen": "bucket",
            "models": modelos,
        }

    return get_cache().resolver("modelos", calcular)


def resultado_mas_reciente(model_id: str) -> dict | None:
    """Último resultado completo de un modelo."""
    return get_cache().resolver(
        f"latest:{model_id}", lambda: _leer_json(_ruta(model_id, ARCHIVO_ULTIMO)))


def _run_id_desde_ruta(ruta: str) -> str:
    return ruta.replace("\\", "/").rsplit("/", 1)[-1].removesuffix(".json")


def historico(model_id: str) -> list[dict]:
    """Corridas de un modelo, de la más reciente a la más antigua."""
    def calcular() -> list[dict]:
        almacen = _storage()
        carpeta = _ruta(model_id)
        archivos = [
            r for r in almacen.listar(carpeta, sufijo=".json")
            if _run_id_desde_ruta(r) != ARCHIVO_ULTIMO.removesuffix(".json")
        ]
        corridas = []
        for ruta in archivos:
            datos = _leer_json(ruta)
            if not datos:
                continue
            corridas.append({
                "run_id": datos.get("run_id", _run_id_desde_ruta(ruta)),
                "status": datos.get("status", "failed"),
                "started_at": datos.get("started_at"),
                "finished_at": datos.get("finished_at"),
                "duration_seconds": datos.get("duration_seconds", 0.0),
                "metrics": datos.get("metrics", {}),
                "dataset": datos.get("dataset", {}),
                "error": datos.get("error"),
            })
        corridas.sort(key=lambda c: c.get("finished_at") or "", reverse=True)
        return corridas

    return get_cache().resolver(f"runs:{model_id}", calcular)


def resultado_de_corrida(model_id: str, run_id: str) -> dict | None:
    """Resultado de un modelo en una corrida concreta."""
    return get_cache().resolver(
        f"run:{model_id}:{run_id}", lambda: _leer_json(_ruta(model_id, f"{run_id}.json")))


def corrida_completa(run_id: str) -> dict[str, Any]:
    """Todos los modelos de una misma corrida."""
    def calcular() -> dict[str, Any]:
        modelos = []
        for modelo in cargar_registro().modelos:
            datos = _leer_json(_ruta(modelo.id, f"{run_id}.json"))
            if datos:
                modelos.append(datos)
        exitosos = sum(1 for m in modelos if m.get("status") == "success")
        return {
            "run_id": run_id,
            "total_models": len(modelos),
            "successful": exitosos,
            "failed": len(modelos) - exitosos,
            "models": modelos,
        }

    return get_cache().resolver(f"corrida:{run_id}", calcular)


def corridas_conocidas() -> list[str]:
    """Identificadores de corrida presentes en el bucket, de la más reciente atrás."""
    def calcular() -> list[str]:
        vistos: set[str] = set()
        for modelo in cargar_registro().modelos:
            for ruta in _storage().listar(_ruta(modelo.id), sufijo=".json"):
                run_id = _run_id_desde_ruta(ruta)
                if run_id != "latest":
                    vistos.add(run_id)
        return sorted(vistos, reverse=True)

    return get_cache().resolver("corridas", calcular)


# --------------------------------------------------------------------------- #
# Artefactos                                                                   #
# --------------------------------------------------------------------------- #
def url_artefacto(model_id: str, run_id: str, nombre: str) -> str | None:
    """URL prefirmada de un artefacto, de vida corta.

    Nunca se le entregan al navegador las credenciales de MinIO: se firma la URL en el
    backend. Si el almacenamiento es local (desarrollo), no hay firma posible y se
    devuelve None para que el endpoint sirva el archivo por streaming.
    """
    ajustes = get_settings()
    ruta = _ruta(model_id, run_id, nombre)
    almacen = _storage()
    if not almacen.existe(ruta):
        return None
    if not ajustes.usa_s3:
        return None
    try:
        return almacen.fs.sign(ruta, expiration=ajustes.url_prefirmada_ttl)
    except (NotImplementedError, AttributeError):
        return None


def leer_artefacto(model_id: str, run_id: str, nombre: str) -> bytes | None:
    """Contenido de un artefacto, para servirlo por el backend."""
    ruta = _ruta(model_id, run_id, nombre)
    almacen = _storage()
    if not almacen.existe(ruta):
        return None
    with almacen.fs.open(ruta, "rb") as fh:
        return fh.read()


def listar_artefactos(model_id: str, run_id: str) -> list[str]:
    carpeta = _ruta(model_id, run_id)
    return [r.replace("\\", "/").rsplit("/", 1)[-1] for r in _storage().listar(carpeta)]
