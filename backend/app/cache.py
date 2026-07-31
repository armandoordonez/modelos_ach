"""Caché en memoria con TTL corto.

Los resultados solo cambian cuando corre el DAG, así que no tiene sentido ir al bucket
en cada petición del tablero. El TTL es corto a propósito: si alguien dispara la
pipeline, el tablero debe reflejarlo en segundos, no en minutos.
"""

from __future__ import annotations

import threading
import time
from collections.abc import Callable
from typing import Any

from .config import get_settings


class CacheTTL:
    """Caché sencilla con expiración por tiempo, segura entre hilos."""

    def __init__(self, ttl_segundos: float) -> None:
        self.ttl = ttl_segundos
        self._datos: dict[str, tuple[float, Any]] = {}
        self._lock = threading.Lock()
        self.aciertos = 0
        self.fallos = 0

    def obtener(self, clave: str) -> Any | None:
        with self._lock:
            entrada = self._datos.get(clave)
            if entrada is None:
                self.fallos += 1
                return None
            expira_en, valor = entrada
            if time.monotonic() > expira_en:
                del self._datos[clave]
                self.fallos += 1
                return None
            self.aciertos += 1
            return valor

    def guardar(self, clave: str, valor: Any) -> Any:
        with self._lock:
            self._datos[clave] = (time.monotonic() + self.ttl, valor)
        return valor

    def resolver(self, clave: str, calcular: Callable[[], Any]) -> Any:
        """Devuelve el valor cacheado o lo calcula y lo guarda."""
        valor = self.obtener(clave)
        if valor is not None:
            return valor
        return self.guardar(clave, calcular())

    def limpiar(self) -> None:
        with self._lock:
            self._datos.clear()

    @property
    def estadisticas(self) -> dict[str, Any]:
        total = self.aciertos + self.fallos
        return {
            "entradas": len(self._datos),
            "ttl_segundos": self.ttl,
            "aciertos": self.aciertos,
            "fallos": self.fallos,
            "tasa_acierto": round(self.aciertos / total, 3) if total else 0.0,
        }


_cache: CacheTTL | None = None


def get_cache() -> CacheTTL:
    global _cache
    if _cache is None:
        _cache = CacheTTL(get_settings().cache_ttl)
    return _cache
