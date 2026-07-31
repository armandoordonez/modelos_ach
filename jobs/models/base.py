"""Andamiaje común de los jobs de modelo.

Todos los modelos reciben el mismo contexto y devuelven el mismo tipo de resultado.
Lo que cambia de un modelo a otro es únicamente su función ``ejecutar``.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from common.config import Settings, get_settings
from common.features import EstrategiaLlave, estrategia_desde_config, leer_manifiesto, uri_curado
from common.registry import ModeloConfig
from common.results import DatasetInfo, Segmento
from common.storage import Storage, get_storage

log = logging.getLogger(__name__)


@dataclass
class ContextoModelo:
    """Todo lo que un modelo necesita para correr, ya resuelto."""

    config: ModeloConfig
    settings: Settings
    storage: Storage
    run_id: str
    started_at: datetime
    estrategia: EstrategiaLlave
    manifiesto: dict = field(default_factory=dict)

    @classmethod
    def crear(cls, config: ModeloConfig, settings: Settings | None = None,
              run_id: str | None = None) -> ContextoModelo:
        settings = settings or get_settings()
        storage = get_storage(settings)
        return cls(
            config=config,
            settings=settings,
            storage=storage,
            run_id=run_id or settings.run_id,
            started_at=datetime.now(UTC),
            estrategia=estrategia_desde_config(config.legacy_key, settings),
            manifiesto=leer_manifiesto(storage),
        )

    def param(self, nombre: str, defecto: Any = None) -> Any:
        return self.config.params.get(nombre, defecto)

    @property
    def semilla(self) -> int:
        """Semilla del modelo. Cada uno declara la suya para reproducir su notebook."""
        return int(self.param("seed", self.settings.seed))

    @property
    def ventana(self) -> tuple[str, str]:
        """Ventana de análisis del modelo.

        Por defecto la común del proyecto, pero un modelo puede acotarla desde el
        registro si tiene una razón metodológica (el Caso 02, por ejemplo, se detiene
        en 2025-09 porque Transfiya se degrada después). Queda declarada en el YAML,
        no escondida en el código ni recalculada según los datos que lleguen.
        """
        return (self.param("ventana_inicio") or self.settings.window_start,
                self.param("ventana_fin") or self.settings.window_end)

    @property
    def settings_ventana(self) -> Settings:
        """Configuración con la ventana del modelo ya aplicada."""
        inicio, fin = self.ventana
        if (inicio, fin) == self.settings.ventana:
            return self.settings
        return self.settings.model_copy(update={"window_start": inicio, "window_end": fin})

    @property
    def meses_ventana(self) -> int:
        return self.settings_ventana.meses_ventana()

    def dataset_info(self, filas: int) -> DatasetInfo:
        return DatasetInfo(
            uri=uri_curado(self.storage),
            manifest_hash=str(self.manifiesto.get("hash", "")),
            rows=int(filas),
            window=self.ventana,
            lineage=self.settings.lineage,
        )

    def params_reportados(self, extra: dict[str, Any] | None = None) -> dict[str, Any]:
        """Parámetros que van al JSON: los del registro más los que calcule el modelo."""
        base = {
            **self.config.params,
            "lineage": self.settings.lineage,
            "estrategia_llave": self.estrategia,
            "ventana": list(self.ventana),
        }
        base.update(extra or {})
        return base


def guardar_artefacto(ctx: ContextoModelo, objeto: Any, nombre: str) -> str:
    """Serializa un artefacto (modelo entrenado) al bucket de resultados."""
    import io

    import joblib

    ruta = ctx.storage.ruta(ctx.settings.bucket_results, ctx.config.id, ctx.run_id, nombre)
    ctx.storage.crear_directorio(ruta.rsplit("/", 1)[0])
    memoria = io.BytesIO()
    joblib.dump(objeto, memoria)
    with ctx.storage.fs.open(ruta, "wb") as destino:
        destino.write(memoria.getvalue())
    log.info("Artefacto guardado: %s", ruta)
    return ruta


def guardar_asignaciones(ctx: ContextoModelo, df: pd.DataFrame, nombre: str = "assignments.parquet") -> str:
    """Guarda la tabla persona → segmento/score del modelo."""
    ruta = ctx.storage.ruta(ctx.settings.bucket_results, ctx.config.id, ctx.run_id, nombre)
    ctx.storage.escribir_parquet(df, ruta)
    return ruta


# --------------------------------------------------------------------------- #
# Utilidades numéricas compartidas                                             #
# --------------------------------------------------------------------------- #
def limpiar_metricas(metricas: dict[str, Any]) -> dict[str, float]:
    """Descarta métricas no finitas en vez de escribir NaN en el JSON."""
    salida: dict[str, float] = {}
    for nombre, valor in metricas.items():
        if valor is None:
            continue
        numero = float(valor)
        if np.isnan(numero) or np.isinf(numero):
            log.warning("Métrica %s descartada por no ser finita (%s)", nombre, valor)
            continue
        salida[nombre] = round(numero, 6)
    return salida


def construir_segmentos(
    etiquetas: pd.Series,
    nombres: dict[int, str],
    perfil: pd.DataFrame,
    columnas_perfil: list[str] | None = None,
) -> list[Segmento]:
    """Arma el bloque ``segments`` a partir del perfil por cluster."""
    total = len(etiquetas)
    columnas_perfil = columnas_perfil or [c for c in perfil.columns if perfil[c].dtype.kind in "if"]
    segmentos = []
    for cluster in sorted(perfil.index):
        n = int((etiquetas == cluster).sum())
        fila = perfil.loc[cluster]
        segmentos.append(Segmento(
            id=int(cluster),
            label=nombres.get(int(cluster), f"Segmento {cluster}"),
            n=n,
            share=round(n / total, 6) if total else 0.0,
            profile={c: round(float(fila[c]), 4) for c in columnas_perfil
                     if pd.notna(fila.get(c)) and np.isfinite(float(fila[c]))},
        ))
    return segmentos


def muestra_scatter(proyeccion: np.ndarray, etiquetas: pd.Series, nombres: dict[int, str],
                    maximo: int = 2000, semilla: int = 42) -> list[dict]:
    """Submuestrea el plano PCA: el JSON no debe cargar 30.000 puntos al navegador."""
    n = len(proyeccion)
    indices = (np.random.RandomState(semilla).choice(n, maximo, replace=False)
               if n > maximo else np.arange(n))
    valores = etiquetas.to_numpy()
    return [
        {"x": round(float(proyeccion[i, 0]), 4),
         "y": round(float(proyeccion[i, 1]), 4),
         "label": nombres.get(int(valores[i]), str(valores[i]))}
        for i in indices
    ]


def cli_para(model_id_defecto: str | None = None):
    """Construye el ``main()`` de un módulo de modelo.

    Todos comparten la misma interfaz de línea de comandos; el módulo solo aporta
    cuál es su modelo por defecto. Los que sirven a más de una entrada del registro
    (como el de propensión) no declaran defecto y exigen ``--model-id``.
    """

    def main(argv: list[str] | None = None) -> int:
        import sys

        from models.runner import main as runner

        argumentos = list(argv if argv is not None else sys.argv[1:])
        if model_id_defecto and "--model-id" not in argumentos:
            argumentos += ["--model-id", model_id_defecto]
        return runner(argumentos)

    return main


def nombres_unicos(perfil: pd.DataFrame, nombrar, orden_por: str) -> dict[int, str]:
    """Aplica la regla de nombres y desambigua repetidos con el tamaño del cluster.

    Los nombres salen del perfil, no del número de cluster: así son estables entre
    corridas aunque K-Means numere distinto.
    """
    propuestos = {int(cl): nombrar(perfil.loc[cl]) for cl in perfil.index}
    asignados: dict[int, str] = {}
    usados: set[str] = set()
    for cluster in perfil.sort_values(orden_por, ascending=False).index:
        etiqueta = propuestos[int(cluster)]
        if etiqueta in usados:
            etiqueta = f"{etiqueta} ({int(perfil.loc[cluster, 'personas']):,})"
        usados.add(etiqueta)
        asignados[int(cluster)] = etiqueta
    return asignados
