"""Modelo #15 — Segmentación RFM de consumidores. Caso de uso 4.

Agrupa a las personas por su comportamiento de compra en el canal PSE sobre tres
ejes: **recencia** (hace cuánto fue el último pago), **frecuencia** (cuántos hizo) y
**monto** (cuánto gastó). Es la técnica estándar de retail y banca para priorizar
campañas de retención y de cross-sell.

Migrado de ``notebooks/exploration/scripts_originales/caso04_comportamientos_consumo.py``
con tres correcciones:

1. **Llave unificada a la cédula ofuscada.** El script cruzaba por nombre normalizado
   más documento sin asteriscos, una tercera variante distinta de las otras dos.
2. **Lee del dataset curado**, no de los XLSX. El original hacía ``pd.read_excel`` del
   archivo completo, tres veces: con la memoria del contenedor eso es un OOM seguro.
3. **Recencia en meses**, no en días. Los extractos son mensuales; medir en días daba
   siempre múltiplos del mes y sugería una precisión que el dato no tiene.
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import davies_bouldin_score, silhouette_score
from sklearn.preprocessing import StandardScaler

from common.features import meses_hasta, tabla_persona_mes
from common.results import Charts, ModelResult, PuntoSeleccionK, Scatter2D, construir_resultado
from models.base import (
    ContextoModelo,
    cli_para,
    construir_segmentos,
    guardar_artefacto,
    guardar_asignaciones,
    limpiar_metricas,
    muestra_scatter,
    nombres_unicos,
)

log = logging.getLogger(__name__)

FEATURES = ["recencia_meses", "log_frecuencia", "log_monetario"]


def construir_rfm(ctx: ContextoModelo) -> pd.DataFrame:
    """Tabla RFM por persona sobre la ventana configurada."""
    pse = tabla_persona_mes("pse", storage=ctx.storage, estrategia=ctx.estrategia,
                            solo_ventana=True, settings=ctx.settings_ventana)
    grupo = pse.groupby("person_id")
    rfm = pd.DataFrame({
        "frecuencia": grupo["n_pagos"].sum(),
        "monetario": grupo["gasto_pse"].sum(),
        "meses_activos": grupo["periodo"].nunique(),
        "n_comercios": grupo["n_comercios"].sum(),
        "ultimo_periodo": grupo["periodo"].max(),
    })
    _, fin = ctx.ventana
    rfm["recencia_meses"] = meses_hasta(rfm["ultimo_periodo"], fin).clip(lower=0)
    rfm["ticket_medio"] = rfm["monetario"] / rfm["frecuencia"].replace(0, np.nan)
    rfm["log_frecuencia"] = np.log1p(rfm["frecuencia"])
    rfm["log_monetario"] = np.log1p(rfm["monetario"])
    return rfm.reset_index()


def nombrar_segmento(fila: pd.Series) -> str:
    """Etiquetas RFM clásicas, por perfil y no por número de cluster."""
    reciente = fila["recencia"] <= 1
    if reciente and fila["frecuencia"] >= fila["_frecuencia_alta"] and fila["monetario"] >= fila["_monto_alto"]:
        return "Campeones"
    if reciente and fila["frecuencia"] >= fila["_frecuencia_media"]:
        return "Leales"
    if reciente:
        return "Nuevos o de bajo uso"
    if fila["monetario"] >= fila["_monto_alto"]:
        return "En riesgo (alto valor)"
    if fila["frecuencia"] >= fila["_frecuencia_media"]:
        return "Hibernando"
    return "Perdidos"


def ejecutar(ctx: ContextoModelo) -> ModelResult:
    semilla = ctx.semilla
    rfm = construir_rfm(ctx)
    log.info("Personas evaluadas: %s", f"{len(rfm):,}")
    if len(rfm) < 50:
        raise ValueError(f"Solo hay {len(rfm)} personas con actividad PSE; insuficiente para segmentar.")

    X = rfm[FEATURES].to_numpy()
    Xs = StandardScaler().fit_transform(X)

    # Se explora desde k=4: con k=2 la silueta es más alta pero el corte es
    # "gasta / no gasta", que no le sirve a nadie para armar una campaña. Es el mismo
    # criterio de accionabilidad que usan los demás modelos de segmentación.
    k_min = int(ctx.param("k_min", 4))
    k_max = min(int(ctx.param("k_max", 8)), len(rfm) - 1)
    # La silueta se calcula sobre una muestra: es O(n²) y aquí hay decenas de miles
    # de personas. La muestra es fija por semilla, así que el resultado es reproducible.
    muestra = np.random.RandomState(semilla).choice(len(Xs), min(8000, len(Xs)), replace=False)
    seleccion, siluetas = [], []
    for k in range(k_min, k_max + 1):
        km = KMeans(n_clusters=k, n_init=20, random_state=semilla).fit(Xs)
        silueta = float(silhouette_score(Xs[muestra], km.labels_[muestra]))
        siluetas.append(silueta)
        seleccion.append(PuntoSeleccionK(k=k, silhouette=round(silueta, 6),
                                         inertia=round(float(km.inertia_), 3)))
    K = list(range(k_min, k_max + 1))[int(np.argmax(siluetas))]
    log.info("k elegido: %d (silueta %.3f)", K, max(siluetas))

    modelo = KMeans(n_clusters=K, n_init=30, random_state=semilla).fit(Xs)
    rfm["cluster"] = modelo.labels_

    perfil = rfm.groupby("cluster").agg(
        personas=("person_id", "size"),
        recencia=("recencia_meses", "median"),
        frecuencia=("frecuencia", "median"),
        monetario=("monetario", "median"),
        ticket=("ticket_medio", "median"),
        meses_activos=("meses_activos", "median"))
    perfil["_frecuencia_alta"] = rfm["frecuencia"].quantile(0.75)
    perfil["_frecuencia_media"] = rfm["frecuencia"].median()
    perfil["_monto_alto"] = rfm["monetario"].quantile(0.75)

    nombres = nombres_unicos(perfil, nombrar_segmento, "monetario")
    rfm["segmento"] = rfm["cluster"].map(nombres)
    perfil = perfil.drop(columns=[c for c in perfil.columns if c.startswith("_")])

    proyeccion = PCA(n_components=2, random_state=semilla).fit(Xs)
    plano = proyeccion.transform(Xs)

    metricas = limpiar_metricas({
        "silhouette": max(siluetas),
        "davies_bouldin": davies_bouldin_score(Xs, modelo.labels_),
        "inertia": modelo.inertia_,
        "k": K,
        "n_entities": len(rfm),
        "gasto_total": float(rfm["monetario"].sum()),
        "gasto_mediano": float(rfm["monetario"].median()),
        "recencia_mediana": float(rfm["recencia_meses"].median()),
    })
    segmentos = construir_segmentos(rfm["cluster"], nombres, perfil)

    artefactos = {
        "model_uri": guardar_artefacto(ctx, {"kmeans": modelo, "features": FEATURES,
                                             "segmentos": nombres}, "model.joblib"),
        "assignments_uri": guardar_asignaciones(ctx, rfm[[
            "person_id", "cluster", "segmento", "recencia_meses", "frecuencia", "monetario",
            "ticket_medio", "meses_activos"]]),
    }

    return construir_resultado(
        model_id=ctx.config.id, model_name=ctx.config.nombre, catalog_ref=ctx.config.catalogo,
        use_case=ctx.config.caso_uso, task_type=ctx.config.task_type, run_id=ctx.run_id,
        started_at=ctx.started_at, dataset=ctx.dataset_info(len(rfm)),
        params=ctx.params_reportados({"k": K, "algoritmo": "KMeans",
                                      "escalador": "StandardScaler", "features": FEATURES}),
        metrics=metricas, segments=segmentos,
        charts=Charts(
            segment_distribution=segmentos,
            k_selection=seleccion,
            scatter_2d=Scatter2D(
                points=muestra_scatter(plano, rfm["cluster"], nombres, semilla=semilla),
                explained_variance=round(float(proyeccion.explained_variance_ratio_.sum()), 4)),
        ),
        artifacts=artefactos,
        notes=[
            "La segmentación usa solo el canal PSE: el gasto en efectivo o con tarjeta "
            "presencial no es visible, así que subestima la actividad real de consumo.",
            "Recencia medida en meses porque los extractos no traen el día de la transacción.",
        ],
    )


main = cli_para("caso04_rfm_consumidores")

if __name__ == "__main__":
    sys.exit(main())
