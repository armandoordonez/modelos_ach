"""Modelo #4 — Segmentación de clientes por valor (CLV). Caso de uso 5, sector Banca.

Agrupa a las personas por el valor transaccional que ACH observa: dinero realmente
movido (recibido + enviado + gasto PSE), frecuencia, permanencia y recencia,
enriquecido con el ingreso declarado en PILA.

Migrado del notebook ``Caso05_Modelo4_CLV_Clustering.ipynb`` sin cambios de método:
mismas features, misma semilla, mismo criterio de k. Su paridad se verifica en
``tests/parity``.
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

from common.features import columnas_gasto, meses_hasta, perfil_persona_ss, tabla_persona_mes
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

COLUMNAS_MODELO = [
    "recibido_total", "enviado_total", "gasto_total", "throughput",
    "n_transacciones", "meses_activos", "recencia_meses",
    "transfiya_share", "gasto_financiero_share", "ibc_ss", "clv_proxy",
]
COLUMNAS_LOG = [
    "recibido_total", "enviado_total", "gasto_total", "throughput",
    "n_transacciones", "ibc_ss", "clv_proxy",
]
COLUMNA_FINANCIERA = "gasto_Financiero / créditos"


def construir_features(ctx: ContextoModelo) -> pd.DataFrame:
    """Tabla persona × features de valor, cruzando las tres fuentes por la llave vigente.

    ``recibido``/``enviado`` ya excluyen cuenta propia: mover dinero entre cuentas
    propias no es ingreso ni gasto, y contarlo inflaba el valor del cliente.
    """
    trf = tabla_persona_mes("trf", storage=ctx.storage, estrategia=ctx.estrategia,
                            solo_ventana=True, settings=ctx.settings)
    pse = tabla_persona_mes("pse", storage=ctx.storage, estrategia=ctx.estrategia,
                            solo_ventana=True, settings=ctx.settings)

    por_trf = trf.groupby("person_id").agg(
        recibido_total=("recibido", "sum"), enviado_total=("enviado", "sum"),
        cuenta_propia_total=("cuenta_propia_total", "sum"),
        n_recibidas=("n_recibidas", "sum"), n_enviadas=("n_enviadas", "sum"),
        recibido_transfiya=("recibido_transfiya", "sum"),
        meses_trf=("periodo", "nunique"), ult_periodo_trf=("periodo", "max"))

    categorias = [c for c in columnas_gasto() if c in pse.columns]
    por_pse = pse.groupby("person_id").agg(
        gasto_total=("gasto_pse", "sum"), n_pagos=("n_pagos", "sum"),
        meses_pse=("periodo", "nunique"), ult_periodo_pse=("periodo", "max"))
    gasto_categoria = pse.groupby("person_id")[categorias].sum()

    f = por_trf.join(por_pse, how="outer").join(gasto_categoria, how="outer")
    en_cero = ["recibido_total", "enviado_total", "cuenta_propia_total", "n_recibidas",
               "n_enviadas", "recibido_transfiya", "gasto_total", "n_pagos", *categorias]
    f[en_cero] = f[en_cero].fillna(0.0)
    f[["meses_trf", "meses_pse"]] = f[["meses_trf", "meses_pse"]].fillna(0)

    perfil_ss = perfil_persona_ss(storage=ctx.storage, estrategia=ctx.estrategia,
                                  settings=ctx.settings).set_index("person_id")
    f = f.join(perfil_ss[["ibc_ss", "tipo_persona", "prestaciones_completas", "cotiza_pension"]],
               how="left")
    f["ibc_ss"] = f["ibc_ss"].fillna(0.0)
    f["tipo_persona"] = f["tipo_persona"].fillna("Sin PILA")
    for columna in ("prestaciones_completas", "cotiza_pension"):
        f[columna] = f[columna].astype("boolean").fillna(False).astype(bool)

    meses = float(ctx.settings.meses_ventana())
    f["flujo_neto"] = f["recibido_total"] - f["enviado_total"]
    f["throughput"] = f["recibido_total"] + f["enviado_total"] + f["gasto_total"]
    f["meses_activos"] = f[["meses_trf", "meses_pse"]].max(axis=1)
    f["n_transacciones"] = f["n_recibidas"] + f["n_enviadas"] + f["n_pagos"]
    f["ticket_medio"] = f["throughput"] / f["n_transacciones"].replace(0, np.nan)
    f["transfiya_share"] = f["recibido_transfiya"] / f["recibido_total"].replace(0, np.nan)
    f["gasto_financiero_share"] = f.get(COLUMNA_FINANCIERA, 0) / f["gasto_total"].replace(0, np.nan)
    ultimo = f[["ult_periodo_trf", "ult_periodo_pse"]].max(axis=1)
    f["recencia_meses"] = meses_hasta(ultimo, ctx.settings.window_end).clip(lower=0)
    f["clv_proxy"] = (f["throughput"] / meses) * (f["meses_activos"] / meses)
    f[["transfiya_share", "gasto_financiero_share", "ticket_medio"]] = \
        f[["transfiya_share", "gasto_financiero_share", "ticket_medio"]].fillna(0.0)
    return f.reset_index()


def nombrar_segmento(fila: pd.Series) -> str:
    """Etiqueta por perfil, no por número de cluster."""
    es_solo_pse = fila["gasto_pse"] > 0 and fila["recibido"] < 0.15 * fila["gasto_pse"]
    en_fuga = fila["recencia"] >= 2 or fila["meses_activos"] <= 6
    if es_solo_pse:
        return "Gastador digital (solo PSE)"
    if en_fuga:
        return "Bajo valor / en fuga"
    nivel = "Alto valor" if fila["_es_top"] else "Valor medio"
    return f"{nivel} · {'ingreso formal' if fila['ibc'] > 0 else 'informal'}"


def ejecutar(ctx: ContextoModelo) -> ModelResult:
    semilla = ctx.semilla
    clv = construir_features(ctx)
    log.info("Personas: %s", f"{len(clv):,}")

    X = clv[COLUMNAS_MODELO].copy()
    for columna in COLUMNAS_LOG:
        X[columna] = np.log1p(X[columna])
    Xs = StandardScaler().fit_transform(X.fillna(0.0))

    # Se explora k >= 4: k=2 o 3 maximizan la silueta pero son cortes gruesos que el
    # negocio no puede accionar.
    k_min, k_max = int(ctx.param("k_min", 4)), int(ctx.param("k_max", 8))
    rango = range(k_min, k_max + 1)
    muestra = np.random.RandomState(semilla).choice(len(Xs), min(8000, len(Xs)), replace=False)
    seleccion, inercias, siluetas = [], [], []
    for k in rango:
        km = KMeans(n_clusters=k, n_init=10, random_state=semilla).fit(Xs)
        silueta = silhouette_score(Xs[muestra], km.labels_[muestra])
        inercias.append(km.inertia_)
        siluetas.append(silueta)
        seleccion.append(PuntoSeleccionK(k=k, silhouette=round(float(silueta), 6),
                                         inertia=round(float(km.inertia_), 3)))
    K = list(rango)[int(np.argmax(siluetas))]
    log.info("k elegido: %d (silueta %.3f)", K, max(siluetas))

    modelo = KMeans(n_clusters=K, n_init=20, random_state=semilla).fit(Xs)
    clv["cluster"] = modelo.labels_

    perfil = clv.groupby("cluster").agg(
        personas=("person_id", "size"),
        recibido=("recibido_total", "median"), gasto_pse=("gasto_total", "median"),
        throughput=("throughput", "median"), ibc=("ibc_ss", "median"),
        n_tx=("n_transacciones", "median"), meses_activos=("meses_activos", "median"),
        recencia=("recencia_meses", "median"), clv_proxy=("clv_proxy", "median"))

    activos = [cl for cl in perfil.index
               if not (perfil.loc[cl, "gasto_pse"] > 0
                       and perfil.loc[cl, "recibido"] < 0.15 * perfil.loc[cl, "gasto_pse"])
               and not (perfil.loc[cl, "recencia"] >= 2 or perfil.loc[cl, "meses_activos"] <= 6)]
    top = (perfil.loc[activos].sort_values("throughput", ascending=False).index.tolist()[:1]
           if activos else [])
    perfil["_es_top"] = [cl in top for cl in perfil.index]

    nombres = nombres_unicos(perfil, nombrar_segmento, "throughput")
    clv["segmento"] = clv["cluster"].map(nombres)
    perfil = perfil.drop(columns=["_es_top"])

    proyeccion = PCA(n_components=2, random_state=semilla).fit(Xs)
    plano = proyeccion.transform(Xs)

    metricas = limpiar_metricas({
        "silhouette": max(siluetas),
        "davies_bouldin": davies_bouldin_score(Xs, modelo.labels_),
        "inertia": modelo.inertia_,
        "k": K,
        "n_entities": len(clv),
        "clv_total": float(clv["clv_proxy"].sum()),
        "clv_mediano": float(clv["clv_proxy"].median()),
        "pct_con_ingreso_formal": float((clv["ibc_ss"] > 0).mean() * 100),
    })

    segmentos = construir_segmentos(clv["cluster"], nombres, perfil)
    artefactos = {
        "model_uri": guardar_artefacto(ctx, {"kmeans": modelo, "features": COLUMNAS_MODELO,
                                             "log_features": COLUMNAS_LOG, "segmentos": nombres},
                                       "model.joblib"),
        "assignments_uri": guardar_asignaciones(
            ctx, clv[["person_id", "cluster", "segmento", "tipo_persona", "clv_proxy",
                      "throughput", "ibc_ss", "meses_activos", "recencia_meses"]]),
    }

    return construir_resultado(
        model_id=ctx.config.id, model_name=ctx.config.nombre, catalog_ref=ctx.config.catalogo,
        use_case=ctx.config.caso_uso, task_type=ctx.config.task_type, run_id=ctx.run_id,
        started_at=ctx.started_at, dataset=ctx.dataset_info(len(clv)),
        params=ctx.params_reportados({"k": K, "algoritmo": "KMeans", "escalador": "StandardScaler",
                                      "features": COLUMNAS_MODELO}),
        metrics=metricas, segments=segmentos,
        charts=Charts(
            segment_distribution=segmentos,
            k_selection=seleccion,
            scatter_2d=Scatter2D(
                points=muestra_scatter(plano, clv["cluster"], nombres, semilla=semilla),
                explained_variance=round(float(proyeccion.explained_variance_ratio_.sum()), 4)),
        ),
        artifacts=artefactos,
        notes=[
            "Transferencias y PSE vienen truncados en el tope de filas de Excel: los volúmenes "
            "son relativos, no censales.",
            "El CLV-proxy pondera dinero movido por permanencia; no incorpora márgenes reales.",
        ],
    )


main = cli_para("caso05_clv")

if __name__ == "__main__":
    sys.exit(main())
