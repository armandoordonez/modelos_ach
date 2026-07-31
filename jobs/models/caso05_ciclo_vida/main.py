"""Modelo #46 — Segmentación por ciclo de vida financiero. Caso de uso 5, sector Banca.

A diferencia del modelo de valor (que pregunta *cuánto vale* un cliente hoy), este
pregunta *en qué etapa de su vida financiera está*: informal emergente, asalariado
formal, consolidado que cotiza pensión, independiente o pensionado. La etapa
determina qué producto ofrecerle.

Migrado del notebook ``Caso05_Modelo46_Modelo_Ciclo_Vida_Financiero.ipynb``. El
clustering usa solo señales de Seguridad Social; el flujo y la deuda del
enriquecimiento transaccional se reservan para perfilar los segmentos, no para
formarlos: tienen cobertura parcial y separarían por disponibilidad de dato en vez
de por etapa financiera real.
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

from common.features import perfil_persona_ss, tabla_persona_mes
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
    "ibc_ss", "ibc_volatilidad", "frac_pension", "frac_riesgos",
    "ibc_tendencia", "n_meses_ss", "is_indep", "is_pens",
]
COLUMNA_FINANCIERA = "gasto_Financiero / créditos"
EPSILON = 1e-9


def _tendencia(serie: np.ndarray) -> float:
    """Pendiente relativa del IBC en el tiempo. 0 si no hay historia suficiente."""
    valores = serie[serie > 0]
    if valores.size < 3:
        return 0.0
    return float(np.polyfit(np.arange(valores.size), valores, 1)[0] / (valores.mean() + EPSILON))


def construir_features(ctx: ContextoModelo) -> pd.DataFrame:
    """Señales de etapa financiera por persona, ancladas en PILA."""
    perfil = perfil_persona_ss(storage=ctx.storage, estrategia=ctx.estrategia,
                               settings=ctx.settings).set_index("person_id")

    # Sin filtro de ventana: la dinámica del ingreso necesita toda la historia de PILA
    # (36 meses), no solo los 18 de la ventana común con las fuentes transaccionales.
    ss = tabla_persona_mes("ss", storage=ctx.storage, estrategia=ctx.estrategia,
                           solo_ventana=False, settings=ctx.settings)
    ss = ss.sort_values(["person_id", "periodo"])

    con_ibc = ss[ss["ibc_salud"] > 0]
    dinamica = pd.DataFrame({
        "ibc_volatilidad": con_ibc.groupby("person_id")["ibc_salud"].apply(
            lambda s: s.std() / (s.mean() + EPSILON)),
        "ibc_tendencia": ss.groupby("person_id")["ibc_salud"].apply(
            lambda s: _tendencia(s.to_numpy())),
        "intensidad_laboral": ss.groupby("person_id")["dias_salud"].mean(),
    })
    features = perfil.join(dinamica)
    features["ibc_volatilidad"] = features["ibc_volatilidad"].fillna(0.0)

    # Enriquecimiento transaccional: solo para perfilar.
    trf = tabla_persona_mes("trf", storage=ctx.storage, estrategia=ctx.estrategia,
                            solo_ventana=True, settings=ctx.settings)
    pse = tabla_persona_mes("pse", storage=ctx.storage, estrategia=ctx.estrategia,
                            solo_ventana=True, settings=ctx.settings)
    meses = float(ctx.settings.meses_ventana())
    flujo = ((trf.groupby("person_id")["recibido"].sum()
              - trf.groupby("person_id")["enviado"].sum()) / meses).rename("flujo_neto_mensual")
    features = features.join(flujo)
    if COLUMNA_FINANCIERA in pse.columns:
        por_persona = pse.groupby("person_id")
        deuda = (por_persona[COLUMNA_FINANCIERA].sum()
                 / por_persona["gasto_pse"].sum().replace(0, np.nan)).rename("gasto_financiero_share")
        features = features.join(deuda)
        features["gasto_financiero_share"] = features["gasto_financiero_share"].fillna(0.0)
    features["flujo_neto_mensual"] = features["flujo_neto_mensual"].fillna(0.0)
    features["tiene_transaccional"] = features.index.isin(set(trf["person_id"]) | set(pse["person_id"]))

    features["is_indep"] = (features["tipo_persona"] == "Independiente").astype(int)
    features["is_pens"] = (features["tipo_persona"] == "Pensionado").astype(int)
    return features.reset_index()


def nombrar_etapa(fila: pd.Series) -> str:
    if fila["pens"] >= 0.5:
        return "Pensionados"
    if fila["indep"] >= 0.5:
        return "Independientes"
    if fila["frac_pension"] >= 0.5:
        return "Formal consolidado (cotiza pensión)"
    if fila["ibc"] < 100_000:
        return "Base sin ingreso salud declarado"
    if fila["volatilidad"] >= 0.5 and fila["tendencia"] < 0:
        return "Ingreso volátil en deterioro"
    return "Asalariado formal (salud sin pensión)"


def ejecutar(ctx: ContextoModelo) -> ModelResult:
    semilla = ctx.semilla
    lc = construir_features(ctx)
    log.info("Personas (universo Seguridad Social): %s", f"{len(lc):,}")
    log.info("Con enriquecimiento transaccional: %.1f%%", lc["tiene_transaccional"].mean() * 100)

    Y = lc[COLUMNAS_MODELO].copy()
    Y["ibc_ss"] = np.log1p(Y["ibc_ss"].clip(lower=0))
    Ys = StandardScaler().fit_transform(Y.fillna(0.0))

    k_min, k_max = int(ctx.param("k_min", 4)), int(ctx.param("k_max", 8))
    rango = range(k_min, k_max + 1)
    muestra = np.random.RandomState(semilla).choice(len(Ys), min(8000, len(Ys)), replace=False)
    seleccion, siluetas = [], []
    for k in rango:
        km = KMeans(n_clusters=k, n_init=10, random_state=semilla).fit(Ys)
        silueta = silhouette_score(Ys[muestra], km.labels_[muestra])
        siluetas.append(silueta)
        seleccion.append(PuntoSeleccionK(k=k, silhouette=round(float(silueta), 6),
                                         inertia=round(float(km.inertia_), 3)))
    K = list(rango)[int(np.argmax(siluetas))]
    log.info("k elegido: %d (silueta %.3f)", K, max(siluetas))

    modelo = KMeans(n_clusters=K, n_init=20, random_state=semilla).fit(Ys)
    lc["cluster"] = modelo.labels_

    perfil = lc.groupby("cluster").agg(
        personas=("person_id", "size"), ibc=("ibc_ss", "median"),
        frac_pension=("frac_pension", "median"), frac_riesgos=("frac_riesgos", "median"),
        tendencia=("ibc_tendencia", "median"), volatilidad=("ibc_volatilidad", "median"),
        indep=("is_indep", "mean"), pens=("is_pens", "mean"), meses=("n_meses_ss", "median"))

    nombres = nombres_unicos(perfil, nombrar_etapa, "ibc")
    lc["etapa"] = lc["cluster"].map(nombres)

    enriquecidos = lc[lc["tiene_transaccional"]]
    perfil_tx = enriquecidos.groupby("cluster").agg(
        flujo_neto_mensual=("flujo_neto_mensual", "median"),
        carga_deuda=("gasto_financiero_share", "median") if "gasto_financiero_share" in lc.columns
        else ("flujo_neto_mensual", "median"))
    perfil = perfil.join(perfil_tx)

    proyeccion = PCA(n_components=2, random_state=semilla).fit(Ys)
    plano = proyeccion.transform(Ys)

    metricas = limpiar_metricas({
        "silhouette": max(siluetas),
        "davies_bouldin": davies_bouldin_score(Ys, modelo.labels_),
        "inertia": modelo.inertia_,
        "k": K,
        "n_entities": len(lc),
        "pct_con_transaccional": float(lc["tiene_transaccional"].mean() * 100),
        "pct_cotiza_pension": float(lc["cotiza_pension"].mean() * 100),
    })
    segmentos = construir_segmentos(lc["cluster"], nombres, perfil)

    columnas_salida = ["person_id", "cluster", "etapa", "tipo_persona", "ibc_ss",
                       "frac_pension", "ibc_tendencia", "ibc_volatilidad", "n_meses_ss",
                       "flujo_neto_mensual", "tiene_transaccional"]
    artefactos = {
        "model_uri": guardar_artefacto(ctx, {"kmeans": modelo, "features": COLUMNAS_MODELO,
                                             "segmentos": nombres}, "model.joblib"),
        "assignments_uri": guardar_asignaciones(
            ctx, lc[[c for c in columnas_salida if c in lc.columns]]),
    }

    return construir_resultado(
        model_id=ctx.config.id, model_name=ctx.config.nombre, catalog_ref=ctx.config.catalogo,
        use_case=ctx.config.caso_uso, task_type=ctx.config.task_type, run_id=ctx.run_id,
        started_at=ctx.started_at, dataset=ctx.dataset_info(len(lc)),
        params=ctx.params_reportados({"k": K, "algoritmo": "KMeans", "escalador": "StandardScaler",
                                      "features": COLUMNAS_MODELO}),
        metrics=metricas, segments=segmentos,
        charts=Charts(
            segment_distribution=segmentos,
            k_selection=seleccion,
            scatter_2d=Scatter2D(
                points=muestra_scatter(plano, lc["cluster"], nombres, semilla=semilla),
                explained_variance=round(float(proyeccion.explained_variance_ratio_.sum()), 4)),
        ),
        artifacts=artefactos,
        notes=[
            "Sin fecha de nacimiento, la etapa se infiere del comportamiento financiero, "
            "no de la edad.",
            "El clustering usa solo señales de PILA; flujo y deuda perfilan los segmentos "
            "pero no los forman, porque su cobertura es parcial.",
        ],
    )


main = cli_para("caso05_ciclo_vida")

if __name__ == "__main__":
    sys.exit(main())
