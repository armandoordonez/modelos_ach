"""Modelo #6 — Estimador de ingresos para independientes. Caso de uso 2.

El IBC declarado puede decir un salario mínimo mientras las transferencias y el gasto
PSE apuntan a tres o cinco veces esa cifra. Este modelo estima el ingreso real a
partir de la señal transaccional y, sobre esa estimación, deriva dos indicadores:

* **#8B Brecha de subdeclaración** — regla de umbral sobre la predicción conservadora
  (p10). No entrena nada: es una lectura de #6.
* **#41 Capacidad de ahorro (FCM)** — score determinista de percentiles ponderados.

Ninguno de los dos es un modelo entrenado, así que no tienen job propio: viajan como
bloques derivados dentro de este JSON.

Migrado de ``notebooks/exploration/scripts_originales/caso02_ingresos_independientes.py``
con dos cambios deliberados, ambos declarados en el resultado:

1. **Llave de persona unificada a la cédula ofuscada.** El script cruzaba por
   ``Nombre|Documento``, que produce un universo distinto. Con ``ACH_LINEAGE=legacy``
   se reproduce la llave original para verificar la migración.
2. **La ventana queda declarada en el registro** (2025-01 a 2025-09, donde las tres
   fuentes están intactas) en vez de estar fija en el código.
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split

from common.features import columnas_gasto, tabla_persona_mes
from common.results import (
    Charts,
    ImportanciaVariable,
    ModelResult,
    construir_resultado,
)
from models.base import (
    ContextoModelo,
    cli_para,
    guardar_artefacto,
    guardar_asignaciones,
    limpiar_metricas,
)

log = logging.getLogger(__name__)

EPSILON = 1.0
COLUMNA_FINANCIERA = "gasto_Financiero / créditos"
COLUMNA_SERVICIOS = "gasto_Telco / servicios públicos"


def _agregar_por_persona(panel: pd.DataFrame, columnas: list[str], prefijo: str) -> pd.DataFrame:
    """Media, mediana, desviación y máximo de cada variable por persona."""
    grupo = panel.groupby("person_id")
    salida = pd.DataFrame(index=grupo.size().index)
    salida[f"{prefijo}meses"] = grupo.size()
    for columna in columnas:
        salida[f"{prefijo}{columna}_med"] = grupo[columna].mean()
        salida[f"{prefijo}{columna}_p50"] = grupo[columna].median()
        salida[f"{prefijo}{columna}_std"] = grupo[columna].std().fillna(0)
        salida[f"{prefijo}{columna}_max"] = grupo[columna].max()
    return salida


def construir_matriz(ctx: ContextoModelo) -> tuple[pd.DataFrame, list[str], dict[str, pd.DataFrame]]:
    """Matriz persona × variables. Las predictoras son SOLO transaccionales."""
    ajustes = ctx.settings_ventana
    ss = tabla_persona_mes("ss", storage=ctx.storage, estrategia=ctx.estrategia,
                           solo_ventana=True, settings=ajustes)
    trf = tabla_persona_mes("trf", storage=ctx.storage, estrategia=ctx.estrategia,
                            solo_ventana=True, settings=ajustes)
    pse = tabla_persona_mes("pse", storage=ctx.storage, estrategia=ctx.estrategia,
                            solo_ventana=True, settings=ajustes)

    categorias = [c for c in columnas_gasto() if c in pse.columns]
    f_ss = _agregar_por_persona(ss, ["ibc_salud", "ibc_pension", "salario_basico", "dias_salud"], "ss_")
    f_trf = _agregar_por_persona(trf, ["recibido", "enviado", "n_recibidas", "n_enviadas"], "ach_")
    f_pse = _agregar_por_persona(pse, ["gasto_pse", "n_pagos", "n_comercios"] + categorias, "pse_")

    m = f_ss.join(f_trf, how="outer").join(f_pse, how="outer")
    m["en_ss"] = m.index.isin(f_ss.index).astype(int)
    m["en_ach"] = m.index.isin(f_trf.index).astype(int)
    m["en_pse"] = m.index.isin(f_pse.index).astype(int)
    for columna in ("ss_meses", "ach_meses", "pse_meses"):
        m[columna] = m[columna].fillna(0)
    numericas = m.select_dtypes(include=[np.number]).columns
    m[numericas] = m[numericas].fillna(0)

    meses = float(ctx.meses_ventana)
    m["ratio_env_rec"] = m["ach_enviado_med"] / (m["ach_recibido_med"] + EPSILON)
    m["ratio_gasto_rec"] = m["pse_gasto_pse_med"] / (m["ach_recibido_med"] + EPSILON)
    m["cv_recibido"] = m["ach_recibido_std"] / (m["ach_recibido_med"] + EPSILON)
    m["cv_gasto"] = m["pse_gasto_pse_std"] / (m["pse_gasto_pse_med"] + EPSILON)
    m["intensidad_ach"] = m["ach_meses"] / meses
    m["intensidad_pse"] = m["pse_meses"] / meses
    m["carga_financiera"] = m.get(f"pse_{COLUMNA_FINANCIERA}_med", 0) / (m["pse_gasto_pse_med"] + EPSILON)
    m["peso_servicios"] = m.get(f"pse_{COLUMNA_SERVICIOS}_med", 0) / (m["pse_gasto_pse_med"] + EPSILON)
    m["flujo_neto"] = m["ach_recibido_med"] - m["ach_enviado_med"]

    # Tipo de persona desde PILA, para segmentar los resultados.
    from common.features import cargar_fuente

    ss_raw = cargar_fuente("ss", storage=ctx.storage, estrategia=ctx.estrategia,
                           solo_ventana=True, settings=ajustes)
    relacion = ss_raw["Relación laboral"].astype("string").str.upper()
    planilla = ss_raw["Tipo planilla"].astype("string").str.upper()
    marcas = pd.DataFrame({
        "person_id": ss_raw["person_id"],
        "es_independiente": relacion.str.contains("INDEPENDIENTE", na=False).fillna(False),
        "es_dependiente": relacion.eq("DEPENDIENTE").fillna(False),
        "es_pensionado": relacion.str.contains("PENSIONADO", na=False).fillna(False),
        "planilla_E": planilla.eq("E").fillna(False),
    }).groupby("person_id").max()
    m = m.join(marcas)
    for columna in ("es_independiente", "es_dependiente", "es_pensionado", "planilla_E"):
        m[columna] = m[columna].fillna(False).astype(bool)

    # Las predictoras excluyen a propósito todo lo que venga de Seguridad Social:
    # el IBC declarado es justamente el objetivo, y meterlo sería fuga de información.
    predictoras = [
        c for c in m.columns
        if (c.startswith("ach_") or c.startswith("pse_") or c in (
            "ratio_env_rec", "ratio_gasto_rec", "cv_recibido", "cv_gasto", "intensidad_ach",
            "intensidad_pse", "carga_financiera", "peso_servicios", "flujo_neto",
            "en_ach", "en_pse"))
        and m[c].dtype.kind in "if"
    ]
    return m, predictoras, {"ss": ss, "trf": trf, "pse": pse}


def entrenar_modelo_6(ctx: ContextoModelo, m: pd.DataFrame, predictoras: list[str]):
    """Entrena sobre asalariados formales, donde el IBC sí refleja el ingreso real."""
    smlmv = float(ctx.param("smlmv", 1_423_500))
    semilla = ctx.semilla

    entrenables = (
        (m["en_ss"] == 1) & m["es_dependiente"] & m["planilla_E"]
        & (m["ss_dias_salud_p50"] >= 28) & (m["ss_meses"] >= 4)
        & (m["ss_ibc_salud_p50"] >= smlmv * 0.9)
        & ((m["en_ach"] == 1) | (m["en_pse"] == 1))
    )
    entrenamiento = m[entrenables]
    if len(entrenamiento) < 200:
        raise ValueError(
            f"Solo {len(entrenamiento)} personas cumplen el filtro de entrenamiento; "
            "se necesitan al menos 200 para que la estimación tenga sentido."
        )
    log.info("Población de entrenamiento: %s personas", f"{len(entrenamiento):,}")

    X = entrenamiento[predictoras].to_numpy()
    y = np.log(entrenamiento["ss_ibc_salud_p50"].to_numpy())
    X_tr, X_te, y_tr, y_te = train_test_split(
        X, y, test_size=float(ctx.param("test_size", 0.25)), random_state=semilla)

    modelo = HistGradientBoostingRegressor(
        max_iter=int(ctx.param("max_iter", 400)),
        learning_rate=float(ctx.param("learning_rate", 0.05)),
        max_depth=int(ctx.param("max_depth", 6)),
        min_samples_leaf=int(ctx.param("min_samples_leaf", 40)),
        l2_regularization=float(ctx.param("l2_regularization", 1.0)),
        random_state=semilla).fit(X_tr, y_tr)

    prediccion = modelo.predict(X_te)
    real, estimado = np.exp(y_te), np.exp(prediccion)
    base = np.full_like(y_te, np.median(y_tr))

    cuantilicos = {}
    for etiqueta, cuantil in (("p10", 0.1), ("p90", 0.9)):
        cuantilicos[etiqueta] = HistGradientBoostingRegressor(
            loss="quantile", quantile=cuantil, max_iter=300, learning_rate=0.05,
            min_samples_leaf=40, random_state=semilla).fit(X_tr, y_tr)
    cobertura = float(np.mean((y_te >= cuantilicos["p10"].predict(X_te))
                              & (y_te <= cuantilicos["p90"].predict(X_te))))

    importancia = permutation_importance(modelo, X_te, y_te, n_repeats=5,
                                         random_state=semilla, n_jobs=1)
    top = (pd.Series(importancia.importances_mean, index=predictoras)
           .sort_values(ascending=False).head(15))

    desempeno = {
        "r2_log": float(r2_score(y_te, prediccion)),
        "mape": float(np.mean(np.abs(real - estimado) / real)),
        "mae_cop": float(mean_absolute_error(real, estimado)),
        "r2_log_baseline": float(r2_score(y_te, base)),
        "cobertura_banda_p10_p90": cobertura,
        "n_entrenamiento": int(len(entrenamiento)),
    }
    return modelo, cuantilicos, desempeno, top, (y_te, prediccion)


def derivar_8b(ctx: ContextoModelo, m: pd.DataFrame, predictoras: list[str],
               modelo, modelo_p10) -> tuple[pd.DataFrame, dict]:
    """#8B: marca subdeclaración cuando incluso la estimación conservadora supera
    con holgura al IBC declarado y este está pegado al salario mínimo."""
    smlmv = float(ctx.param("smlmv", 1_423_500))
    cotizantes = m[(m["en_ss"] == 1) & (m["ss_ibc_salud_p50"] > 0)
                   & ((m["en_ach"] == 1) | (m["en_pse"] == 1))].copy()
    X = cotizantes[predictoras].to_numpy()
    cotizantes["ingreso_estimado"] = np.exp(modelo.predict(X))
    cotizantes["ingreso_est_p10"] = np.exp(modelo_p10.predict(X))
    cotizantes["brecha"] = cotizantes["ingreso_estimado"] / cotizantes["ss_ibc_salud_p50"]
    cotizantes["brecha_conservadora"] = cotizantes["ingreso_est_p10"] / cotizantes["ss_ibc_salud_p50"]
    cotizantes["marca_brecha"] = ((cotizantes["brecha_conservadora"] > 1.2)
                                  & (cotizantes["ss_ibc_salud_p50"] <= 1.1 * smlmv)).astype(int)

    resumen = {
        "cotizantes_evaluados": int(len(cotizantes)),
        "pct_marcados_subdeclaracion": float(cotizantes["marca_brecha"].mean() * 100),
        "brecha_mediana": float(cotizantes["brecha"].median()),
    }
    for etiqueta, mascara in (("dependiente", cotizantes["es_dependiente"]),
                              ("independiente", cotizantes["es_independiente"]),
                              ("pensionado", cotizantes["es_pensionado"])):
        subconjunto = cotizantes[mascara]
        if len(subconjunto) > 50:
            resumen[f"brecha_mediana_{etiqueta}"] = float(subconjunto["brecha"].median())
    return cotizantes, resumen


def derivar_41(ctx: ContextoModelo, paneles: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, dict]:
    """#41: score de capacidad de ahorro sobre el flujo de caja mensual."""
    trf, pse, ss = paneles["trf"], paneles["pse"], paneles["ss"]
    columnas_pse = ["person_id", "periodo", "gasto_pse"]
    for columna in (COLUMNA_FINANCIERA, COLUMNA_SERVICIOS):
        if columna in pse.columns:
            columnas_pse.append(columna)

    panel = (trf[["person_id", "periodo", "recibido", "enviado"]]
             .merge(pse[columnas_pse], on=["person_id", "periodo"], how="outer")
             .merge(ss[["person_id", "periodo", "ibc_salud"]], on=["person_id", "periodo"], how="outer")
             .fillna(0))

    # Máximo, no suma: el salario que además pasa por ACH se contaría dos veces.
    panel["ingreso"] = panel[["ibc_salud", "recibido"]].max(axis=1)
    panel["fcm"] = (panel["ingreso"] - panel.get(COLUMNA_FINANCIERA, 0)
                    - panel.get(COLUMNA_SERVICIOS, 0) - panel["enviado"])

    grupo = panel.groupby("person_id")
    ahorro = pd.DataFrame({
        "fcm_med": grupo["fcm"].mean(),
        "fcm_p50": grupo["fcm"].median(),
        "fcm_std": grupo["fcm"].std().fillna(0),
        "meses_obs": grupo.size(),
        "prop_meses_positivos": grupo["fcm"].apply(lambda s: float((s > 0).mean())),
        "ingreso_med": grupo["ingreso"].mean(),
    })
    ahorro["margen"] = ahorro["fcm_med"] / (ahorro["ingreso_med"] + EPSILON)
    ahorro["estabilidad"] = 1 - (ahorro["fcm_std"] / (ahorro["fcm_med"].abs() + EPSILON)).clip(0, 3) / 3
    ahorro["score_41"] = (100 * (
        0.40 * ahorro["fcm_med"].clip(lower=0).rank(pct=True)
        + 0.30 * ahorro["prop_meses_positivos"]
        + 0.20 * ahorro["estabilidad"]
        + 0.10 * ahorro["margen"].clip(-1, 1).rank(pct=True)
    )).round(1)
    ahorro["banda_ahorro"] = pd.cut(ahorro["score_41"], [0, 20, 40, 60, 80, 100],
                                    labels=["muy baja", "baja", "media", "alta", "muy alta"])

    # La validación del score es su monotonicidad: bandas más altas, mejor FCM.
    por_banda = ahorro.groupby("banda_ahorro", observed=True)["fcm_med"].median()
    monotono = bool(por_banda.is_monotonic_increasing)
    resumen = {
        "personas_evaluadas_ahorro": int(len(ahorro)),
        "fcm_mensual_mediano": float(ahorro["fcm_p50"].median()),
        "pct_fcm_positivo": float((ahorro["fcm_med"] > 0).mean() * 100),
        "bandas_ahorro_monotonas": float(monotono),
    }
    return ahorro, resumen


def ejecutar(ctx: ContextoModelo) -> ModelResult:
    m, predictoras, paneles = construir_matriz(ctx)
    log.info("Matriz base: %s personas × %d predictoras", f"{len(m):,}", len(predictoras))

    modelo, cuantilicos, desempeno, importancia, (y_test, y_pred) = entrenar_modelo_6(
        ctx, m, predictoras)
    cotizantes, resumen_8b = derivar_8b(ctx, m, predictoras, modelo, cuantilicos["p10"])
    ahorro, resumen_41 = derivar_41(ctx, paneles)

    independientes = m[m["es_independiente"] & ((m["en_ach"] == 1) | (m["en_pse"] == 1))]
    brecha_independientes = float("nan")
    if len(independientes):
        estimado = np.exp(modelo.predict(independientes[predictoras].to_numpy()))
        declarado = independientes["ss_ibc_salud_p50"].replace(0, np.nan)
        brecha_independientes = float(np.nanmedian(estimado / declarado))

    universo = m[(m["en_ach"] == 1) | (m["en_pse"] == 1)].copy()
    X_universo = universo[predictoras].to_numpy()
    universo["ingreso_estimado"] = np.exp(modelo.predict(X_universo))
    universo["ingreso_est_p10"] = np.exp(cuantilicos["p10"].predict(X_universo))
    universo["ingreso_est_p90"] = np.exp(cuantilicos["p90"].predict(X_universo))
    universo["brecha_declaracion"] = (universo["ingreso_estimado"]
                                      / universo["ss_ibc_salud_p50"].replace(0, np.nan))
    universo = universo.join(ahorro[["score_41", "fcm_med", "prop_meses_positivos"]], how="left")
    universo["segmento"] = np.select(
        [universo["es_independiente"], universo["es_dependiente"], universo["es_pensionado"],
         universo["en_ss"] == 0],
        ["independiente", "dependiente", "pensionado", "sin_registro_ss"], default="otro")

    metricas = limpiar_metricas({
        **desempeno, **resumen_8b, **resumen_41,
        "n_entities": len(universo),
        "brecha_mediana_independientes": brecha_independientes,
    })

    salida = universo.reset_index()[[
        "person_id", "segmento", "ingreso_estimado", "ingreso_est_p10", "ingreso_est_p90",
        "ss_ibc_salud_p50", "brecha_declaracion", "score_41", "fcm_med", "en_ss"]].rename(
        columns={"ss_ibc_salud_p50": "ibc_declarado", "fcm_med": "fcm_mensual",
                 "score_41": "score_capacidad_ahorro"})

    artefactos = {
        "model_uri": guardar_artefacto(ctx, {
            "regresor": modelo, "p10": cuantilicos["p10"], "p90": cuantilicos["p90"],
            "features": predictoras}, "model.joblib"),
        "assignments_uri": guardar_asignaciones(ctx, salida),
    }

    residuales = {
        "actual": [round(float(v), 4) for v in y_test[:1000]],
        "predicted": [round(float(v), 4) for v in y_pred[:1000]],
    }

    return construir_resultado(
        model_id=ctx.config.id, model_name=ctx.config.nombre, catalog_ref=ctx.config.catalogo,
        use_case=ctx.config.caso_uso, task_type=ctx.config.task_type, run_id=ctx.run_id,
        started_at=ctx.started_at, dataset=ctx.dataset_info(len(universo)),
        params=ctx.params_reportados({
            "algoritmo": "HistGradientBoostingRegressor",
            "n_predictoras": len(predictoras),
            "objetivo": "log(IBC de salud declarado)",
            "derivados": ["#8B brecha de subdeclaración", "#41 capacidad de ahorro"],
        }),
        metrics=metricas,
        charts=Charts(
            feature_importance=[
                ImportanciaVariable(feature=nombre, importance=round(float(valor), 6))
                for nombre, valor in importancia.items()
            ],
            residuals=residuales,
        ),
        artifacts=artefactos,
        notes=[
            "#8B y #41 no son modelos entrenados: el primero es una regla sobre la salida de "
            "#6 y el segundo un score determinista. Por eso viajan aquí y no en jobs aparte.",
            "El modelo se entrena en asalariados formales, donde el IBC sí refleja el ingreso, "
            "y se aplica a independientes: es una transferencia de dominio, no una validación "
            "sobre la población objetivo.",
            "Las predictoras excluyen toda variable de Seguridad Social para no filtrar el "
            "objetivo.",
        ],
    )


main = cli_para("caso02_ingresos_independientes")

if __name__ == "__main__":
    sys.exit(main())
