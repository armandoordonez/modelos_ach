"""Modelos #17 y #55 — Propensión de compra por categoría. Caso de uso 4.

Un único módulo parametrizado por categoría: ``#17`` lo instancia sobre salud y
``#55`` sobre viajes y turismo. Agregar una tercera categoría es agregar una entrada
a ``models_config.yml``, sin escribir código.

**Separación temporal, no aleatoria.** Las variables se calculan con los primeros
meses de la ventana y la etiqueta —si la persona gastó o no en la categoría— se mide
en los últimos, que el modelo nunca ve al entrenar. Eso es lo que lo hace un modelo
de propensión y no una descripción del presente.

**Corrección de una fuga que traía el script original.** El notebook del que salió
excluía a propósito la categoría objetivo de las variables predictoras
(``excluir_categoria``); al migrarlo a script esa exclusión se perdió, de modo que el
gasto pasado *en salud* entraba a predecir el gasto futuro *en salud*. Aquí se
restaura: ``_columnas_de_la_categoria`` saca del predictor toda columna derivada de
la categoría que se está prediciendo. El AUC baja, y es lo correcto.

Se reporta además el **AUC restringido a nuevos adoptantes** (personas sin gasto
previo en la categoría), que es la métrica que distingue un modelo que encuentra
clientes nuevos de uno que solo confirma la persistencia de los que ya estaban.
"""

from __future__ import annotations

import logging
import sys
import unicodedata

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

from common.features import columnas_gasto, meses_hasta, perfil_persona_ss, tabla_persona_mes
from common.results import (
    Charts,
    CurvaROC,
    ImportanciaVariable,
    MatrizConfusion,
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


def _sin_tildes(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")


def _sufijo(categoria: str) -> str:
    """Sufijo de columna a partir del nombre de la categoría."""
    limpio = _sin_tildes(categoria).lower()
    for caracter in (" ", "/", "-", ","):
        limpio = limpio.replace(caracter, "_")
    return "_".join(parte for parte in limpio.split("_") if parte)


def _columnas_de_la_categoria(columnas: list[str], categoria: str) -> list[str]:
    """Columnas derivadas de la categoría objetivo: son las que hay que excluir."""
    marca = _sufijo(categoria)
    return [c for c in columnas if marca in _sufijo(c)]


def construir_paneles(ctx: ContextoModelo, categoria: str):
    """Panel histórico de features y etiqueta futura, separados en el tiempo."""
    ajustes = ctx.settings_ventana
    meses_objetivo = int(ctx.param("meses_objetivo", 4))

    pse = tabla_persona_mes("pse", storage=ctx.storage, estrategia=ctx.estrategia,
                            solo_ventana=True, settings=ajustes)
    trf = tabla_persona_mes("trf", storage=ctx.storage, estrategia=ctx.estrategia,
                            solo_ventana=True, settings=ajustes)

    periodos = sorted(pse["periodo"].dropna().unique())
    if len(periodos) <= meses_objetivo:
        raise ValueError(
            f"La ventana solo tiene {len(periodos)} periodos y se piden {meses_objetivo} "
            "para la etiqueta. Amplía la ventana o baja meses_objetivo."
        )
    corte = periodos[-meses_objetivo]
    historico = pse[pse["periodo"] < corte]
    objetivo = pse[pse["periodo"] >= corte]
    log.info("Histórico: %s a %s · objetivo: %s a %s",
             periodos[0], periodos[-meses_objetivo - 1], corte, periodos[-1])

    columna_categoria = f"gasto_{categoria}"
    if columna_categoria not in pse.columns:
        disponibles = [c.replace("gasto_", "") for c in columnas_gasto() if c in pse.columns]
        raise ValueError(
            f"La categoría {categoria!r} no aparece en el dataset. Disponibles: {disponibles}"
        )

    grupo = historico.groupby("person_id")
    categorias_presentes = [c for c in columnas_gasto() if c in historico.columns]
    features = pd.DataFrame({
        "pse_meses": grupo["periodo"].nunique(),
        "gasto_total": grupo["gasto_pse"].sum(),
        "gasto_medio": grupo["gasto_pse"].mean(),
        "gasto_std": grupo["gasto_pse"].std().fillna(0.0),
        "gasto_max": grupo["gasto_pse"].max(),
        "n_pagos": grupo["n_pagos"].sum(),
        "n_comercios": grupo["n_comercios"].sum(),
        "ultimo_periodo": grupo["periodo"].max(),
    })
    features["recencia_meses"] = meses_hasta(
        features["ultimo_periodo"], periodos[-meses_objetivo - 1]).clip(lower=0)
    features = features.drop(columns=["ultimo_periodo"])

    for columna in categorias_presentes:
        agregado = historico.groupby("person_id")[columna]
        features[f"{columna}_med"] = agregado.mean()
        features[f"{columna}_max"] = agregado.max()

    proporciones = features[[f"{c}_med" for c in categorias_presentes]]
    total = proporciones.sum(axis=1).replace(0, np.nan)
    entropia = -(proporciones.div(total, axis=0)
                 .replace(0, np.nan)
                 .pipe(lambda d: d * np.log(d))).sum(axis=1)
    features["diversificacion"] = (entropia / np.log(max(len(categorias_presentes), 2))).fillna(0.0)

    por_trf = trf[trf["periodo"] < corte].groupby("person_id")
    features["ach_recibido_med"] = por_trf["recibido"].mean()
    features["ach_enviado_med"] = por_trf["enviado"].mean()
    features["ach_meses"] = por_trf["periodo"].nunique()

    perfil_ss = perfil_persona_ss(storage=ctx.storage, estrategia=ctx.estrategia,
                                  settings=ajustes).set_index("person_id")
    features["ss_ibc"] = perfil_ss["ibc_ss"]
    features["ss_meses"] = perfil_ss["n_meses_ss"]
    features["ss_frac_pension"] = perfil_ss["frac_pension"]
    features = features.fillna(0.0)

    gasto_futuro = objetivo.groupby("person_id")[columna_categoria].sum()
    etiqueta = (gasto_futuro.reindex(features.index).fillna(0.0) > 0).astype(int)
    activo_antes = (historico.groupby("person_id")[columna_categoria].sum()
                    .reindex(features.index).fillna(0.0) > 0)
    return features, etiqueta, activo_antes, columna_categoria


def ejecutar(ctx: ContextoModelo) -> ModelResult:
    semilla = ctx.semilla
    categoria = ctx.param("categoria")
    if not categoria:
        raise ValueError(
            f"El modelo {ctx.config.id} no declara 'categoria' en models_config.yml. "
            "Debe ser explícita: el script original la elegía sola y eso hacía que el "
            "mismo model_id entrenara modelos distintos según los datos."
        )

    features, etiqueta, activo_antes, columna_categoria = construir_paneles(ctx, categoria)

    universo = features.index[features["pse_meses"] >= 2]
    X_df = features.loc[universo]
    y = etiqueta.loc[universo]
    activo = activo_antes.loc[universo]

    # --- Corrección de la fuga: fuera todo lo derivado de la categoría objetivo ---
    a_excluir = _columnas_de_la_categoria(list(X_df.columns), categoria)
    X_df = X_df.drop(columns=a_excluir)
    log.info("Variables excluidas por ser de la categoría objetivo: %s", a_excluir)

    if y.nunique() < 2:
        raise ValueError(
            f"En la ventana objetivo la categoría {categoria!r} no tiene ambas clases "
            f"({int(y.sum())} positivos de {len(y)}): no se puede entrenar un clasificador."
        )
    positivos = int(y.sum())
    log.info("Universo: %s personas · tasa positiva %.4f (%s casos)",
             f"{len(X_df):,}", y.mean(), f"{positivos:,}")

    estratificar = y if y.value_counts().min() >= 2 else None
    X_tr, X_te, y_tr, y_te, activo_tr, activo_te = train_test_split(
        X_df, y, activo, test_size=float(ctx.param("test_size", 0.20)),
        random_state=semilla, stratify=estratificar)

    modelo = Pipeline([
        ("imputacion", SimpleImputer(strategy="median")),
        ("clasificador", RandomForestClassifier(
            n_estimators=int(ctx.param("n_estimators", 300)),
            max_depth=int(ctx.param("max_depth", 12)),
            min_samples_leaf=int(ctx.param("min_samples_leaf", 3)),
            class_weight="balanced", random_state=semilla, n_jobs=-1)),
    ]).fit(X_tr, y_tr)

    prediccion = modelo.predict(X_te)
    probabilidad = modelo.predict_proba(X_te)[:, 1]
    clase_base = int(y_tr.mode().iloc[0])
    base = np.repeat(clase_base, len(y_te))

    auc = float(roc_auc_score(y_te, probabilidad)) if y_te.nunique() == 2 else float("nan")

    # La métrica que de verdad importa: ¿encuentra gente nueva, o solo confirma
    # a quien ya gastaba en la categoría?
    nuevos = ~activo_te.to_numpy(dtype=bool)
    auc_nuevos = float("nan")
    if nuevos.sum() > 30 and len(set(y_te[nuevos])) == 2:
        auc_nuevos = float(roc_auc_score(y_te[nuevos], probabilidad[nuevos]))

    fpr, tpr, _ = roc_curve(y_te, probabilidad)
    paso = max(1, len(fpr) // 100)
    matriz = confusion_matrix(y_te, prediccion)

    importancias = modelo.named_steps["clasificador"].feature_importances_
    top = (pd.Series(importancias, index=X_df.columns).sort_values(ascending=False).head(15))

    metricas = limpiar_metricas({
        "roc_auc": auc,
        "roc_auc_nuevos_adoptantes": auc_nuevos,
        "average_precision": float(average_precision_score(y_te, probabilidad)),
        "f1": float(f1_score(y_te, prediccion, zero_division=0)),
        "precision": float(precision_score(y_te, prediccion, zero_division=0)),
        "recall": float(recall_score(y_te, prediccion, zero_division=0)),
        "accuracy": float(accuracy_score(y_te, prediccion)),
        "accuracy_baseline": float(accuracy_score(y_te, base)),
        "tasa_positiva": float(y.mean()),
        "n_entities": len(X_df),
        "n_positivos": positivos,
        "n_variables": X_df.shape[1],
        "pct_positivos_ya_activos": float(
            (y_te.to_numpy() & activo_te.to_numpy(dtype=bool)).sum() / max(int(y_te.sum()), 1) * 100),
    })

    puntuadas = pd.DataFrame({
        "person_id": X_df.index,
        "probabilidad": modelo.predict_proba(X_df)[:, 1],
        "gasto_previo_en_categoria": activo.to_numpy(),
        "gasto_futuro_real": y.to_numpy(),
    })

    artefactos = {
        "model_uri": guardar_artefacto(ctx, {
            "pipeline": modelo, "features": list(X_df.columns), "categoria": categoria,
            "excluidas_por_fuga": a_excluir}, "model.joblib"),
        "assignments_uri": guardar_asignaciones(ctx, puntuadas),
    }

    return construir_resultado(
        model_id=ctx.config.id, model_name=ctx.config.nombre, catalog_ref=ctx.config.catalogo,
        use_case=ctx.config.caso_uso, task_type=ctx.config.task_type, run_id=ctx.run_id,
        started_at=ctx.started_at, dataset=ctx.dataset_info(len(X_df)),
        params=ctx.params_reportados({
            "algoritmo": "RandomForestClassifier",
            "categoria_objetivo": categoria,
            "columna_objetivo": columna_categoria,
            "variables_excluidas_por_fuga": a_excluir,
            "n_variables": X_df.shape[1],
        }),
        metrics=metricas,
        charts=Charts(
            roc_curve=CurvaROC(
                fpr=[round(float(v), 5) for v in fpr[::paso]],
                tpr=[round(float(v), 5) for v in tpr[::paso]],
                auc=round(auc, 6) if not np.isnan(auc) else 0.0),
            confusion_matrix=MatrizConfusion(
                labels=[f"no gasta en {categoria}", f"gasta en {categoria}"],
                matrix=[[int(v) for v in fila] for fila in matriz]),
            feature_importance=[
                ImportanciaVariable(feature=nombre, importance=round(float(valor), 6))
                for nombre, valor in top.items()
            ],
        ),
        artifacts=artefactos,
        notes=[
            f"Las variables derivadas de {categoria!r} se excluyeron del predictor "
            "para no predecir el gasto futuro con el gasto pasado de la misma categoría. "
            "El script original no lo hacía y su AUC estaba inflado.",
            "El AUC en nuevos adoptantes es la métrica de adquisición: si cae al azar, el "
            "modelo sirve para retención pero no para encontrar clientes nuevos.",
        ],
    )


# Sirve a más de una entrada del registro (#17 y #55), así que exige --model-id.
main = cli_para()

if __name__ == "__main__":
    sys.exit(main())
