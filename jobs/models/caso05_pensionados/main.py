"""Modelo #101 — Segmentación de pensionados por consumo. Caso de uso 5, Retail y Banca.

Identifica a los pensionados con la evidencia formal de PILA (no con un proxy
conductual) y los agrupa por *cómo* administran su mesada: intensidad del gasto,
ritmo, composición y presión sobre el ingreso.

Migrado del notebook ``Caso05_Modelo101_Segmentacion_Pensionados_Clustering.ipynb``.
Dos decisiones de método que se conservan tal cual, porque son el resultado de
descartar alternativas peores:

* La selección de escalador y de k exige que **ningún segmento baje del 5% ni supere
  el 50%** de la base. Sin esa restricción, RobustScaler alcanza una silueta de 0,62
  aislando outliers y dejando al 79% de la base en un solo grupo: métrica alta,
  segmentación inservible.
* Las variables de composición muy dispersas (salud, retail, ocio por separado tienen
  entre 66% y 93% de ceros) no entran al clustering; entran sumadas.
"""

from __future__ import annotations

import logging
import sys

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import calinski_harabasz_score, davies_bouldin_score, silhouette_score
from sklearn.preprocessing import RobustScaler, StandardScaler

from common.features import (
    GRUPO_OBJETIVO,
    cargar_fuente,
    categorizar_comercio,
    decodificar_entidad_objetivo,
    meses_hasta,
    num,
)
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

FEATURES = [
    "gasto_mensual_medio", "ticket_medio", "n_comercios",   # intensidad
    "regularidad", "diversificacion",                        # hábito
    "share_deuda", "share_esencial",                         # composición
    "tasa_consumo",                                          # presión sobre la mesada
]
FEATURES_LOG = ["gasto_mensual_medio", "ticket_medio", "n_comercios"]

BLOQUES_GASTO = {
    "deuda": ["Financiero / créditos"],
    "servicios": ["Telco / servicios públicos"],
    "salud": ["Salud", "Seguros"],
    "retail": ["Comercio electrónico / retail"],
    "ocio": ["Viajes / transporte", "Streaming / digital", "Apuestas y juegos"],
    "obligaciones": ["Gobierno / impuestos", "Educación", "Seguridad social / nómina"],
    "intermediado": ["Pasarelas / agregadores", "Otros / no clasificado"],
}
MAPA_BLOQUE = {cat: bloque for bloque, cats in BLOQUES_GASTO.items() for cat in cats}

CATALOGO_PAGADOR = {
    "Colpensiones": ["ADMINISTRADORA COLOMBIANA DE PENSIONES"],
    "Porvenir": ["PORV"],
    "Protección": ["FONDO DE PENSIONES OBLIGATORIAS PROTECCION", "PROTECCION"],
    "Colfondos": ["FONDO DE PENSIONES OBLIGATORIAS COLFONDOS", "COLFONDOS"],
    "Skandia": ["SKANDIA"],
    "Consorcio FOPEP": ["CONSORCIO FO"],
    "Aseguradora (renta vitalicia)": ["SEGUROS", "COMPANIA SEGUROS", "POSITIVA", "ASULADO",
                                      "MAPFRE", "AXA COLPATRIA", "LA EQUIDAD", "COLMENA",
                                      "BBVA SEGUROS"],
    "Fondo público / especial": ["FONDO DE PASIVO SOCIAL", "FONDO DE PREVISION", "FIDUPREV",
                                 "FIDECOMISOS PATRIMONIOS AUTONOMOS", "FIDUCIARIA"],
}


def _clasificar_pagador(nombre) -> str:
    texto = str(nombre).upper()
    for etiqueta, claves in CATALOGO_PAGADOR.items():
        if any(texto.startswith(clave) or clave in texto for clave in claves):
            return etiqueta
    return "Empleador / otro pagador"


def _clasificar_regimen(relacion) -> str:
    texto = str(relacion).upper()
    if "PRIMA MEDIA" in texto:
        return "Prima media (RPM)"
    if "AHORRO INDIVIDUAL" in texto:
        return "Ahorro individual (RAIS)"
    if "RIESGOS PROFESIONALES" in texto:
        return "Riesgos profesionales (ARL)"
    if "POR EL EMPLEADOR" in texto:
        return "Pensión a cargo del empleador"
    return "No determinado"


def _dominante(df: pd.DataFrame, columna: str) -> pd.Series:
    conteo = df.groupby(["person_id", columna]).size().reset_index(name="n")
    return (conteo.sort_values(["n", columna]).drop_duplicates("person_id", keep="last")
            .set_index("person_id")[columna])


def identificar_pensionados(ctx: ContextoModelo) -> tuple[pd.DataFrame, dict]:
    """Aísla a los pensionados con los tres marcadores de PILA y perfila su mesada."""
    ss = cargar_fuente("ss", storage=ctx.storage, estrategia=ctx.estrategia,
                       solo_ventana=False, settings=ctx.settings)

    relacion = ss["Relación laboral"].astype("string").str.upper()
    planilla = ss["Tipo planilla"].astype("string").str.upper()
    aportante = ss["Tipo aportante"].astype("string").str.upper()

    def bool_puro(serie) -> np.ndarray:
        return serie.fillna(False).to_numpy(dtype=bool)

    marcadores = {
        "relacion_laboral": bool_puro(relacion.str.contains("PENSIONADO", na=False)),
        "tipo_planilla_P": bool_puro(planilla.eq("P")),
        "tipo_aportante": bool_puro(aportante.str.contains("PENSIONES", na=False)),
    }
    es_pension = marcadores["relacion_laboral"] | marcadores["tipo_planilla_P"] | marcadores["tipo_aportante"]

    filas = ss.loc[es_pension].assign(
        _pagador=ss.loc[es_pension, "Razón social"].map(_clasificar_pagador),
        _regimen=relacion[es_pension].map(_clasificar_regimen),
        _mesada=num(ss.loc[es_pension, "Ingreso base salud"]),
    )
    perfil = pd.DataFrame({
        "pagador_mesada": _dominante(filas, "_pagador"),
        "regimen_pension": _dominante(filas, "_regimen"),
        "meses_como_pensionado": filas.groupby("person_id")["periodo"].nunique(),
    })
    mesada = (filas[filas["_mesada"] > 0].groupby("person_id")["_mesada"].median()
              .rename("mesada_declarada"))
    perfil = perfil.join(mesada).fillna({"mesada_declarada": 0.0})

    # Si la planilla de pensionados no reporta IBC, se usa la mediana del IBC de la
    # misma persona en el resto de sus registros PILA. No se imputa con datos ajenos.
    ibc_global = (ss.assign(_i=num(ss["Ingreso base salud"])).query("_i > 0")
                  .groupby("person_id")["_i"].median())
    sin_mesada = perfil["mesada_declarada"] <= 0
    perfil.loc[sin_mesada, "mesada_declarada"] = perfil.index[sin_mesada].map(ibc_global).to_numpy()
    perfil["mesada_declarada"] = perfil["mesada_declarada"].fillna(0.0)

    diagnostico = {
        "identificados_pila": int(perfil.shape[0]),
        "marcador_relacion_laboral": int(marcadores["relacion_laboral"].sum()),
        "marcador_tipo_planilla": int(marcadores["tipo_planilla_P"].sum()),
        "marcador_tipo_aportante": int(marcadores["tipo_aportante"].sum()),
        "sin_mesada_en_planilla": int(sin_mesada.sum()),
        "mesada_recuperada": int((perfil.loc[sin_mesada, "mesada_declarada"] > 0).sum()),
    }
    log.info("Pensionados identificados en PILA: %s", f"{len(perfil):,}")
    return perfil, diagnostico


def construir_features(ctx: ContextoModelo, perfil_pension: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Tabla una fila = un pensionado con la señal de consumo de la ventana."""
    meses = float(ctx.settings.meses_ventana())
    min_meses = int(ctx.param("min_meses_pse", 2))

    pse = cargar_fuente("pse", storage=ctx.storage, estrategia=ctx.estrategia,
                        solo_ventana=True, settings=ctx.settings)
    pse = pse.assign(
        valor=num(pse["Valor"]), cantidad=num(pse["Cantidad"]),
        categoria=categorizar_comercio(pse["Comercio"]))
    objetivo = decodificar_entidad_objetivo(pse["Comercio"])
    pse["entidad_objetivo"] = objetivo["entidad_objetivo"]

    pensionados = set(perfil_pension.index)
    meses_pse = pse[pse["person_id"].isin(pensionados)].groupby("person_id")["periodo"].nunique()
    con_consumo = set(meses_pse[meses_pse >= min_meses].index)
    con_mesada = set(perfil_pension.index[perfil_pension["mesada_declarada"] > 0])
    universo = con_consumo & con_mesada
    if not universo:
        raise ValueError(
            "Ningún pensionado cumple las dos condiciones del universo "
            f"(>= {min_meses} meses de consumo PSE y mesada observable)."
        )
    log.info("Universo de modelado: %s pensionados", f"{len(universo):,}")

    datos = pse[pse["person_id"].isin(universo)].copy()
    datos["bloque"] = datos["categoria"].map(MAPA_BLOQUE).fillna("intermediado")

    panel = datos.groupby(["person_id", "periodo"], as_index=False).agg(
        gasto=("valor", "sum"), pagos=("cantidad", "sum"))
    por_mes = panel.groupby("person_id")
    por_persona = datos.groupby("person_id")

    F = pd.DataFrame({
        "gasto_total": por_persona["valor"].sum(),
        "n_pagos": por_persona["cantidad"].sum(),
        "n_comercios": por_persona["Comercio"].nunique(),
        "meses_activos": por_mes["periodo"].nunique(),
        "gasto_mensual_medio": por_mes["gasto"].mean(),
        "ultimo_periodo": por_mes["periodo"].max(),
    })
    F["recencia_meses"] = meses_hasta(F["ultimo_periodo"], ctx.settings.window_end).clip(lower=0)
    F["regularidad"] = F["meses_activos"] / meses
    F["ticket_medio"] = F["gasto_total"] / F["n_pagos"].replace(0, np.nan)
    F["volatilidad_gasto"] = (por_mes["gasto"].std() / por_mes["gasto"].mean()).fillna(0.0)

    share_mes = panel["gasto"] / panel.groupby("person_id")["gasto"].transform("sum").replace(0, np.nan)
    F["hhi_mensual"] = share_mes.pow(2).groupby(panel["person_id"]).sum().fillna(1.0)

    pivote = (datos.pivot_table(index="person_id", columns="bloque", values="valor",
                                aggfunc="sum", fill_value=0.0)
              .reindex(columns=list(BLOQUES_GASTO), fill_value=0.0))
    shares = pivote.div(pivote.sum(axis=1).replace(0, np.nan), axis=0).fillna(0.0)
    F = F.join(shares.add_prefix("share_"))
    F["diversificacion"] = (-(shares * np.log(shares.replace(0, np.nan))).sum(axis=1)
                            / np.log(shares.shape[1])).fillna(0.0)
    F["share_esencial"] = F[["share_servicios", "share_salud", "share_obligaciones"]].sum(axis=1)
    F["share_discrecional"] = F[["share_retail", "share_ocio"]].sum(axis=1)

    F = F.join(perfil_pension[["mesada_declarada", "pagador_mesada", "regimen_pension",
                               "meses_como_pensionado"]])
    F["tasa_consumo"] = (F["gasto_mensual_medio"] / F["mesada_declarada"]).clip(upper=5.0)
    F["en_riesgo_desconexion"] = F["recencia_meses"] >= 3

    # Relación con las entidades objetivo (banca, retail y cajas de compensación).
    objetivos = datos[datos["entidad_objetivo"] != "No objetivo"]
    F["gasto_entidades_objetivo"] = objetivos.groupby("person_id")["valor"].sum().reindex(F.index).fillna(0.0)
    F["share_entidades_objetivo"] = (F["gasto_entidades_objetivo"]
                                     / F["gasto_total"].replace(0, np.nan)).fillna(0.0)
    for grupo, etiqueta in (("Caja de compensación", "usa_caja_compensacion"),
                            ("Retail", "usa_retail_objetivo"), ("Banca", "usa_banca_objetivo")):
        ids = set(objetivos.loc[objetivos["entidad_objetivo"].map(GRUPO_OBJETIVO) == grupo, "person_id"])
        F[etiqueta] = F.index.isin(ids)

    penetracion = {
        f"pct_{etiqueta}": round(float(F[etiqueta].mean() * 100), 2)
        for etiqueta in ("usa_caja_compensacion", "usa_retail_objetivo", "usa_banca_objetivo")
    }
    diagnostico = {
        "universo": len(universo),
        "excluidos_sin_mesada": len(con_consumo - con_mesada),
        "excluidos_sin_consumo": len(pensionados & set(pse["person_id"])) - len(con_consumo),
        **penetracion,
    }
    return F.reset_index(), diagnostico


def _matriz(F: pd.DataFrame) -> tuple[pd.DataFrame, dict[str, tuple[float, float]]]:
    """Winsoriza al 1-99 y aplica log a montos y conteos."""
    X = F[FEATURES].copy().astype(float)
    limites = {c: (float(X[c].quantile(0.01)), float(X[c].quantile(0.99))) for c in FEATURES}
    for columna, (bajo, alto) in limites.items():
        X[columna] = X[columna].clip(bajo, alto)
    for columna in FEATURES_LOG:
        X[columna] = np.log1p(X[columna])
    return X, limites


def nombrar_segmento(fila: pd.Series) -> str:
    if fila["tasa_consumo"] >= 1.5:
        return "Pensionado de alto consumo (supera su mesada)"
    if fila["share_esencial"] >= 0.35:
        return "Pensionado de gasto esencial (servicios y salud)"
    if fila["regularidad"] >= 0.75 and fila["comercios"] >= 8:
        return "Pensionado digital de rutina"
    return "Pensionado de uso esporádico"


def ejecutar(ctx: ContextoModelo) -> ModelResult:
    semilla = ctx.semilla
    min_seg = float(ctx.param("min_segmento", 0.05))
    max_seg = float(ctx.param("max_segmento", 0.50))

    perfil_pension, diag_poblacion = identificar_pensionados(ctx)
    F, diag_universo = construir_features(ctx, perfil_pension)
    X, limites = _matriz(F)

    # Escalador y k se eligen juntos, con el criterio de accionabilidad explícito.
    k_min, k_max = int(ctx.param("k_min", 3)), int(ctx.param("k_max", 7))
    candidatos = []
    for nombre_escalador, escalador in (("StandardScaler", StandardScaler()),
                                        ("RobustScaler", RobustScaler())):
        Z = escalador.fit_transform(X)
        for k in range(k_min, k_max + 1):
            etiquetas = KMeans(n_clusters=k, n_init=30, random_state=semilla).fit_predict(Z)
            tamanos = pd.Series(etiquetas).value_counts(normalize=True)
            candidatos.append({
                "escalador": nombre_escalador, "k": k,
                "silueta": float(silhouette_score(Z, etiquetas)),
                "davies_bouldin": float(davies_bouldin_score(Z, etiquetas)),
                "menor": float(tamanos.min()), "mayor": float(tamanos.max()),
                "equilibrada": bool(tamanos.min() >= min_seg and tamanos.max() <= max_seg),
            })
    tabla = pd.DataFrame(candidatos)
    viables = tabla[tabla["equilibrada"]]
    if viables.empty:
        raise ValueError(
            f"Ninguna configuración deja todos los segmentos entre {min_seg:.0%} y {max_seg:.0%} "
            "de la base. Revisa el universo o afloja el criterio en models_config.yml."
        )
    ganadora = viables.loc[viables["silueta"].idxmax()]
    escalador = StandardScaler() if ganadora["escalador"] == "StandardScaler" else RobustScaler()
    K = int(ganadora["k"])
    Xs = escalador.fit_transform(X)
    log.info("Configuración elegida: %s · k=%d · silueta %.3f",
             ganadora["escalador"], K, ganadora["silueta"])

    seleccion = [
        PuntoSeleccionK(k=int(fila["k"]), silhouette=round(fila["silueta"], 6),
                        davies_bouldin=round(fila["davies_bouldin"], 6))
        for _, fila in tabla[tabla["escalador"] == ganadora["escalador"]].iterrows()
    ]

    modelo = KMeans(n_clusters=K, n_init=50, random_state=semilla).fit(Xs)
    F["cluster"] = modelo.labels_

    perfil = F.groupby("cluster").agg(
        personas=("person_id", "size"),
        gasto_mensual=("gasto_mensual_medio", "median"), mesada=("mesada_declarada", "median"),
        tasa_consumo=("tasa_consumo", "median"), n_pagos=("n_pagos", "median"),
        ticket=("ticket_medio", "median"), comercios=("n_comercios", "median"),
        regularidad=("regularidad", "median"), recencia=("recencia_meses", "median"),
        diversificacion=("diversificacion", "median"), share_deuda=("share_deuda", "median"),
        share_esencial=("share_esencial", "median"),
        share_intermediado=("share_intermediado", "median"),
        desconexion=("en_riesgo_desconexion", "mean"))

    nombres = nombres_unicos(perfil, nombrar_segmento, "gasto_mensual")
    F["segmento"] = F["cluster"].map(nombres)

    proyeccion = PCA(n_components=2, random_state=semilla).fit(Xs)
    plano = proyeccion.transform(Xs)

    metricas = limpiar_metricas({
        "silhouette": ganadora["silueta"],
        "davies_bouldin": ganadora["davies_bouldin"],
        "calinski_harabasz": calinski_harabasz_score(Xs, modelo.labels_),
        "k": K,
        "n_entities": len(F),
        "pensionados_identificados": diag_poblacion["identificados_pila"],
        "mesada_mediana": float(F["mesada_declarada"].median()),
        "tasa_consumo_mediana": float(F["tasa_consumo"].median()),
        "pct_en_riesgo_desconexion": float(F["en_riesgo_desconexion"].mean() * 100),
        **{k: v for k, v in diag_universo.items() if k.startswith("pct_")},
    })
    segmentos = construir_segmentos(F["cluster"], nombres, perfil)

    columnas_salida = ["person_id", "cluster", "segmento", "pagador_mesada", "regimen_pension",
                       "mesada_declarada", "gasto_total", "gasto_mensual_medio", "tasa_consumo",
                       "n_pagos", "n_comercios", "ticket_medio", "meses_activos", "regularidad",
                       "recencia_meses", "volatilidad_gasto", "hhi_mensual", "diversificacion",
                       "share_deuda", "share_esencial", "share_intermediado",
                       "share_entidades_objetivo", "usa_caja_compensacion", "usa_retail_objetivo",
                       "usa_banca_objetivo", "en_riesgo_desconexion"]
    artefactos = {
        "model_uri": guardar_artefacto(ctx, {
            "escalador": escalador, "kmeans": modelo, "features": FEATURES,
            "log_features": FEATURES_LOG, "limites_winsor": limites, "segmentos": nombres,
        }, "model.joblib"),
        "assignments_uri": guardar_asignaciones(ctx, F[[c for c in columnas_salida if c in F.columns]]),
    }

    return construir_resultado(
        model_id=ctx.config.id, model_name=ctx.config.nombre, catalog_ref=ctx.config.catalogo,
        use_case=ctx.config.caso_uso, task_type=ctx.config.task_type, run_id=ctx.run_id,
        started_at=ctx.started_at, dataset=ctx.dataset_info(len(F)),
        params=ctx.params_reportados({
            "k": K, "algoritmo": "KMeans", "escalador": ganadora["escalador"],
            "features": FEATURES, "criterio_seleccion": (
                f"mejor silueta con todos los segmentos entre {min_seg:.0%} y {max_seg:.0%}"),
            **diag_poblacion,
        }),
        metrics=metricas, segments=segmentos,
        charts=Charts(
            segment_distribution=segmentos,
            k_selection=seleccion,
            scatter_2d=Scatter2D(
                points=muestra_scatter(plano, F["cluster"], nombres, semilla=semilla),
                explained_variance=round(float(proyeccion.explained_variance_ratio_.sum()), 4)),
        ),
        artifacts=artefactos,
        notes=[
            "La condición de pensionado sale de PILA; el consumo, de PSE. La vía puramente "
            "transaccional no funciona: los originadores de las mesadas no son visibles en ACH.",
            "Los datos son mensuales: el efecto 'día de pago de la mesada' no es medible y se "
            "sustituye por regularidad y concentración mensual.",
            "Universo de cientos de personas: los segmentos son direccionales, no censales.",
        ],
    )


main = cli_para("caso05_pensionados")

if __name__ == "__main__":
    sys.exit(main())
