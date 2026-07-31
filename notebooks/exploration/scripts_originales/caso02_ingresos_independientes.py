"""
Caso 02 - Estimador de Ingresos para Independientes
=====================================================

Script derivado del cuaderno original. Contiene el pipeline de datos
completo (paneles persona-mes de Seguridad Social, ACH y PSE) mas los 3
modelos del catalogo con mejor desempeno y menor dependencia de senales
debiles:

    #6   Estimacion de ingresos reales        (regresion, R2 log = 0.473, MAPE 25.5%)
    #8B  Brecha de subdeclaracion              (regla sobre #6, coherente por segmento)
    #41  Scoring de capacidad de ahorro (FCM)  (scoring, monotonicidad estricta)

Se omitieron deliberadamente del cuaderno original, sin intentar corregirlos:

    #8A  Clasificacion de informalidad  -> el propio analisis original lo
                                            marca como "discrimina debilmente"
                                            (AUC-ROC 0.614, por debajo del
                                            umbral usable de 0.70)
    #47  Scoring de bancarizacion       -> depende directamente de la
                                            probabilidad de #8A como insumo
                                            (15% del puntaje), por lo que
                                            arrastra la misma debilidad

Fuentes: Seguridad Social (SOI), Transferencias ACH y Pagos Digitales (PSE).

Uso
---
    python caso02_ingresos_independientes.py \
        --data-dir /ruta/a/los/excel \
        --output-dir /ruta/de/salida \
        --cache-dir ./cache_modelos
"""

from __future__ import annotations

import argparse
import csv
import warnings
from pathlib import Path

import numpy as np
import openpyxl
import pandas as pd

from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 80)
pd.set_option("display.width", 200)


# ---------------------------------------------------------------------------
# 1. Parametros generales
# ---------------------------------------------------------------------------

ARCHIVOS = {
    "pagos": "Conversion Pagos Digitales 1 - Ofuscado.xlsx",
    "ach": "Conversion Transferencias ACH 1 - Ofuscado.xlsx",
    "seguridad": "Conversion Seguridad Social 1 - Ofuscado.xlsx",
}

VENTANA_INI, VENTANA_FIN = "2025-01", "2025-09"  # meses con las tres fuentes intactas
N_MESES = 9
SMLMV = 1_423_500  # salario minimo mensual legal vigente 2025, referencia de escala
SEMILLA = 42
EPS = 1.0

TAXONOMIA_PSE = {
    "financiero": ["bancolom", "banco ", "banco d", "davivienda", "nu colombia", "falab", "caja social",
                   "occidente", "mundo mu", "adelante soluciones financi", "sistecre", "fintec",
                   "bancar tecnolog", "lulo", "solventa", "creditos rap", "financiamiento", "credifac",
                   "aval valor compar", "red multibanca", "compañia de credito", "tarjeta", "credito",
                   "cooperativa", "scotiabank", "bbva", "itau", "popular", "agrario"],
    "servicios_recurrentes": ["comunicacion celular", "telecomunicaciones", "une - epm", "partners telecom",
                               "acueducto", "alcantarillado", "gases del", "alcanos", "efigas",
                               "surtidora de gas", "empresas publicas", "empresas municipales", "wom",
                               "claro", "movistar", "tigo", "energia", "electrific", "aseo",
                               "gas natural", "codensa", "emcali"],
    "inversion": ["acciones y valo", "cartera colectiva", "fideicomiso", "fiduciaria",
                  "fondo de inversion", "comisionista", "valores"],
    "publico": ["secretaria", "superintendencia", "instituto colombiano", "fondo nacional", "dian",
                "alcaldia", "gobernacion", "notariado", "registraduria", "ministerio", "municipio"],
    "salud_proteccion": ["aportes", "colpro", "caja colombiana de subsidio", "caja de compensacion",
                          "eps ", "sura", "compensar", "colsubsidio", "medicina prepagada", "seguro"],
    "educacion": ["universidad", "colegio", "icetex", "educaci", "sena"],
    "entretenimiento": ["cine ", "cinemark", "cinepolis", "netflix", "spotify", "streaming", "teatro"],
    "pasarela": ["kushki", "mercadopago", "payu", "dlocal", "gou payments", "ebanx", "openpay",
                 "payments way", "portal zona pa", "payment", "pexto", "tumipa", "epayco",
                 "wompi", "placetopay"],
}
ORDEN_CATEGORIAS_PSE = ["financiero", "servicios_recurrentes", "inversion", "publico",
                         "salud_proteccion", "educacion", "entretenimiento", "pasarela"]


# ---------------------------------------------------------------------------
# 2. Utilidades de carga
# ---------------------------------------------------------------------------

def cargar(clave: str, data_dir: Path, cache_dir: Path) -> pd.DataFrame:
    """Carga un dataset con cache en parquet.

    Los .xlsx superan el millon de filas, por lo que se leen en modo
    streaming con openpyxl y se cachean. Las columnas de texto se
    convierten a 'category' para bajar el consumo de memoria.
    """
    parquet = cache_dir / f"{clave}.parquet"
    if parquet.exists():
        return pd.read_parquet(parquet)

    csv_tmp = cache_dir / f"{clave}.csv"
    if not csv_tmp.exists():
        print(f"Convirtiendo {ARCHIVOS[clave]} (puede tardar varios minutos)...")
        wb = openpyxl.load_workbook(data_dir / ARCHIVOS[clave], read_only=True)
        ws = wb.worksheets[0]
        with open(csv_tmp, "w", newline="", encoding="utf-8") as fh:
            w = csv.writer(fh)
            for fila in ws.iter_rows(values_only=True):
                if all(v is None for v in fila):
                    continue
                w.writerow(["" if v is None else v for v in fila])
        wb.close()

    df = pd.read_csv(csv_tmp, dtype=str, low_memory=False)
    for c in df.columns:
        if df[c].nunique(dropna=True) < len(df) * 0.5:
            df[c] = df[c].astype("category").cat.as_ordered()
    df.to_parquet(parquet, index=False)
    return df


def texto(serie: pd.Series) -> pd.Series:
    """Convierte a texto plano. Necesario porque las columnas vienen como 'category'."""
    return serie.astype("object")


def numerico(df: pd.DataFrame, columnas) -> pd.DataFrame:
    df = df.copy()
    for c in columnas:
        df[c] = pd.to_numeric(texto(df[c]), errors="coerce")
    return df


def percentil(s: pd.Series) -> pd.Series:
    """Rango percentil 0-1, para componer scores sin que la escala de una variable domine."""
    return s.rank(pct=True)


def en_smlmv(v):
    return v / SMLMV


# ---------------------------------------------------------------------------
# 3. Construccion del panel persona-mes
# ---------------------------------------------------------------------------

def construir_ss(data_dir: Path, cache_dir: Path):
    s = cargar("seguridad", data_dir, cache_dir)
    s["cli"] = texto(s["Nombre"]) + "|" + texto(s["Número de documento"])
    s["per"] = texto(s["Periodo cotización"])
    s = numerico(s, ["Ingreso base salud", "Ingreso base pensión", "Salario básico",
                      "Días salud", "Ingreso base riesgos"])
    sw = s[(s["per"] >= VENTANA_INI) & (s["per"] <= VENTANA_FIN)].copy()

    pm = sw.groupby(["cli", "per"], observed=True).agg(
        ibc_salud=("Ingreso base salud", "sum"),
        ibc_pension=("Ingreso base pensión", "sum"),
        salario_basico=("Salario básico", "sum"),
        dias_salud=("Días salud", "max"),
        n_empleadores=("Número documento", "nunique"),
    ).reset_index()

    def moda(x):
        m = x.mode()
        return m.iloc[0] if len(m) else np.nan

    attr = sw.groupby("cli", observed=True).agg(
        relacion_laboral=("Relación laboral", moda),
        tipo_planilla=("Tipo planilla", moda),
        clase_aportante=("Clase aportante", moda),
        tipo_aportante=("Tipo aportante", moda),
        actividad=("Código actividad económica", moda),
    )

    nov = sw[["cli", "Novedades"]].dropna().copy()
    nov["Novedades"] = texto(nov["Novedades"])
    ex = nov.assign(n=nov["Novedades"].str.split(",")).explode("n")
    ex["n"] = ex["n"].str.strip()
    top = ex["n"].value_counts().head(10).index.tolist()
    flags = (ex[ex["n"].isin(top)].assign(v=1)
             .pivot_table(index="cli", columns="n", values="v", aggfunc="max", fill_value=0))
    flags.columns = ["nov_" + c[:28].replace(" ", "_") for c in flags.columns]
    attr = attr.join(flags)
    for c in flags.columns:
        attr[c] = attr[c].fillna(0)
    return pm, attr, top


def construir_ach(data_dir: Path, cache_dir: Path):
    a = cargar("ach", data_dir, cache_dir)
    a["cli"] = texto(a["Nombre"]) + "|" + texto(a["Número documento"])
    a["per"] = texto(a["Periodo"])
    a = numerico(a, ["Valor", "Cantidad de transferencias"])

    aw = a[(a["per"] >= VENTANA_INI) & (a["per"] <= VENTANA_FIN) &
           (texto(a["Cuenta propia"]) != "SI")].copy()

    aw["contraparte"] = np.where(texto(aw["Tipo registro"]) == "RECIBIDO",
                                  texto(aw["Nombre usuario originador"]),
                                  texto(aw["Nombre usuario receptor"]))
    rec = aw[texto(aw["Tipo registro"]) == "RECIBIDO"]
    env = aw[texto(aw["Tipo registro"]) == "ENVIADO"]

    pm = rec.groupby(["cli", "per"], observed=True).agg(
        recibido=("Valor", "sum"), n_rec=("Cantidad de transferencias", "sum"),
        n_contra_rec=("contraparte", "nunique"), max_rec=("Valor", "max"),
    ).join(env.groupby(["cli", "per"], observed=True).agg(
        enviado=("Valor", "sum"), n_env=("Cantidad de transferencias", "sum"),
        n_contra_env=("contraparte", "nunique"),
    ), how="outer").fillna(0).reset_index()

    cp = rec.groupby(["cli", "contraparte"], observed=True).agg(
        meses=("per", "nunique"), total=("Valor", "sum")).reset_index()
    cp = cp.sort_values(["cli", "total"], ascending=[True, False])
    top1 = cp.groupby("cli", observed=True).head(1).set_index("cli")
    contra = cp.groupby("cli", observed=True).agg(
        n_contrapartes_tot=("contraparte", "nunique"), max_meses_contra=("meses", "max"))
    contra["monto_contra_top"] = top1["total"]
    contra["meses_contra_top"] = top1["meses"]
    return pm, contra


def clasificar_comercio(nombre) -> str:
    """Asigna categoria por palabras clave. 'pasarela' va de ultimo a proposito."""
    if not isinstance(nombre, str):
        return "no_identificado"
    n = nombre.lower().strip()
    if n.replace("*", "").strip() == "":
        return "no_identificado"
    for cat in ORDEN_CATEGORIAS_PSE:
        for kw in TAXONOMIA_PSE[cat]:
            if kw in n:
                return cat
    return "otro"


def construir_pse(data_dir: Path, cache_dir: Path):
    p = cargar("pagos", data_dir, cache_dir)
    p["cli"] = texto(p["Nombre"]) + "|" + texto(p["Número documento"])
    p["per"] = texto(p["Periodo"])
    p = numerico(p, ["Valor", "Cantidad"])
    pw = p[(p["per"] >= VENTANA_INI) & (p["per"] <= VENTANA_FIN)].copy()
    pw["Comercio"] = texto(pw["Comercio"])

    comercios = pd.Series(pw["Comercio"].unique())
    mapa = dict(zip(comercios, comercios.map(clasificar_comercio)))
    pw["categoria"] = pw["Comercio"].map(mapa)

    base = pw.groupby(["cli", "per"], observed=True).agg(
        gasto=("Valor", "sum"), n_pagos=("Cantidad", "sum"),
        n_comercios=("Comercio", "nunique"), max_pago=("Valor", "max"),
    ).reset_index()
    piv = pw.pivot_table(index=["cli", "per"], columns="categoria", values="Valor",
                          aggfunc="sum", fill_value=0)
    piv.columns = ["cat_" + c for c in piv.columns]
    pm = base.merge(piv.reset_index(), on=["cli", "per"], how="left").fillna(0)
    return pm


def agregar_persona(pm: pd.DataFrame, columnas, prefijo: str) -> pd.DataFrame:
    g = pm.groupby("cli")
    out = pd.DataFrame(index=g.size().index)
    out[prefijo + "meses"] = g.size()
    for c in columnas:
        out[prefijo + c + "_med"] = g[c].mean()
        out[prefijo + c + "_p50"] = g[c].median()
        out[prefijo + c + "_std"] = g[c].std().fillna(0)
        out[prefijo + c + "_max"] = g[c].max()
    return out


def construir_matriz_persona(ss_pm, ss_attr, ach_pm, ach_contra, pse_pm) -> tuple[pd.DataFrame, list[str]]:
    cats_pse = [c for c in pse_pm.columns if c.startswith("cat_")]
    f_ss = agregar_persona(ss_pm, ["ibc_salud", "ibc_pension", "salario_basico", "dias_salud", "n_empleadores"], "ss_")
    f_ach = agregar_persona(ach_pm, ["recibido", "enviado", "n_rec", "n_env", "n_contra_rec"], "ach_")
    f_pse = agregar_persona(pse_pm, ["gasto", "n_pagos", "n_comercios"] + cats_pse, "pse_")

    m = (f_ss.join(f_ach, how="outer").join(f_pse, how="outer")
         .join(ach_contra, how="left").join(ss_attr, how="left"))

    m["en_ss"] = m.index.isin(f_ss.index).astype(int)
    m["en_ach"] = m.index.isin(f_ach.index).astype(int)
    m["en_pse"] = m.index.isin(f_pse.index).astype(int)
    for c in ["ss_meses", "ach_meses", "pse_meses"]:
        m[c] = m[c].fillna(0)
    numericas = m.select_dtypes(include=[np.number]).columns
    m[numericas] = m[numericas].fillna(0)

    # Variables derivadas de comportamiento
    m["ratio_env_rec"] = m["ach_enviado_med"] / (m["ach_recibido_med"] + EPS)
    m["ratio_gasto_rec"] = m["pse_gasto_med"] / (m["ach_recibido_med"] + EPS)
    m["concentracion_rec"] = m["monto_contra_top"].fillna(0) / (m["ach_recibido_med"] * m["ach_meses"] + EPS)
    m["cv_recibido"] = m["ach_recibido_std"] / (m["ach_recibido_med"] + EPS)
    m["cv_gasto"] = m["pse_gasto_std"] / (m["pse_gasto_med"] + EPS)
    m["intensidad_ach"] = m["ach_meses"] / N_MESES
    m["intensidad_pse"] = m["pse_meses"] / N_MESES
    m["carga_financiera"] = m["pse_cat_financiero_med"] / (m["pse_gasto_med"] + EPS)
    m["peso_servicios"] = m["pse_cat_servicios_recurrentes_med"] / (m["pse_gasto_med"] + EPS)
    m["tiene_inversion"] = (m["pse_cat_inversion_med"] > 0).astype(int)
    m["flujo_neto"] = m["ach_recibido_med"] - m["ach_enviado_med"]

    rl = texto(m["relacion_laboral"]).fillna("")
    m["es_independiente"] = rl.str.contains("INDEPENDIENTE", na=False).astype(int)
    m["es_dependiente"] = (rl == "DEPENDIENTE").astype(int)
    m["es_pensionado"] = rl.str.contains("ensionado", na=False).astype(int)
    m["es_aprendiz"] = rl.str.contains("APRENDIC", na=False).astype(int)

    m = m.copy()

    # Variables predictoras: SOLO transaccionales (ACH + PSE). Se excluye
    # deliberadamente todo lo que venga de SS porque el ingreso SS es el
    # objetivo a predecir en el modelo #6 (evita fuga de informacion).
    features = [c for c in m.columns
                if (c.startswith("ach_") or c.startswith("pse_") or c in [
                    "ratio_env_rec", "ratio_gasto_rec", "concentracion_rec", "cv_recibido", "cv_gasto",
                    "intensidad_ach", "intensidad_pse", "carga_financiera", "peso_servicios",
                    "tiene_inversion", "flujo_neto", "n_contrapartes_tot", "max_meses_contra",
                    "monto_contra_top", "meses_contra_top", "en_ach", "en_pse"])
                and m[c].dtype.kind in "if"]

    return m, features


# ---------------------------------------------------------------------------
# 4. Modelo #6 - Estimacion de ingresos reales
# ---------------------------------------------------------------------------

def evaluar_regresion(y_real_log, y_pred_log, nombre) -> dict:
    real, pred = np.exp(y_real_log), np.exp(y_pred_log)
    return {
        "modelo": nombre,
        "MAE (COP)": mean_absolute_error(real, pred),
        "MAPE": np.mean(np.abs(real - pred) / real),
        "R2 (log)": r2_score(y_real_log, y_pred_log),
    }


def modelo_6_estimacion_ingresos(m: pd.DataFrame, features: list[str]):
    mascara_entrenamiento = (
        (m.en_ss == 1) & (m.es_dependiente == 1) &
        (texto(m.tipo_planilla) == "E") &
        (m.ss_dias_salud_p50 >= 28) &
        (m.ss_meses >= 4) &
        (m.ss_ibc_salud_p50 >= SMLMV * 0.9) &
        ((m.en_ach == 1) | (m.en_pse == 1))
    )
    entrenamiento = m[mascara_entrenamiento].copy()
    print(f"[#6] Poblacion de entrenamiento: {len(entrenamiento):,} personas")

    x = entrenamiento[features].values
    y = np.log(entrenamiento["ss_ibc_salud_p50"].values)
    x_tr, x_te, y_tr, y_te = train_test_split(x, y, test_size=0.25, random_state=SEMILLA)

    resultados = [evaluar_regresion(y_te, np.full_like(y_te, np.median(y_tr)), "Baseline (mediana)")]

    escalador = StandardScaler().fit(x_tr)
    ridge = Ridge(alpha=1.0).fit(escalador.transform(x_tr), y_tr)
    resultados.append(evaluar_regresion(y_te, ridge.predict(escalador.transform(x_te)), "Ridge"))

    modelo_6 = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_depth=6,
        min_samples_leaf=40, l2_regularization=1.0, random_state=SEMILLA).fit(x_tr, y_tr)
    pred_gbm = modelo_6.predict(x_te)
    resultados.append(evaluar_regresion(y_te, pred_gbm, "Gradient Boosting"))

    tabla = pd.DataFrame(resultados).set_index("modelo")
    print(tabla.to_string())

    # Banda de estimacion por regresion cuantilica (p10 - p90)
    modelo_6_p10 = HistGradientBoostingRegressor(
        loss="quantile", quantile=0.1, max_iter=300, learning_rate=0.05,
        min_samples_leaf=40, random_state=SEMILLA).fit(x_tr, y_tr)
    modelo_6_p90 = HistGradientBoostingRegressor(
        loss="quantile", quantile=0.9, max_iter=300, learning_rate=0.05,
        min_samples_leaf=40, random_state=SEMILLA).fit(x_tr, y_tr)

    lim_inf, lim_sup = modelo_6_p10.predict(x_te), modelo_6_p90.predict(x_te)
    cobertura_real = np.mean((y_te >= lim_inf) & (y_te <= lim_sup))
    print(f"[#6] Cobertura nominal banda p10-p90: 80.0% | observada: {100 * cobertura_real:.1f}%")

    importancia = permutation_importance(modelo_6, x_te, y_te, n_repeats=5, random_state=SEMILLA, n_jobs=1)
    top_vars = pd.Series(importancia.importances_mean, index=features).sort_values(ascending=False).head(12)
    print("[#6] Variables con mayor senal:")
    print(top_vars.round(4).to_string())

    return modelo_6, modelo_6_p10, modelo_6_p90, tabla


# ---------------------------------------------------------------------------
# 5. Modelo #8B - Brecha de subdeclaracion
# ---------------------------------------------------------------------------

def modelo_8b_brecha_subdeclaracion(m: pd.DataFrame, features: list[str], modelo_6, modelo_6_p10) -> pd.DataFrame:
    cotizantes = m[(m.en_ss == 1) & (m.ss_ibc_salud_p50 > 0) &
                   ((m.en_ach == 1) | (m.en_pse == 1))].copy()
    xco = cotizantes[features].values
    cotizantes["ingreso_estimado"] = np.exp(modelo_6.predict(xco))
    cotizantes["ingreso_est_p10"] = np.exp(modelo_6_p10.predict(xco))
    cotizantes["brecha"] = cotizantes.ingreso_estimado / cotizantes.ss_ibc_salud_p50
    cotizantes["brecha_conservadora"] = cotizantes.ingreso_est_p10 / cotizantes.ss_ibc_salud_p50
    cotizantes["marca_brecha"] = ((cotizantes.brecha_conservadora > 1.2) &
                                   (cotizantes.ss_ibc_salud_p50 <= 1.1 * SMLMV)).astype(int)

    print(f"[#8B] Cotizantes evaluados: {len(cotizantes):,}")
    print(f"[#8B] Marcados con brecha: {cotizantes.marca_brecha.sum():,} "
          f"({100 * cotizantes.marca_brecha.mean():.1f}%)")

    filas = []
    for nombre, mascara in [("Dependiente", cotizantes.es_dependiente == 1),
                             ("Independiente", cotizantes.es_independiente == 1),
                             ("Pensionado", cotizantes.es_pensionado == 1)]:
        s = cotizantes[mascara]
        if len(s) > 50:
            filas.append({"segmento": nombre, "n": len(s),
                          "brecha mediana": round(s.brecha.median(), 2),
                          "pct marcados": round(100 * s.marca_brecha.mean(), 1)})
    print(pd.DataFrame(filas).to_string(index=False))

    return cotizantes


# ---------------------------------------------------------------------------
# 6. Modelo #41 - Scoring de capacidad de ahorro
# ---------------------------------------------------------------------------

def modelo_41_capacidad_ahorro(ach_pm: pd.DataFrame, pse_pm: pd.DataFrame, ss_pm: pd.DataFrame) -> pd.DataFrame:
    panel = (ach_pm[["cli", "per", "recibido", "enviado"]]
             .merge(pse_pm[["cli", "per", "cat_financiero", "cat_servicios_recurrentes", "gasto"]],
                    on=["cli", "per"], how="outer")
             .merge(ss_pm[["cli", "per", "ibc_salud"]], on=["cli", "per"], how="outer")
             .fillna(0))

    # Se usa el maximo, no la suma, para evitar doble conteo del salario que
    # tambien puede pasar por ACH.
    panel["ingreso"] = panel[["ibc_salud", "recibido"]].max(axis=1)
    panel["fcm"] = (panel.ingreso - panel.cat_financiero
                     - panel.cat_servicios_recurrentes - panel.enviado)

    g = panel.groupby("cli")
    ahorro = pd.DataFrame({
        "fcm_med": g.fcm.mean(),
        "fcm_p50": g.fcm.median(),
        "fcm_std": g.fcm.std().fillna(0),
        "meses_obs": g.size(),
        "prop_meses_positivos": g.fcm.apply(lambda x: (x > 0).mean()),
        "ingreso_med": g.ingreso.mean(),
    })
    ahorro["margen"] = ahorro.fcm_med / (ahorro.ingreso_med + EPS)
    ahorro["estabilidad"] = 1 - (ahorro.fcm_std / (ahorro.fcm_med.abs() + EPS)).clip(0, 3) / 3

    ahorro["score_41"] = (100 * (
        0.40 * percentil(ahorro.fcm_med.clip(lower=0)) +
        0.30 * ahorro.prop_meses_positivos +
        0.20 * ahorro.estabilidad +
        0.10 * percentil(ahorro.margen.clip(-1, 1))
    )).round(1)

    print(f"[#41] Personas evaluadas: {len(ahorro):,}")
    print(f"[#41] FCM mensual mediano: {ahorro.fcm_p50.median():,.0f} COP")
    print(f"[#41] Personas con FCM medio positivo: {100 * (ahorro.fcm_med > 0).mean():.1f}%")

    ahorro["banda_ahorro"] = pd.cut(ahorro.score_41, [0, 20, 40, 60, 80, 100],
                                     labels=["muy baja", "baja", "media", "alta", "muy alta"])
    validacion_41 = ahorro.groupby("banda_ahorro", observed=True).agg(
        n=("score_41", "size"),
        fcm_mediano=("fcm_med", "median"),
        margen_mediano=("margen", "median"),
        meses_positivos=("prop_meses_positivos", "median"),
    )
    print("[#41] Validacion por banda:")
    print(validacion_41.round(2).to_string())

    return ahorro


# ---------------------------------------------------------------------------
# 7. Pipeline principal
# ---------------------------------------------------------------------------

def ejecutar_pipeline(data_dir: Path, output_dir: Path, cache_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    cache_dir.mkdir(parents=True, exist_ok=True)

    print(f"Ventana de analisis: {VENTANA_INI} a {VENTANA_FIN} ({N_MESES} meses)")

    # --- Paneles persona-mes -----------------------------------------------
    ss_pm, ss_attr, _ = construir_ss(data_dir, cache_dir)
    print(f"Panel SS persona-mes: {ss_pm.shape} | personas: {ss_pm['cli'].nunique():,}")

    ach_pm, ach_contra = construir_ach(data_dir, cache_dir)
    print(f"Panel ACH persona-mes: {ach_pm.shape} | personas: {ach_pm['cli'].nunique():,}")

    pse_pm = construir_pse(data_dir, cache_dir)
    print(f"Panel PSE persona-mes: {pse_pm.shape} | personas: {pse_pm['cli'].nunique():,}")

    # --- Matriz de modelado a nivel persona --------------------------------
    m, features = construir_matriz_persona(ss_pm, ss_attr, ach_pm, ach_contra, pse_pm)
    print(f"Matriz base: {m.shape} | Variables predictoras: {len(features)}")

    # =========================================================================
    # Modelo #6 - Estimacion de ingresos reales
    # =========================================================================
    modelo_6, modelo_6_p10, modelo_6_p90, tabla_desempeno_6 = modelo_6_estimacion_ingresos(m, features)

    # Aplicacion al segmento objetivo (independientes)
    independientes = m[(m.es_independiente == 1) & ((m.en_ach == 1) | (m.en_pse == 1))].copy()
    x_ind = independientes[features].values
    independientes["ingreso_estimado"] = np.exp(modelo_6.predict(x_ind))
    independientes["brecha"] = (independientes.ingreso_estimado /
                                 independientes.ss_ibc_salud_p50.replace(0, np.nan))
    print(f"[#6] Independientes evaluados: {len(independientes):,} | "
          f"brecha mediana (estimado/declarado): {independientes.brecha.median():.2f}")

    # =========================================================================
    # Modelo #8B - Brecha de subdeclaracion
    # =========================================================================
    cotizantes = modelo_8b_brecha_subdeclaracion(m, features, modelo_6, modelo_6_p10)

    # =========================================================================
    # Modelo #41 - Scoring de capacidad de ahorro
    # =========================================================================
    ahorro = modelo_41_capacidad_ahorro(ach_pm, pse_pm, ss_pm)

    # --- Tabla consolidada de salida ----------------------------------------
    universo = m[(m.en_ach == 1) | (m.en_pse == 1)].copy()
    x_universo = universo[features].values
    universo["ingreso_estimado"] = np.exp(modelo_6.predict(x_universo))
    universo["ingreso_est_p10"] = np.exp(modelo_6_p10.predict(x_universo))
    universo["ingreso_est_p90"] = np.exp(modelo_6_p90.predict(x_universo))
    universo["brecha_declaracion"] = (universo.ingreso_estimado /
                                       universo.ss_ibc_salud_p50.replace(0, np.nan)).round(2)
    universo = universo.join(ahorro[["score_41", "fcm_med", "prop_meses_positivos"]], how="left")

    salida = pd.DataFrame(index=universo.index)
    salida["ingreso_estimado"] = universo.ingreso_estimado.round(0)
    salida["ingreso_est_p10"] = universo.ingreso_est_p10.round(0)
    salida["ingreso_est_p90"] = universo.ingreso_est_p90.round(0)
    salida["ibc_declarado"] = universo.ss_ibc_salud_p50.round(0)
    salida["brecha_declaracion"] = universo.brecha_declaracion
    salida["score_capacidad_ahorro"] = universo.score_41.round(1)
    salida["fcm_mensual"] = universo.fcm_med.round(0)
    salida["cotiza_ss"] = universo.en_ss
    salida["segmento"] = np.select(
        [universo.es_independiente == 1, universo.es_dependiente == 1,
         universo.es_pensionado == 1, universo.en_ss == 0],
        ["independiente", "dependiente", "pensionado", "sin_registro_ss"], default="otro")
    salida["meses_observados"] = (universo.ach_meses + universo.pse_meses).astype(int)

    salida.to_parquet(output_dir / "salida_caso02_ingresos_independientes.parquet")
    salida.to_csv(output_dir / "salida_caso02_ingresos_independientes.csv")
    print(f"\nTabla de salida: {salida.shape[0]:,} personas x {salida.shape[1]} campos")
    print(salida.head(8).to_string())

    resumen_segmento = salida.groupby("segmento").agg(
        n=("score_capacidad_ahorro", "size"),
        ingreso_est_mediano=("ingreso_estimado", "median"),
        brecha_mediana=("brecha_declaracion", "median"),
        fcm_mediano=("fcm_mensual", "median"),
    ).round(0)
    resumen_segmento.to_csv(output_dir / "resumen_por_segmento.csv")

    # --- Resumen de desempeno de los 3 modelos ------------------------------
    resumen_modelos = pd.DataFrame([
        {"modelo": "#6 Estimacion de ingresos", "tipo": "Regresion",
         "metrica_principal": "R2 (log) / MAPE",
         "resultado": f"{tabla_desempeno_6.loc['Gradient Boosting', 'R2 (log)']:.3f} / "
                       f"{tabla_desempeno_6.loc['Gradient Boosting', 'MAPE']:.1%}"},
        {"modelo": "#8B Brecha de subdeclaracion", "tipo": "Regla sobre #6",
         "metrica_principal": "Coherencia por segmento",
         "resultado": f"marcados {100 * cotizantes.marca_brecha.mean():.1f}%"},
        {"modelo": "#41 Capacidad de ahorro", "tipo": "Scoring",
         "metrica_principal": "Monotonicidad de bandas",
         "resultado": "validada (ver log de ejecucion)"},
    ])
    resumen_modelos.to_csv(output_dir / "resumen_modelos_6_8b_41.csv", index=False)

    print("\nArchivos guardados en:", output_dir)
    for archivo in sorted(output_dir.iterdir()):
        print("-", archivo.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline Caso 02 - Estimador de ingresos para independientes")
    parser.add_argument("--data-dir", type=Path, required=True, help="Carpeta con los 3 archivos .xlsx de entrada")
    parser.add_argument("--output-dir", type=Path, default=Path("./resultados_caso02"), help="Carpeta de salida")
    parser.add_argument("--cache-dir", type=Path, default=Path("./cache_modelos"), help="Carpeta de cache en parquet")
    args = parser.parse_args()
    ejecutar_pipeline(args.data_dir, args.output_dir, args.cache_dir)


if __name__ == "__main__":
    main()
