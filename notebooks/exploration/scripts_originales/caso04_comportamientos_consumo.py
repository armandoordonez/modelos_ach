"""
ACH Data - Caso 04: Comportamientos de consumo
================================================

Script derivado del cuaderno original de Isabella Hernandez. Contiene el
pipeline de datos completo mas los 3 modelos del catalogo con mejor
desempeno tecnico:

    #15  Segmentacion RFM de consumidores        (clustering, silhouette 0.498)
    #17  Propension de compra por categoria       (clasificacion, ROC-AUC 0.835)
    #55  Propension a viajes y turismo            (clasificacion, ROC-AUC 0.741)

Se omitieron deliberadamente del cuaderno original, sin intentar corregirlos:

    #16  Share of wallet observado   -> senal debil (R2 = 0.231 sobre baseline)
    #57  Propension a suscripciones  -> etiqueta casi sin positivos (23 casos,
                                         F1 = 0.000, no aprende nada util)

Fuentes: Pagos Digitales (PSE), Transferencias ACH y Seguridad Social (SOI).

Uso
---
    python caso04_comportamientos_consumo.py \
        --data-dir /ruta/a/los/excel \
        --output-dir /ruta/de/salida

Los 3 archivos de entrada esperados en --data-dir (nombres configurables
en NOMBRES_ARCHIVOS):
    - Conversion Transferencias ACH 1 - Ofuscado.xlsx
    - Conversion Seguridad Social 1 - Ofuscado.xlsx
    - Conversion Pagos Digitales 1 - Ofuscado.xlsx
"""

from __future__ import annotations

import argparse
import re
import unicodedata
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")
pd.set_option("display.max_columns", 120)
pd.set_option("display.max_rows", 100)


# ---------------------------------------------------------------------------
# 1. Parametros generales
# ---------------------------------------------------------------------------

NOMBRES_ARCHIVOS = {
    "transferencias_ach": "Conversion Transferencias ACH 1 - Ofuscado.xlsx",
    "seguridad_social": "Conversion Seguridad Social 1 - Ofuscado.xlsx",
    "pagos_digitales": "Conversion Pagos Digitales 1 - Ofuscado.xlsx",
}

RANDOM_STATE = 42
TEST_SIZE = 0.20

# Proporcion final del tiempo usada como ventana objetivo en modelos supervisados.
PROPORCION_VENTANA_OBJETIVO = 0.25

# Categoria principal para el modelo #17. Si no aparece en la ventana
# objetivo, el script selecciona automaticamente una categoria con
# suficiente cobertura.
CATEGORIA_OBJETIVO_17 = "salud"

CONFIG_MANUAL = {
    "pagos_digitales": {"nombre": None, "documento": None, "fecha": None, "valor": None, "comercio": None},
    "transferencias_ach": {"nombre": None, "documento": None, "fecha": None, "valor": None, "contraparte": None},
    "seguridad_social": {"nombre": None, "documento": None, "fecha": None, "ibc": None},
}

PATRONES = {
    "nombre": ["nombre", "nom_persona", "razon_social", "titular"],
    "documento": ["documento", "identificacion", "cedula", "numero_id", "num_doc"],
    "fecha": ["fecha", "periodo", "mes", "timestamp"],
    "valor": ["valor", "monto", "importe", "total_transaccion"],
    "comercio": ["comercio", "merchant", "empresa", "entidad_autorizadora", "recaudador", "descripcion"],
    "contraparte": ["contraparte", "entidad", "originador", "receptor", "banco", "empresa"],
    "ibc": ["ibc", "ingreso_base", "base_cotizacion", "salario"],
}

TAXONOMIA = {
    "entidades_publicas": ["DIAN", "IMPUEST", "ALCALDIA", "GOBERNACION", "SECRETARIA", "RUNT", "TRANSITO", "CAMARA DE COMERCIO"],
    "entidades_financieras": ["BANCO", "DAVIVIENDA", "BANCOLOMBIA", "BBVA", "SCOTIABANK", "ITAU", "FINANCIERA", "CREDITO", "COOPERATIVA"],
    "turismo": ["AVIANCA", "LATAM", "WINGO", "JETSMART", "AEROLINEA", "HOTEL", "HOSTAL", "BOOKING", "AIRBNB", "VIAJE", "TURISMO", "DESPEGAR", "EXPEDIA"],
    "suscripciones_digitales": ["NETFLIX", "SPOTIFY", "DISNEY", "HBO", "MAX", "PRIME", "YOUTUBE", "GOOGLE", "APPLE", "MICROSOFT", "ADOBE", "CANVA", "OPENAI", "CHATGPT", "SOFTWARE", "STREAMING"],
    "educacion": ["UNIVERSIDAD", "COLEGIO", "INSTITUTO", "ICETEX", "ACADEMIA", "CURSO", "EDUCACION", "MATRICULA"],
    "salud": ["CLINICA", "HOSPITAL", "DROGUERIA", "FARMACIA", "SALUD", "EPS", "IPS", "ODONTO", "LABORATORIO"],
    "servicios_publicos": ["ENERGIA", "ACUEDUCTO", "AGUA", "GAS", "EMCALI", "EPM", "ENEL", "INTERNET", "TELEFONIA"],
    "comercio_minorista": ["SUPERMERCADO", "EXITO", "JUMBO", "D1", "ARA", "ALKOSTO", "FALABELLA", "MERCADO", "TIENDA"],
}


# ---------------------------------------------------------------------------
# 2. Carga de archivos
# ---------------------------------------------------------------------------

def leer_excel_robusto(ruta: Path, cache_dir: Path) -> pd.DataFrame:
    cache_dir.mkdir(parents=True, exist_ok=True)
    cache = cache_dir / f"{ruta.stem}.parquet"
    if cache.exists() and cache.stat().st_mtime >= ruta.stat().st_mtime:
        print(f"Cargando cache: {cache.name}")
        return pd.read_parquet(cache)
    cache_caso02 = {
        "Conversion Transferencias ACH 1 - Ofuscado": "ach.parquet",
        "Conversion Seguridad Social 1 - Ofuscado": "seguridad.parquet",
        "Conversion Pagos Digitales 1 - Ofuscado": "pagos.parquet",
    }
    cache_compartida = cache_dir.parent / "cache_caso02" / cache_caso02[ruta.stem]
    if cache_compartida.exists() and cache_compartida.stat().st_mtime >= ruta.stat().st_mtime:
        print(f"Cargando cache compartida: {cache_compartida.name}")
        return pd.read_parquet(cache_compartida)
    try:
        print(f"Leyendo Excel y creando cache: {ruta.name}")
        df = pd.read_excel(ruta, engine="openpyxl")
        df.to_parquet(cache, index=False)
        return df
    except Exception as exc:
        raise RuntimeError(f"No fue posible leer {ruta.name}: {exc}") from exc


def cargar_datasets(data_dir: Path, cache_dir: Path) -> dict[str, pd.DataFrame]:
    archivos = {fuente: data_dir / nombre for fuente, nombre in NOMBRES_ARCHIVOS.items()}
    faltantes = [str(ruta) for ruta in archivos.values() if not ruta.exists()]
    if faltantes:
        raise FileNotFoundError("No se encontraron estos archivos:\n" + "\n".join(faltantes))

    datasets = {fuente: leer_excel_robusto(ruta, cache_dir) for fuente, ruta in archivos.items()}
    for nombre, df in datasets.items():
        print(f"{nombre}: {df.shape[0]:,} filas x {df.shape[1]:,} columnas")
    return datasets


# ---------------------------------------------------------------------------
# 3. Estandarizacion de columnas y deteccion semiautomatica
# ---------------------------------------------------------------------------

def normalizar_nombre_columna(nombre) -> str:
    texto = str(nombre).strip().lower()
    texto = "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")
    texto = re.sub(r"[^a-z0-9]+", "_", texto)
    return texto.strip("_")


def estandarizar_columnas(df: pd.DataFrame) -> pd.DataFrame:
    resultado = df.copy()
    resultado.columns = [normalizar_nombre_columna(c) for c in resultado.columns]
    return resultado


def buscar_columna(df: pd.DataFrame, candidatos: list[str]) -> str | None:
    columnas = list(df.columns)
    for candidato in candidatos:
        if candidato in columnas:
            return candidato
    for candidato in candidatos:
        matches = [c for c in columnas if candidato in c]
        if matches:
            return matches[0]
    return None


def resolver_columnas(df: pd.DataFrame, fuente: str) -> dict:
    salida = {}
    for rol, valor_manual in CONFIG_MANUAL[fuente].items():
        if valor_manual is not None:
            if valor_manual not in df.columns:
                raise KeyError(f"La columna manual '{valor_manual}' no existe en {fuente}.")
            salida[rol] = valor_manual
        else:
            salida[rol] = buscar_columna(df, PATRONES[rol])
    return salida


# ---------------------------------------------------------------------------
# 4. Limpieza de identificadores, fechas y valores
# ---------------------------------------------------------------------------

def normalizar_texto(valor):
    if pd.isna(valor):
        return pd.NA
    texto = str(valor).strip().upper()
    texto = "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")
    texto = re.sub(r"[^A-Z0-9 ]+", " ", texto)
    texto = re.sub(r"\s+", " ", texto).strip()
    return texto if texto else pd.NA


def normalizar_documento(valor):
    if pd.isna(valor):
        return pd.NA
    texto = str(valor).strip()
    texto = re.sub(r"\.0$", "", texto)
    texto = re.sub(r"[^0-9*]", "", texto)
    return texto if texto else pd.NA


def a_numero(serie: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(serie):
        return pd.to_numeric(serie, errors="coerce")

    texto = serie.astype("string").str.strip()
    texto = texto.str.replace(r"[^0-9,.-]", "", regex=True)

    ambos = texto.str.contains(",", na=False) & texto.str.contains(".", na=False)
    texto.loc[ambos] = texto.loc[ambos].str.replace(",", "", regex=False)

    solo_coma = texto.str.contains(",", na=False) & ~texto.str.contains(".", na=False)
    texto.loc[solo_coma] = texto.loc[solo_coma].str.replace(",", ".", regex=False)

    return pd.to_numeric(texto, errors="coerce")


def preparar_fuente(df: pd.DataFrame, columnas: dict, prefijo: str) -> pd.DataFrame:
    salida = df.copy()

    col_nombre = columnas.get("nombre")
    col_documento = columnas.get("documento")
    col_fecha = columnas.get("fecha")

    salida["nombre_normalizado"] = salida[col_nombre].apply(normalizar_texto) if col_nombre else pd.NA

    if col_documento:
        salida["documento_normalizado"] = salida[col_documento].apply(normalizar_documento)
        salida["documento_visible"] = salida["documento_normalizado"].astype("string").str.replace("*", "", regex=False)
    else:
        salida["documento_normalizado"] = pd.NA
        salida["documento_visible"] = pd.NA

    salida["llave_persona"] = (
        salida["nombre_normalizado"].fillna("").astype(str) + "|" + salida["documento_visible"].fillna("").astype(str)
    )
    invalida = salida["nombre_normalizado"].isna() | (salida["llave_persona"].str.strip("|") == "")
    salida.loc[invalida, "llave_persona"] = pd.NA

    # Convertir primero a objeto evita conservar dtype ``category`` cuando
    # la fuente proviene de la cache optimizada del Caso 02.
    salida["fecha_analisis"] = (
        pd.to_datetime(salida[col_fecha].astype("object"), errors="coerce")
        if col_fecha
        else pd.NaT
    )

    return salida


# ---------------------------------------------------------------------------
# 5. Taxonomia de consumo PSE
# ---------------------------------------------------------------------------

def categorizar_comercio(valor) -> str:
    texto = normalizar_texto(valor)
    if pd.isna(texto):
        return "sin_informacion"
    for categoria, palabras in TAXONOMIA.items():
        if any(normalizar_texto(palabra) in texto for palabra in palabras):
            return categoria
    return "otros"


# ---------------------------------------------------------------------------
# 6. Perfil de consumo (recencia, frecuencia, monto, diversificacion)
# ---------------------------------------------------------------------------

def entropia_normalizada(serie: pd.Series) -> float:
    proporciones = serie / serie.sum()
    proporciones = proporciones[proporciones > 0]
    if len(proporciones) <= 1:
        return 0.0
    entropia = -(proporciones * np.log(proporciones)).sum()
    return float(entropia / np.log(len(proporciones)))


def construir_perfil_consumo(df: pd.DataFrame, perfil_ss: pd.DataFrame, perfil_ach: pd.DataFrame) -> pd.DataFrame:
    fecha_referencia = df["fecha_analisis"].max()

    base = (
        df.groupby("llave_persona")
        .agg(
            recencia_dias=("fecha_analisis", lambda s: (fecha_referencia - s.max()).days),
            frecuencia=("valor_analisis", "size"),
            monto_total=("valor_analisis", "sum"),
            monto_promedio=("valor_analisis", "mean"),
            monto_mediano=("valor_analisis", "median"),
            monto_desviacion=("valor_analisis", "std"),
            comercios_unicos=("comercio_normalizado", "nunique"),
            categorias_unicas=("categoria_consumo", "nunique"),
            meses_activos=("mes", "nunique"),
        )
        .reset_index()
    )

    gasto_cat = (
        df.pivot_table(index="llave_persona", columns="categoria_consumo", values="valor_analisis", aggfunc="sum", fill_value=0)
        .add_prefix("gasto_cat_")
        .reset_index()
    )
    frec_cat = (
        df.pivot_table(index="llave_persona", columns="categoria_consumo", values="valor_analisis", aggfunc="size", fill_value=0)
        .add_prefix("frec_cat_")
        .reset_index()
    )
    entropia = (
        df.groupby(["llave_persona", "categoria_consumo"])["valor_analisis"]
        .sum()
        .groupby(level=0)
        .apply(entropia_normalizada)
        .rename("indice_diversificacion")
        .reset_index()
    )

    perfil = (
        base.merge(gasto_cat, on="llave_persona", how="left")
        .merge(frec_cat, on="llave_persona", how="left")
        .merge(entropia, on="llave_persona", how="left")
        .merge(perfil_ss, on="llave_persona", how="left")
        .merge(perfil_ach, on="llave_persona", how="left")
    )

    perfil["ibc_promedio"] = pd.to_numeric(
        perfil["ibc_promedio"], errors="coerce"
    ).astype("float64")
    perfil["gasto_sobre_ibc"] = np.where(perfil["ibc_promedio"] > 0, perfil["monto_total"] / perfil["ibc_promedio"], np.nan)
    perfil["frecuencia_mensual"] = np.where(perfil["meses_activos"] > 0, perfil["frecuencia"] / perfil["meses_activos"], np.nan)
    perfil["ticket_cv"] = np.where(perfil["monto_promedio"] > 0, perfil["monto_desviacion"] / perfil["monto_promedio"], np.nan)

    return perfil


# ---------------------------------------------------------------------------
# 7. Modelo #15 - Segmentacion RFM
# ---------------------------------------------------------------------------

def modelo_15_rfm(perfil_historico: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rfm = perfil_historico[["llave_persona", "recencia_dias", "frecuencia", "monto_total"]].dropna().copy()

    for c in ["frecuencia", "monto_total"]:
        rfm[f"log_{c}"] = np.log1p(rfm[c])

    variables_rfm = ["recencia_dias", "log_frecuencia", "log_monto_total"]
    escalador_rfm = StandardScaler()
    x_rfm = escalador_rfm.fit_transform(rfm[variables_rfm])

    resultados_k = []
    max_k = min(8, len(rfm) - 1)
    for k in range(2, max_k + 1):
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=20)
        etiquetas = km.fit_predict(x_rfm)
        resultados_k.append({"k": k, "silhouette": silhouette_score(x_rfm, etiquetas)})
    resultados_k = pd.DataFrame(resultados_k)

    k_optimo = int(resultados_k.loc[resultados_k["silhouette"].idxmax(), "k"])
    modelo_rfm = KMeans(n_clusters=k_optimo, random_state=RANDOM_STATE, n_init=30)
    rfm["segmento_rfm"] = modelo_rfm.fit_predict(x_rfm)

    print(f"[#15] k seleccionado: {k_optimo} | silhouette: {resultados_k['silhouette'].max():.3f}")

    perfil_segmentos = (
        rfm.groupby("segmento_rfm")
        .agg(
            personas=("llave_persona", "nunique"),
            recencia_mediana=("recencia_dias", "median"),
            frecuencia_mediana=("frecuencia", "median"),
            monto_mediano=("monto_total", "median"),
        )
        .reset_index()
    )
    perfil_segmentos["etiqueta_negocio"] = "Consumo intermedio"
    perfil_segmentos.loc[perfil_segmentos["monto_mediano"].idxmax(), "etiqueta_negocio"] = "Alto valor"
    perfil_segmentos.loc[perfil_segmentos["frecuencia_mediana"].idxmax(), "etiqueta_negocio"] = "Alta frecuencia"
    perfil_segmentos.loc[perfil_segmentos["recencia_mediana"].idxmax(), "etiqueta_negocio"] = "En riesgo de inactividad"

    print(perfil_segmentos.to_string(index=False))

    return rfm, perfil_segmentos, resultados_k


# ---------------------------------------------------------------------------
# 8. Funcion comun para los modelos de propension (#17 y #55)
# ---------------------------------------------------------------------------

def preparar_x(df: pd.DataFrame, excluir: list[str]) -> pd.DataFrame:
    x = df.drop(columns=excluir, errors="ignore").copy()
    x = x.select_dtypes(include=[np.number, "bool"])
    x = x.replace([np.inf, -np.inf], np.nan)
    return x


def construir_target_categoria(df_objetivo: pd.DataFrame, categoria: str) -> pd.DataFrame:
    return (
        df_objetivo.assign(etiqueta=lambda d: (d["categoria_consumo"] == categoria).astype(int))
        .groupby("llave_persona")["etiqueta"]
        .max()
        .rename("target")
        .reset_index()
    )


def evaluar_clasificador_categoria(perfil_historico: pd.DataFrame, objetivo: pd.DataFrame, categoria: str, nombre_modelo: str):
    target = construir_target_categoria(objetivo, categoria)
    datos = perfil_historico.merge(target, on="llave_persona", how="inner").dropna(subset=["target"])

    if datos["target"].nunique() < 2:
        print(f"{nombre_modelo}: no se puede entrenar porque el objetivo solo contiene una clase para '{categoria}'.")
        return None

    x = preparar_x(datos, excluir=["target"])
    y = datos["target"].astype(int)
    estratificar = y if y.value_counts().min() >= 2 else None

    x_train, x_test, y_train, y_test = train_test_split(
        x, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=estratificar
    )

    modelo = Pipeline([
        ("imputacion", SimpleImputer(strategy="median")),
        ("modelo", RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_leaf=3,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1,
        )),
    ])
    modelo.fit(x_train, y_train)
    pred = modelo.predict(x_test)
    proba = modelo.predict_proba(x_test)[:, 1]

    clase_base = int(y_train.mode().iloc[0])
    pred_base = np.repeat(clase_base, len(y_test))

    auc = roc_auc_score(y_test, proba) if y_test.nunique() == 2 else np.nan

    metricas = pd.DataFrame([
        {
            "modelo": "Baseline mayoritaria",
            "accuracy": accuracy_score(y_test, pred_base),
            "precision": precision_score(y_test, pred_base, zero_division=0),
            "recall": recall_score(y_test, pred_base, zero_division=0),
            "f1": f1_score(y_test, pred_base, zero_division=0),
            "roc_auc": np.nan,
        },
        {
            "modelo": "Random Forest",
            "accuracy": accuracy_score(y_test, pred),
            "precision": precision_score(y_test, pred, zero_division=0),
            "recall": recall_score(y_test, pred, zero_division=0),
            "f1": f1_score(y_test, pred, zero_division=0),
            "roc_auc": auc,
        },
    ])

    print(f"\n{nombre_modelo}")
    print(f"Categoria: {categoria} | Personas: {len(datos):,}")
    print(metricas.to_string(index=False))
    print(classification_report(y_test, pred, zero_division=0))

    resultados = datos[["llave_persona", "target"]].copy()
    resultados["probabilidad"] = modelo.predict_proba(x)[:, 1]

    return {"categoria": categoria, "modelo": modelo, "metricas": metricas, "resultados": resultados, "datos": datos}


# ---------------------------------------------------------------------------
# 9. Pipeline principal
# ---------------------------------------------------------------------------

def ejecutar_pipeline(data_dir: Path, output_dir: Path, cache_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    # --- Carga y estandarizacion -----------------------------------------
    datasets = cargar_datasets(data_dir, cache_dir)
    df_ach = estandarizar_columnas(datasets["transferencias_ach"])
    df_ss = estandarizar_columnas(datasets["seguridad_social"])
    df_pse = estandarizar_columnas(datasets["pagos_digitales"])

    columnas = {
        "pagos_digitales": resolver_columnas(df_pse, "pagos_digitales"),
        "transferencias_ach": resolver_columnas(df_ach, "transferencias_ach"),
        "seguridad_social": resolver_columnas(df_ss, "seguridad_social"),
    }

    requeridas_pse = ["nombre", "documento", "fecha", "valor", "comercio"]
    faltantes_pse = [r for r in requeridas_pse if columnas["pagos_digitales"].get(r) is None]
    if faltantes_pse:
        print("ADVERTENCIA: no se detectaron estas columnas de PSE:", faltantes_pse)

    # --- Limpieza de identificadores, fechas y valores --------------------
    pse = preparar_fuente(df_pse, columnas["pagos_digitales"], "pse")
    ach = preparar_fuente(df_ach, columnas["transferencias_ach"], "ach")
    ss = preparar_fuente(df_ss, columnas["seguridad_social"], "ss")

    pse["valor_analisis"] = a_numero(pse[columnas["pagos_digitales"]["valor"]]) if columnas["pagos_digitales"]["valor"] else np.nan
    ach["valor_analisis"] = a_numero(ach[columnas["transferencias_ach"]["valor"]]) if columnas["transferencias_ach"]["valor"] else np.nan
    ss["ibc_analisis"] = a_numero(ss[columnas["seguridad_social"]["ibc"]]) if columnas["seguridad_social"]["ibc"] else np.nan

    # --- Taxonomia de consumo PSE ------------------------------------------
    col_comercio = columnas["pagos_digitales"].get("comercio")
    if col_comercio is None:
        raise ValueError("No se detecto el campo de comercio de PSE. Configuralo manualmente en CONFIG_MANUAL.")
    pse["comercio_normalizado"] = pse[col_comercio].apply(normalizar_texto)
    pse["categoria_consumo"] = pse[col_comercio].apply(categorizar_comercio)

    # --- Filtrado a filas utiles para modelado -----------------------------
    pse_modelo = pse.dropna(subset=["llave_persona", "fecha_analisis", "valor_analisis"]).copy()
    pse_modelo = pse_modelo[pse_modelo["valor_analisis"] >= 0]
    pse_modelo["mes"] = pse_modelo["fecha_analisis"].dt.to_period("M").dt.to_timestamp()
    print(f"Filas utiles para modelado: {len(pse_modelo):,} | Personas: {pse_modelo['llave_persona'].nunique():,}")

    # --- Integracion de IBC y senales ACH -----------------------------------
    perfil_ss = (
        ss.dropna(subset=["llave_persona"])
        .groupby("llave_persona")
        .agg(ibc_promedio=("ibc_analisis", "mean"), ibc_mediano=("ibc_analisis", "median"), registros_ss=("llave_persona", "size"))
        .reset_index()
    )
    perfil_ach = (
        ach.dropna(subset=["llave_persona"])
        .groupby("llave_persona")
        .agg(
            ach_valor_total=("valor_analisis", "sum"),
            ach_valor_promedio=("valor_analisis", "mean"),
            ach_transacciones=("llave_persona", "size"),
            ach_meses_activos=("fecha_analisis", lambda s: s.dt.to_period("M").nunique()),
        )
        .reset_index()
    )

    # --- Ventanas temporales (evita fuga de informacion) --------------------
    fecha_min = pse_modelo["fecha_analisis"].min()
    fecha_max = pse_modelo["fecha_analisis"].max()
    rango_dias = max((fecha_max - fecha_min).days, 1)
    dias_objetivo = max(int(rango_dias * PROPORCION_VENTANA_OBJETIVO), 30)
    fecha_corte = fecha_max - pd.Timedelta(days=dias_objetivo)

    historico = pse_modelo[pse_modelo["fecha_analisis"] < fecha_corte].copy()
    objetivo = pse_modelo[pse_modelo["fecha_analisis"] >= fecha_corte].copy()
    print(f"Ventana historica: {historico['fecha_analisis'].min()} a {historico['fecha_analisis'].max()}")
    print(f"Ventana objetivo:  {objetivo['fecha_analisis'].min()} a {objetivo['fecha_analisis'].max()}")

    perfil_historico = construir_perfil_consumo(historico, perfil_ss, perfil_ach)
    print(f"Perfil historico: {perfil_historico.shape}")

    # =========================================================================
    # Modelo #15 - Segmentacion RFM
    # =========================================================================
    rfm, perfil_segmentos, resultados_k = modelo_15_rfm(perfil_historico)

    # =========================================================================
    # Modelo #17 - Propension de compra por categoria
    # =========================================================================
    categorias_disponibles = objetivo["categoria_consumo"].value_counts()
    categoria_17 = CATEGORIA_OBJETIVO_17
    if categoria_17 not in categorias_disponibles.index:
        candidatas = categorias_disponibles.drop(labels=["otros", "sin_informacion"], errors="ignore")
        categoria_17 = candidatas.index[0] if not candidatas.empty else categorias_disponibles.index[0]
    print(f"\nCategoria seleccionada para #17: {categoria_17}")

    resultado_17 = evaluar_clasificador_categoria(
        perfil_historico, objetivo, categoria_17, "#17 Propension de compra por categoria"
    )

    # =========================================================================
    # Modelo #55 - Propension a viajes y turismo
    # =========================================================================
    resultado_55 = evaluar_clasificador_categoria(
        perfil_historico, objetivo, "turismo", "#55 Propension a viajes y turismo"
    )

    # --- Viabilidad consolidada ----------------------------------------------
    def _f1_rf(resultado):
        if resultado is None:
            return np.nan
        return float(resultado["metricas"].loc[resultado["metricas"]["modelo"] == "Random Forest", "f1"].iloc[0])

    viabilidad = pd.DataFrame([
        {"modelo": "#15", "nombre": "Segmentacion RFM", "tipo": "Clustering",
         "estado": "Ejecutado", "metrica_principal": float(resultados_k["silhouette"].max())},
        {"modelo": "#17", "nombre": "Propension por categoria", "tipo": "Clasificacion",
         "estado": "Ejecutado" if resultado_17 is not None else "Parcial / sin clases suficientes",
         "metrica_principal": _f1_rf(resultado_17)},
        {"modelo": "#55", "nombre": "Propension a turismo", "tipo": "Clasificacion",
         "estado": "Ejecutado" if resultado_55 is not None else "Parcial / sin clases suficientes",
         "metrica_principal": _f1_rf(resultado_55)},
    ])
    print("\nViabilidad de los modelos seleccionados:")
    print(viabilidad.to_string(index=False))

    # --- Salida consolidada por persona --------------------------------------
    salida_personas = perfil_historico.copy()
    salida_personas = salida_personas.merge(rfm[["llave_persona", "segmento_rfm"]], on="llave_persona", how="left")
    salida_personas = salida_personas.merge(
        perfil_segmentos[["segmento_rfm", "etiqueta_negocio"]], on="segmento_rfm", how="left"
    )

    for numero, resultado in [("17", resultado_17), ("55", resultado_55)]:
        if resultado is not None:
            probabilidades = resultado["resultados"][["llave_persona", "probabilidad"]].rename(
                columns={"probabilidad": f"probabilidad_modelo_{numero}"}
            )
            salida_personas = salida_personas.merge(probabilidades, on="llave_persona", how="left")

    print(f"\nSalida consolidada: {salida_personas.shape}")

    # --- Exportacion de entregables -------------------------------------------
    salida_personas.to_csv(output_dir / "perfil_comportamiento_consumo.csv", index=False)
    rfm.to_csv(output_dir / "segmentacion_rfm.csv", index=False)
    viabilidad.to_csv(output_dir / "viabilidad_modelos_15_17_55.csv", index=False)

    print("\nArchivos guardados en:", output_dir)
    for archivo in sorted(output_dir.iterdir()):
        print("-", archivo.name)


def main() -> None:
    parser = argparse.ArgumentParser(description="Pipeline Caso 04 - Comportamientos de consumo")
    parser.add_argument("--data-dir", type=Path, required=True, help="Carpeta con los 3 archivos .xlsx de entrada")
    parser.add_argument("--output-dir", type=Path, default=Path("./resultados_caso04"), help="Carpeta de salida")
    parser.add_argument("--cache-dir", type=Path, default=Path("./cache_caso04"), help="Carpeta de cache en parquet")
    args = parser.parse_args()
    ejecutar_pipeline(args.data_dir, args.output_dir, args.cache_dir)


if __name__ == "__main__":
    main()
