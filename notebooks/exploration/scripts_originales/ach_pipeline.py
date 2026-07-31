# -*- coding: utf-8 -*-
"""
Capa de INGENIERÍA DE DATOS — Proyecto ACH Data · Caso de uso 5 (Clustering).

Este módulo es la frontera entre ingeniería de datos y ciencia de datos: los
notebooks de modelado NO leen los parquet crudos, importan de aquí. Responsabilidades:

  1. load_raw(fuente)           -> carga el extracto ofuscado (cache Data/parquet/).
  2. add_person_key(df, fuente) -> LLAVE DE PERSONA consistente entre las 3 fuentes,
                                   resistente a la ofuscación.
  3. decode_entidad_autorizadora(serie) -> recupera el banco desde el PREFIJO visible
                                   (no del nº de asteriscos); DESCONOCIDO si va enmascarado.
  4. categorize_comercio(serie) -> taxonomía de consumo por palabras clave (PSE).
  5. build_person_month(...)    -> tabla analítica persona × mes (base de los modelos).

PRINCIPIO METODOLÓGICO: todo lo que se calcula aquí usa SOLO caracteres realmente
visibles en el dato. No se imputa ningún valor enmascarado por conjetura. Cuando una
inferencia no es inequívoca se marca con un nivel de confianza y se conserva el crudo.

--- La llave de persona (resuelve la ofuscación de identidad) -----------------
El `Número documento` llega enmascarado idéntico en las 3 fuentes (primeros 8
dígitos + `****`); usado solo, colisiona: en Seguridad Social el 13,0% de las
cédulas enmascaradas cubren >1 persona distinta (hasta 13 personas bajo una llave).
La llave compuesta = documento enmascarado + longitud del nombre + última letra del
nombre. Las tres piezas están disponibles y son comparables en las 3 fuentes
(el largo se conserva como nº de caracteres; la última letra es visible en todas).
Con esa llave la colisión baja a 0,25%. Es un cuasi-identificador, no un match
exacto: la solución definitiva sigue siendo re-solicitar la llave hasheada.
"""
from __future__ import annotations

import hashlib
import re
from pathlib import Path

import numpy as np
import pandas as pd

# --------------------------------------------------------------------------- #
# Rutas y nombres de columna por fuente (SS usa "… de documento"; TRF/PSE no)  #
# --------------------------------------------------------------------------- #
ROOT = Path(__file__).resolve().parents[1]
PARQUET_DIR = ROOT / "Data" / "parquet"
PROCESSED_DIR = ROOT / "Data" / "processed"

FUENTES = {
    "ss":  {"file": "seguridad_social.parquet",   "doc": "Número de documento",
            "tdoc": "Tipo de documento", "name": "Nombre", "periodo": "Periodo cotización"},
    "trf": {"file": "transferencias_ach.parquet", "doc": "Número documento",
            "tdoc": "Tipo documento",    "name": "Nombre", "periodo": "Periodo"},
    "pse": {"file": "pagos_digitales.parquet",     "doc": "Número documento",
            "tdoc": "Tipo documento",    "name": "Nombre", "periodo": "Periodo"},
}


def load_raw(fuente: str) -> pd.DataFrame:
    """Carga un extracto ofuscado desde el cache parquet."""
    if fuente not in FUENTES:
        raise ValueError(f"fuente debe ser una de {list(FUENTES)}")
    return pd.read_parquet(PARQUET_DIR / FUENTES[fuente]["file"])


def num(s: pd.Series) -> pd.Series:
    """Serie numérica float64 robusta (tolera strings ofuscados y nulos)."""
    return pd.to_numeric(s, errors="coerce").astype("float64")


# --------------------------------------------------------------------------- #
# 1) LLAVE DE PERSONA consistente entre fuentes                                #
# --------------------------------------------------------------------------- #
def person_key_parts(df: pd.DataFrame, fuente: str) -> pd.DataFrame:
    """Devuelve las piezas de la llave: doc, largo y última letra del nombre."""
    cfg = FUENTES[fuente]
    doc = df[cfg["doc"]].astype("string").str.strip()
    tdoc = df[cfg["tdoc"]].astype("string").str.strip().str.upper()
    name = df[cfg["name"]].astype("string").str.strip()
    return pd.DataFrame({
        "doc": doc,
        "tdoc": tdoc,
        "name_len": name.str.len().astype("Int64"),
        "name_last": name.str[-1:].str.upper(),
    })


def _hash_key(parts: pd.DataFrame) -> pd.Series:
    """SHA-1 corto del compuesto tdoc|doc|largo|última letra (llave estable)."""
    compuesto = (parts["tdoc"].fillna("") + "|" + parts["doc"].fillna("") + "|"
                 + parts["name_len"].astype("string").fillna("") + "|"
                 + parts["name_last"].fillna(""))

    def h(x: str) -> str:
        return hashlib.sha1(x.encode("utf-8")).hexdigest()[:16]

    return compuesto.map(h).astype("string")


def add_person_key(df: pd.DataFrame, fuente: str) -> pd.DataFrame:
    """Agrega `person_id` = CÉDULA enmascarada (documento).

    DECISIÓN (iteración 2): se cruza SOLO por documento. El documento se enmascara
    idéntico en las 3 fuentes, así que esta llave permite cruzar las TRES de forma
    directa (SS entra a ambos modelos). Se ASUME el riesgo de colisión —~13% de las
    cédulas enmascaradas cubren >1 persona (ver `collision_report`)—, que el negocio
    considera bajo/aceptable (afecta a pocos clientes). La llave compuesta
    (`person_key_parts` + `_hash_key`) queda disponible solo para documentar ese
    riesgo en el EDA."""
    cfg = FUENTES[fuente]
    out = df.copy()
    out["person_id"] = df[cfg["doc"]].astype("string").str.strip()
    return out


# --------------------------------------------------------------------------- #
# 2) DECODIFICADOR de 'Entidad autorizadora' (PSE)                             #
#    Se lee el PREFIJO visible y se mapea contra un catálogo canónico por      #
#    prefijo inequívoco. Nada se adivina por el número de asteriscos.          #
# --------------------------------------------------------------------------- #
# Catálogo canónico (nombre tal como aparecería sin enmascarar, en mayúsculas).
CATALOGO_ENTIDADES = [
    "BANCOLOMBIA", "BANCO DE BOGOTA", "BANCO DAVIVIENDA", "BANCO BBVA COLOMBIA",
    "BANCO CAJA SOCIAL", "BANCO POPULAR", "BANCO AGRARIO DE COLOMBIA",
    "BANCO DE OCCIDENTE", "BANCO AV VILLAS", "SCOTIABANK COLPATRIA",
    "BANCO GNB SUDAMERIS", "BANCO FALABELLA", "BANCO PICHINCHA", "BANCO ITAU",
    "BANCOOMEVA", "CITIBANK", "BANCO COOPERATIVO COOPCENTRAL", "BANCO FINANDINA",
    "BANCO MUNDO MUJER", "BANCO W", "BANCO SERFINANZA", "BANCAMIA", "BANCO SANTANDER",
    "LULO BANK", "NEQUI", "DAVIPLATA", "NU", "COLTEFINANCIERA", "RAPPIPAY",
]
# Etiqueta comercial normalizada (varios canónicos pueden colapsar a un mismo banco).
_ETIQUETA = {c: c for c in CATALOGO_ENTIDADES}
_ETIQUETA["BANCO DAVIVIENDA"] = "DAVIVIENDA"
_ETIQUETA["BANCO BBVA COLOMBIA"] = "BBVA COLOMBIA"
_ETIQUETA["BANCO COOPERATIVO COOPCENTRAL"] = "COOPCENTRAL"
_ETIQUETA["SCOTIABANK COLPATRIA"] = "SCOTIABANK COLPATRIA"

# Overrides curados para prefijos ambiguos de ALTO volumen: decisión documentada,
# marcada con confianza < alta para que el modelo pueda descartarla si conviene.
_OVERRIDES = {
    # "BAN"+8 asteriscos: BANCOLOMBIA es el único banco de una sola palabra que
    # arranca en "BAN" y domina el mercado (>200k filas). Confianza media.
    "BAN": ("BANCOLOMBIA", "media"),
}


def _decode_one(valor: str) -> tuple[str, str]:
    """(etiqueta, confianza) para un valor de 'Entidad autorizadora'."""
    if valor is None:
        return ("DESCONOCIDO", "nula")
    s = str(valor).upper().strip()
    if s == "" or s.startswith("*"):          # totalmente enmascarada
        return ("DESCONOCIDO", "nula")
    prefijo = re.sub(r"\*+$", "", s).strip()   # quita asteriscos de relleno
    if prefijo == "":
        return ("DESCONOCIDO", "nula")
    if prefijo in _OVERRIDES:
        return _OVERRIDES[prefijo]
    candidatos = {_ETIQUETA[c] for c in CATALOGO_ENTIDADES if c.startswith(prefijo)}
    if len(candidatos) == 1:
        etiqueta = next(iter(candidatos))
        conf = "alta" if len(prefijo) >= 5 else "media"
        return (etiqueta, conf)
    if len(candidatos) == 0:
        return (prefijo, "sin_catalogo")       # visible pero fuera del catálogo
    return ("AMBIGUO", "baja")                  # prefijo compatible con >1 banco


def decode_entidad_autorizadora(serie: pd.Series) -> pd.DataFrame:
    """Vectoriza el decodificador sobre valores únicos (rápido). Devuelve
    columnas `entidad` y `entidad_confianza`."""
    unicos = serie.astype("string")
    tabla = {v: _decode_one(v) for v in unicos.dropna().unique()}
    etiqueta = unicos.map(lambda v: tabla.get(v, ("DESCONOCIDO", "nula"))[0])
    conf = unicos.map(lambda v: tabla.get(v, ("DESCONOCIDO", "nula"))[1])
    return pd.DataFrame({"entidad": etiqueta, "entidad_confianza": conf}, index=serie.index)


# --------------------------------------------------------------------------- #
# 3) TAXONOMÍA de comercios PSE (reutilizada del EDA)                          #
# --------------------------------------------------------------------------- #
CATEGORIAS_COMERCIO = {
    "Pasarelas / agregadores": ["PAYU", "KUSHKI", "EPAYCO", "PLACETOPAY", "PLACE TO PAY",
        "MERCADOPAGO", "MERCADO PAGO", "WOMPI", "PAYVALIDA", "PAGOSONLINE", "TU COMPRA",
        "PAYZEN", "PEXTO", "COBRE"],
    "Financiero / créditos": ["BANCO", "BANCOL", "DAVIVIEN", "NEQUI", "DAVIPLATA", "CREDITO",
        "CRÉDITO", "FINANC", "COOPERATIVA", "FONDO", "SUFI", "TARJETA", "LULO", "RAPPIPAY",
        "ADDI", "SISTECREDITO", "COLTEFINANCIERA", "JURISCOOP", "CONFIAR", "SCOTIA", "ITAU",
        "ITAÚ", "BBVA", "COLPATRIA", "AV VILLAS", "POPULAR", "PICHINCHA", "FALABELLA CMR",
        "SERFINANZA", "FINANDINA", "BANCAMIA", "MIBANCO"],
    "Apuestas y juegos": ["BETPLAY", "WPLAY", "RUSHBET", "LUCKIA", "CODERE", "STAKE", "BWIN",
        "APUESTA", "RIVALO", "YAJUEGO", "SPORTIUM", "MEGAPUESTA", "BETSSON", "ZAMBA", "GANA"],
    "Gobierno / impuestos": ["HACIENDA", "ALCALDIA", "ALCALDÍA", "GOBERNACION", "GOBERNACIÓN",
        "DIAN", "SECRETARIA", "SECRETARÍA", "MUNICIPIO", "TRANSITO", "TRÁNSITO", "RUNT",
        "REGISTRADURIA", "DISTRIT", "MINISTERIO", "FISCAL", "POLICIA", "IMPUESTO"],
    "Seguridad social / nómina": ["SOI", "APORTES EN LINEA", "APORTES EN LÍNEA", "ARUS",
        "MI PLANILLA", "COLPENSIONES", "PROTECCION", "PROTECCIÓN", "PORVENIR", "COMPENSAR",
        "COLFONDOS", "SIMPLE S.A", "ASOPAGOS", "ENLACE OPERATIVO"],
    "Telco / servicios públicos": ["CLARO", "MOVISTAR", "TIGO", "WOM ", "ETB", "EPM", "CODENSA",
        "ENEL", "VANTI", "GASES", "ACUEDUCTO", "AIRE", "AFINIA", "EMCALI", "CELSIA", "ESSA",
        "ELECTRIFICADORA", "ENERGIA", "ENERGÍA"],
    "Comercio electrónico / retail": ["MERCADOLIBRE", "MERCADO LIBRE", "AMAZON", "ALIEXPRESS",
        "FALABELLA", "EXITO", "ÉXITO", "OLIMPICA", "OLÍMPICA", "ALKOSTO", "KTRONIX", "TEMU",
        "SHEIN", "LINIO", "HOMECENTER", "JUMBO", "MAKRO", "PRICESMART", "D1 ", "ARA ", "SODIMAC"],
    "Streaming / digital": ["NETFLIX", "SPOTIFY", "DISNEY", "HBO", "PARAMOUNT", "APPLE", "GOOGLE",
        "MICROSOFT", "PLAYSTATION", "STEAM", "TWITCH", "PRIME", "CRUNCHYROLL"],
    "Educación": ["UNIVERSIDAD", "ICETEX", "COLEGIO", "ICFES", "SENA", "ACADEM", "EDUCA",
        "POLITECNICO", "POLITÉCNICO", "UNIMINUTO", "CORPORACION UNIV", "FUNDACION UNIV",
        "GIMNASIO", "LICEO"],
    "Viajes / transporte": ["AVIANCA", "LATAM", "VIVA", "WINGO", "SATENA", "BOOKING", "DESPEGAR",
        "AEROL", "HOTEL", "EXPRESO", "BOLIVARIANO", "TERMINAL", "UBER", "DIDI", "CABIFY",
        "REDBUS", "PULLMAN"],
    "Salud": ["EPS", "FARMAC", "DROGUERIA", "DROGUERÍA", "CLINICA", "CLÍNICA", "HOSPITAL",
        "SANITAS", "COLSANITAS", "MEDPLUS", "AUDIFARMA", "CRUZ VERDE", "COLSUBSIDIO", "SALUD",
        "LABORATORIO"],
    "Seguros": ["SEGURO", "ALLIANZ", "MAPFRE", "AXA", "BOLIVAR", "BOLÍVAR", "LIBERTY",
        "PREVISORA", "MUNDIAL", "SOAT", "HDI", "CHUBB", "SBS"],
}


def _categorizar_comercio(nombre: str) -> str:
    n = str(nombre).upper()
    for cat, claves in CATEGORIAS_COMERCIO.items():
        if any(k in n for k in claves):
            return cat
    return "Otros / no clasificado"


def categorize_comercio(serie: pd.Series) -> pd.Series:
    """Mapea cada comercio ofuscado a una categoría temática (vía valores únicos)."""
    mapa = {c: _categorizar_comercio(c) for c in serie.dropna().astype("string").unique()}
    return serie.astype("string").map(mapa).fillna("Otros / no clasificado")


# --------------------------------------------------------------------------- #
# 4) TABLAS ANALÍTICAS persona × mes (base de los modelos de clustering)       #
#    Se agregan las columnas de DETALLE (Valor, Cantidad); no se usan las      #
#    columnas de resumen repetidas, para no mezclar niveles de grano.          #
# --------------------------------------------------------------------------- #
VENTANA_COMUN = ("2025-01", "2026-06")   # rango común a las 3 fuentes


def _en_ventana(periodo: pd.Series, ventana=VENTANA_COMUN) -> pd.Series:
    p = periodo.astype("string").str.slice(0, 7)
    return p.between(ventana[0], ventana[1])


def build_person_month(fuente: str, solo_ventana_comun: bool = False) -> pd.DataFrame:
    """Tabla persona × período con features transaccionales de la fuente.
    Deduplica filas exactas. En Transferencias separa lo EXTERNO (ingreso/gasto
    real) de lo de CUENTA PROPIA (movimientos entre cuentas de la misma persona,
    que NO son ingreso ni gasto)."""
    df = add_person_key(load_raw(fuente).drop_duplicates(), fuente)
    cfg = FUENTES[fuente]
    per = df[cfg["periodo"]].astype("string").str.slice(0, 7)

    if fuente == "trf":
        v, cnt = num(df["Valor"]), num(df["Cantidad de transferencias"])
        env = df["Tipo registro"].astype("string").str.upper().eq("ENVIADO")
        serv = df["Servicio"].astype("string").str.upper()
        propio = df["Cuenta propia"].astype("string").str.upper().eq("SI")
        ext = ~propio                                   # transferencia con terceros
        base = pd.DataFrame({
            "person_id": df["person_id"], "periodo": per,
            # recibido/enviado = SOLO externo → ingreso y gasto reales
            "recibido": v.where(~env & ext, 0.0), "enviado": v.where(env & ext, 0.0),
            "cuenta_propia_total": v.where(propio, 0.0),   # movimientos internos (no ingreso)
            "n_recibidas": cnt.where(~env & ext, 0.0), "n_enviadas": cnt.where(env & ext, 0.0),
            "recibido_ach": v.where(~env & ext & serv.eq("ACH"), 0.0),
            "recibido_transfiya": v.where(~env & ext & serv.eq("TRANSFIYA"), 0.0),
        })
        pm = base.groupby(["person_id", "periodo"], as_index=False).sum()

    elif fuente == "pse":
        v, cnt = num(df["Valor"]), num(df["Cantidad"])
        cat = categorize_comercio(df["Comercio"])
        base = pd.DataFrame({"person_id": df["person_id"], "periodo": per,
                             "gasto": v, "n_pagos": cnt, "categoria": cat})
        pm = (base.groupby(["person_id", "periodo"], as_index=False)
              .agg(gasto_pse=("gasto", "sum"), n_pagos=("n_pagos", "sum"),
                   n_comercios=("categoria", "size")))
        piv = (base.pivot_table(index=["person_id", "periodo"], columns="categoria",
                                values="gasto", aggfunc="sum", fill_value=0.0)
               .add_prefix("gasto_").reset_index())
        pm = pm.merge(piv, on=["person_id", "periodo"], how="left")

    elif fuente == "ss":
        base = pd.DataFrame({
            "person_id": df["person_id"], "periodo": per,
            "ibc_salud": num(df["Ingreso base salud"]),
            "salario_basico": num(df["Salario básico"]),
            "dias_pension": num(df["Días pensión"]),
            "dias_salud": num(df["Días salud"]),
        })
        pm = (base.groupby(["person_id", "periodo"], as_index=False)
              .agg(ibc_salud=("ibc_salud", "max"), salario_basico=("salario_basico", "max"),
                   dias_pension=("dias_pension", "sum"), dias_salud=("dias_salud", "sum")))
    else:
        raise ValueError(fuente)

    if solo_ventana_comun:
        pm = pm[_en_ventana(pm["periodo"])].reset_index(drop=True)
    return pm


def materialize(dest_dir: Path = PROCESSED_DIR) -> dict:
    """Genera y guarda a parquet las 3 tablas persona × mes. Devuelve shapes."""
    dest_dir.mkdir(parents=True, exist_ok=True)
    shapes = {}
    for f in ["ss", "trf", "pse"]:
        pm = build_person_month(f)
        out = dest_dir / f"person_month_{f}.parquet"
        pm.to_parquet(out, index=False)
        shapes[f] = pm.shape
    return shapes


def person_bridge_map(fuente: str) -> pd.DataFrame:
    """person_id -> bridge (documento + última letra). El bridge es una llave más
    laxa que SÍ es comparable entre SS y los transaccionales (donde person_id no lo
    es, porque SS enmascara el nombre distinto). Se usa solo para enriquecer, nunca
    como identidad primaria."""
    df = load_raw(fuente)
    p = person_key_parts(df, fuente)
    bridge = p["doc"].fillna("") + "|" + p["name_last"].fillna("")
    return (pd.DataFrame({"person_id": _hash_key(p), "bridge": bridge})
            .drop_duplicates("person_id").reset_index(drop=True))


# --------------------------------------------------------------------------- #
# 5) FEATURES POR MODELO (Caso de uso 5 · clustering)                          #
# --------------------------------------------------------------------------- #
GASTO_CATS = [  # categorías de consumo PSE (columnas gasto_* de person_month_pse)
    "gasto_Financiero / créditos", "gasto_Comercio electrónico / retail",
    "gasto_Gobierno / impuestos", "gasto_Telco / servicios públicos", "gasto_Salud",
    "gasto_Educación", "gasto_Seguros", "gasto_Viajes / transporte",
    "gasto_Streaming / digital", "gasto_Apuestas y juegos",
    "gasto_Seguridad social / nómina", "gasto_Pasarelas / agregadores",
    "gasto_Otros / no clasificado",
]
FIN_END = "2026-06"   # fin de la ventana común (para recencia)


def _meses_desde(periodo_serie: pd.Series, fin: str = FIN_END) -> pd.Series:
    """Meses transcurridos entre cada 'YYYY-MM' y `fin` (recencia)."""
    a = pd.PeriodIndex(periodo_serie.astype("string"), freq="M")
    b = pd.Period(fin, freq="M")
    return pd.Series((b - a).map(lambda x: x.n), index=periodo_serie.index)


def build_person_ss_profile() -> pd.DataFrame:
    """Clasificación persona-nivel desde Seguridad Social (cédula, deduplicado):
    tipo de persona, cobertura de prestaciones, ingreso declarado y dinámica del IBC.
    Alimenta el enriquecimiento de AMBOS modelos."""
    df = add_person_key(load_raw("ss").drop_duplicates(), "ss")
    rel = df["Relación laboral"].astype("string").str.upper()
    tap = df["Tipo aportante"].astype("string").str.upper()
    plan = df["Tipo planilla"].astype("string").str.upper()

    # Tipo por fila (el orden importa: pensionado pisa a empleado/independiente)
    tipo = pd.Series("Otro", index=df.index, dtype="object")
    tipo[rel.eq("DEPENDIENTE") | tap.eq("EMPLEADOR") | plan.eq("E")] = "Empleado"
    tipo[rel.str.startswith("INDEPENDIENTE", na=False) | tap.eq("INDEPENDIENTE")
         | plan.isin(["I", "Y"])] = "Independiente"
    tipo[rel.str.contains("APRENDIC", na=False)] = "Aprendiz"
    tipo[rel.str.contains("MAGISTERIO", na=False) | tap.str.contains("MAGISTERIO", na=False)] = "Magisterio"
    tipo[rel.str.contains("PENSIONADO", case=False, na=False) | plan.eq("P")
         | tap.str.contains("PENSIONES", na=False)] = "Pensionado"

    d = df.assign(
        _tipo=tipo,
        _pension=num(df["Días pensión"]) > 0, _salud=num(df["Días salud"]) > 0,
        _caja=num(df["Días caja compensación"]) > 0, _riesgos=num(df["Días riesgos"]) > 0,
        _ibc=num(df["Ingreso base salud"]), _sal=num(df["Salario básico"]),
        _per=df["Periodo cotización"].astype("string").str.slice(0, 7))

    # Tipo dominante por persona = el más frecuente entre sus filas
    tc = d.groupby(["person_id", "_tipo"]).size().reset_index(name="n")
    tipo_dom = (tc.sort_values("n").drop_duplicates("person_id", keep="last")
                .set_index("person_id")["_tipo"])

    g = d.groupby("person_id")
    prof = pd.DataFrame({
        "tipo_persona": tipo_dom,
        "n_meses_ss": g["_per"].nunique(),
        "frac_pension": g["_pension"].mean(), "frac_salud": g["_salud"].mean(),
        "frac_caja": g["_caja"].mean(), "frac_riesgos": g["_riesgos"].mean(),
    })
    prof["ibc_ss"] = d[d["_ibc"] > 0].groupby("person_id")["_ibc"].median()
    prof["salario_ss"] = d[d["_sal"] > 0].groupby("person_id")["_sal"].median()
    prof[["ibc_ss", "salario_ss"]] = prof[["ibc_ss", "salario_ss"]].fillna(0.0)
    prof["prestaciones_completas"] = (
        prof[["frac_pension", "frac_salud", "frac_caja", "frac_riesgos"]] >= 0.5).all(axis=1)
    prof["cotiza_pension"] = prof["frac_pension"] >= 0.5
    return prof.reset_index()


def build_clv_features() -> pd.DataFrame:
    """Modelo de valor / CLV (iteración 2). Llave = cédula → cruza TRF+PSE+SS.
    `recibido`/`enviado` ya EXCLUYEN cuenta propia (ingreso y gasto reales).
    Enriquecido con el ingreso declarado (SS)."""
    trf = build_person_month("trf", solo_ventana_comun=True)
    pse = build_person_month("pse", solo_ventana_comun=True)

    t = trf.groupby("person_id").agg(
        recibido_total=("recibido", "sum"), enviado_total=("enviado", "sum"),
        cuenta_propia_total=("cuenta_propia_total", "sum"),
        n_recibidas=("n_recibidas", "sum"), n_enviadas=("n_enviadas", "sum"),
        recibido_transfiya=("recibido_transfiya", "sum"),
        meses_trf=("periodo", "nunique"), ult_periodo_trf=("periodo", "max"))
    cat_presentes = [c for c in GASTO_CATS if c in pse.columns]
    p = pse.groupby("person_id").agg(
        gasto_total=("gasto_pse", "sum"), n_pagos=("n_pagos", "sum"),
        meses_pse=("periodo", "nunique"), ult_periodo_pse=("periodo", "max"))
    catsum = pse.groupby("person_id")[cat_presentes].sum()

    f = t.join(p, how="outer").join(catsum, how="outer")
    zero = ["recibido_total", "enviado_total", "cuenta_propia_total", "n_recibidas",
            "n_enviadas", "recibido_transfiya", "gasto_total", "n_pagos", *cat_presentes]
    f[zero] = f[zero].fillna(0.0)
    f[["meses_trf", "meses_pse"]] = f[["meses_trf", "meses_pse"]].fillna(0)

    # Enriquecimiento SS (ingreso declarado y tipo de persona) por cédula
    ssp = build_person_ss_profile().set_index("person_id")
    f = f.join(ssp[["ibc_ss", "tipo_persona", "prestaciones_completas", "cotiza_pension"]], how="left")
    f["ibc_ss"] = f["ibc_ss"].fillna(0.0)
    f["tipo_persona"] = f["tipo_persona"].fillna("Sin PILA")
    for col in ["prestaciones_completas", "cotiza_pension"]:
        f[col] = f[col].astype("boolean").fillna(False).astype(bool)

    f["flujo_neto"] = f["recibido_total"] - f["enviado_total"]
    f["throughput"] = f["recibido_total"] + f["enviado_total"] + f["gasto_total"]
    f["meses_activos"] = f[["meses_trf", "meses_pse"]].max(axis=1)
    f["n_transacciones"] = f["n_recibidas"] + f["n_enviadas"] + f["n_pagos"]
    f["ticket_medio"] = f["throughput"] / f["n_transacciones"].replace(0, np.nan)
    f["transfiya_share"] = f["recibido_transfiya"] / f["recibido_total"].replace(0, np.nan)
    f["gasto_financiero_share"] = f.get("gasto_Financiero / créditos", 0) / f["gasto_total"].replace(0, np.nan)
    ult = f[["ult_periodo_trf", "ult_periodo_pse"]].max(axis=1)
    f["recencia_meses"] = _meses_desde(ult).clip(lower=0)
    f["clv_proxy"] = (f["throughput"] / 18.0) * (f["meses_activos"] / 18.0)
    f[["transfiya_share", "gasto_financiero_share", "ticket_medio"]] = \
        f[["transfiya_share", "gasto_financiero_share", "ticket_medio"]].fillna(0.0)
    return f.reset_index()


def build_lifecycle_features() -> pd.DataFrame:
    """Ciclo de vida financiero (iteración 2). Llave = cédula → cruza las 3 fuentes.
    Ancla en SS (tipo de persona, prestaciones, ingreso y su dinámica); enriquece
    con capacidad de ahorro y carga de deuda (ya sin cuenta propia)."""
    ssp = build_person_ss_profile().set_index("person_id")

    ss = build_person_month("ss").sort_values(["person_id", "periodo"])

    def _trend(y: np.ndarray) -> float:
        y = y[y > 0]
        if y.size < 3:
            return 0.0
        return float(np.polyfit(np.arange(y.size), y, 1)[0] / (y.mean() + 1e-9))

    dyn = pd.DataFrame({
        "ibc_volatilidad": (ss[ss["ibc_salud"] > 0].groupby("person_id")["ibc_salud"]
                            .apply(lambda s: s.std() / (s.mean() + 1e-9))),
        "ibc_tendencia": ss.groupby("person_id")["ibc_salud"].apply(lambda s: _trend(s.to_numpy())),
        "intensidad_laboral": ss.groupby("person_id")["dias_salud"].mean(),
    })
    feats = ssp.join(dyn)
    feats["ibc_volatilidad"] = feats["ibc_volatilidad"].fillna(0.0)

    # Enriquecimiento transaccional por cédula (ya externo, sin cuenta propia)
    trf = build_person_month("trf", solo_ventana_comun=True)
    pse = build_person_month("pse", solo_ventana_comun=True)
    flujo = ((trf.groupby("person_id")["recibido"].sum()
              - trf.groupby("person_id")["enviado"].sum()) / 18.0).rename("flujo_neto_mensual")
    fin_col = "gasto_Financiero / créditos"
    if fin_col in pse.columns:
        gp = pse.groupby("person_id")
        deuda = (gp[fin_col].sum() / gp["gasto_pse"].sum().replace(0, np.nan)).rename("gasto_financiero_share")
    else:
        deuda = None
    feats = feats.join(flujo)
    if deuda is not None:
        feats = feats.join(deuda)
    feats["flujo_neto_mensual"] = feats["flujo_neto_mensual"].fillna(0.0)
    if "gasto_financiero_share" in feats:
        feats["gasto_financiero_share"] = feats["gasto_financiero_share"].fillna(0.0)
    feats["tiene_transaccional"] = feats.index.isin(set(trf["person_id"]) | set(pse["person_id"]))
    return feats.reset_index()


# --------------------------------------------------------------------------- #
# Utilidad de validación                                                       #
# --------------------------------------------------------------------------- #
def collision_report(df: pd.DataFrame, fuente: str) -> dict:
    """Compara colisiones de la llave SOLO-documento vs la llave compuesta,
    usando el nombre visible de la fuente como proxy de identidad real."""
    cfg = FUENTES[fuente]
    tmp = person_key_parts(df, fuente)
    tmp["name"] = df[cfg["name"]].astype("string").str.strip().str.upper()
    solo_doc = tmp.groupby("doc")["name"].nunique()
    comp = tmp.groupby(["doc", "name_len", "name_last"])["name"].nunique()
    return {
        "llaves_solo_doc": int(solo_doc.shape[0]),
        "colision_solo_doc_%": round(float((solo_doc > 1).mean()) * 100, 2),
        "llaves_compuestas": int(comp.shape[0]),
        "colision_compuesta_%": round(float((comp > 1).mean()) * 100, 2),
        "max_personas_por_llave_doc": int(solo_doc.max()),
    }
