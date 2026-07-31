"""Capa de ingeniería de datos compartida por todos los modelos.

Es la migración de ``scripts/ach_pipeline.py`` del proyecto exploratorio, con dos
cambios importantes:

1. **Lee del dataset curado**, no de los XLSX ni de parquets sueltos en disco.
2. **La llave de persona es un parámetro**, no una decisión enterrada en cada script.
   Los tres orígenes migrados traían tres llaves distintas; producción usa siempre
   la cédula ofuscada (``cedula``) y las otras dos quedan disponibles solo para
   verificar que la migración reproduce las métricas originales.

Principio metodológico heredado: solo se usa lo que es realmente visible en el dato.
No se imputa ningún valor enmascarado por conjetura.
"""

from __future__ import annotations

import logging
import re
import unicodedata
from typing import Literal

import numpy as np
import pandas as pd

from .config import Settings, get_settings
from .schema import COLUMNA_NOMBRE, COLUMNA_PERIODO, COLUMNA_PERSONA
from .storage import Storage, get_storage

log = logging.getLogger(__name__)

EstrategiaLlave = Literal["cedula", "nombre_documento", "nombre_normalizado_documento_visible"]

RUTA_CURADO = "dataset.parquet"
ARCHIVO_MANIFIESTO = "_manifest.json"


# --------------------------------------------------------------------------- #
# 1) Llave de persona                                                          #
# --------------------------------------------------------------------------- #
def _normalizar_texto(serie: pd.Series) -> pd.Series:
    """Mayúsculas sin tildes ni signos. Reproduce la normalización del Caso 04."""

    def limpiar(valor) -> str | None:
        if pd.isna(valor):
            return None
        texto = str(valor).strip().upper()
        texto = "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")
        texto = re.sub(r"[^A-Z0-9 ]+", " ", texto)
        texto = re.sub(r"\s+", " ", texto).strip()
        return texto or None

    return serie.map(limpiar).astype("string")


def llave_persona(df: pd.DataFrame, fuente: str, estrategia: EstrategiaLlave = "cedula") -> pd.Series:
    """Devuelve la llave de persona de cada fila según la estrategia pedida.

    - ``cedula``: el documento ofuscado tal cual llega. Es idéntico en las tres
      fuentes, así que cruza las tres de forma directa. **Estrategia de producción.**
    - ``nombre_documento``: ``Nombre|Documento`` — la del script original del Caso 02.
    - ``nombre_normalizado_documento_visible``: nombre normalizado + documento sin
      asteriscos — la del script original del Caso 04.
    """
    doc = df[COLUMNA_PERSONA[fuente]].astype("string").str.strip()

    if estrategia == "cedula":
        return doc.rename("person_id")

    nombre = df[COLUMNA_NOMBRE[fuente]].astype("string").str.strip()
    if estrategia == "nombre_documento":
        return (nombre.fillna("") + "|" + doc.fillna("")).astype("string").rename("person_id")

    if estrategia == "nombre_normalizado_documento_visible":
        nombre_norm = _normalizar_texto(nombre)
        doc_visible = doc.str.replace("*", "", regex=False)
        llave = (nombre_norm.fillna("") + "|" + doc_visible.fillna("")).astype("string")
        invalida = nombre_norm.isna() | (llave.str.strip("|") == "")
        return llave.mask(invalida).rename("person_id")

    raise ValueError(f"Estrategia de llave desconocida: {estrategia!r}")


def estrategia_desde_config(legacy: EstrategiaLlave | None, settings: Settings | None = None) -> EstrategiaLlave:
    """Resuelve qué llave usar: la de producción o la legacy del modelo."""
    settings = settings or get_settings()
    if settings.lineage == "cedula-v1":
        return "cedula"
    if legacy is None:
        return "cedula"
    return legacy


def num(serie: pd.Series) -> pd.Series:
    """Serie numérica float64 robusta, tolerante a texto ofuscado y nulos."""
    return pd.to_numeric(serie, errors="coerce").astype("float64")


# --------------------------------------------------------------------------- #
# 2) Taxonomía única de comercios PSE                                          #
# --------------------------------------------------------------------------- #
# IMPORTANTE: esta taxonomía es idéntica, palabra por palabra, a la del
# ach_pipeline exploratorio. Los modelos del Caso 05 se validan por paridad exacta
# contra sus notebooks, así que ampliarla cambiaría sus métricas. Mejorarla es una
# decisión aparte, con su propio re-baseline documentado.
CATEGORIAS_COMERCIO: dict[str, list[str]] = {
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
CATEGORIA_POR_DEFECTO = "Otros / no clasificado"


def _categorizar_uno(nombre: str) -> str:
    texto = str(nombre).upper()
    for categoria, claves in CATEGORIAS_COMERCIO.items():
        if any(clave in texto for clave in claves):
            return categoria
    return CATEGORIA_POR_DEFECTO


def categorizar_comercio(serie: pd.Series) -> pd.Series:
    """Mapea cada comercio ofuscado a una categoría temática, vía valores únicos."""
    unicos = serie.dropna().astype("string").unique()
    mapa = {c: _categorizar_uno(c) for c in unicos}
    return serie.astype("string").map(mapa).fillna(CATEGORIA_POR_DEFECTO)


# --------------------------------------------------------------------------- #
# 3) Decodificadores de entidades ofuscadas                                    #
# --------------------------------------------------------------------------- #
# La ofuscación reemplaza los ÚLTIMOS 8 caracteres del nombre por asteriscos, así que
# la longitud real del nombre es recuperable: len(visible) + 8. Eso permite exigir
# prefijo Y longitud, que desambigua casos que el prefijo solo dejaría abiertos
# (Compensar vs. Cafam, ambos "CAJA DE COMPENSACION FAMILIAR C...").
N_MASCARA = 8

CATALOGO_ENTIDADES = [
    "BANCOLOMBIA", "BANCO DE BOGOTA", "BANCO DAVIVIENDA", "BANCO BBVA COLOMBIA",
    "BANCO CAJA SOCIAL", "BANCO POPULAR", "BANCO AGRARIO DE COLOMBIA", "BANCO DE OCCIDENTE",
    "BANCO AV VILLAS", "SCOTIABANK COLPATRIA", "BANCO GNB SUDAMERIS", "BANCO FALABELLA",
    "BANCO PICHINCHA", "BANCO ITAU", "BANCOOMEVA", "CITIBANK", "BANCO COOPERATIVO COOPCENTRAL",
    "BANCO FINANDINA", "BANCO MUNDO MUJER", "BANCO W", "BANCO SERFINANZA", "BANCAMIA",
    "BANCO SANTANDER", "LULO BANK", "NEQUI", "DAVIPLATA", "NU", "COLTEFINANCIERA", "RAPPIPAY",
]
_ETIQUETA = {c: c for c in CATALOGO_ENTIDADES}
_ETIQUETA["BANCO DAVIVIENDA"] = "DAVIVIENDA"
_ETIQUETA["BANCO BBVA COLOMBIA"] = "BBVA COLOMBIA"
_ETIQUETA["BANCO COOPERATIVO COOPCENTRAL"] = "COOPCENTRAL"
_OVERRIDES = {"BAN": ("BANCOLOMBIA", "media")}


def _decodificar_entidad(valor) -> tuple[str, str]:
    if valor is None or (isinstance(valor, float) and np.isnan(valor)):
        return ("DESCONOCIDO", "nula")
    texto = str(valor).upper().strip()
    if texto == "" or texto.startswith("*"):
        return ("DESCONOCIDO", "nula")
    prefijo = re.sub(r"\*+$", "", texto).strip()
    if prefijo == "":
        return ("DESCONOCIDO", "nula")
    if prefijo in _OVERRIDES:
        return _OVERRIDES[prefijo]
    candidatos = {_ETIQUETA[c] for c in CATALOGO_ENTIDADES if c.startswith(prefijo)}
    if len(candidatos) == 1:
        return (next(iter(candidatos)), "alta" if len(prefijo) >= 5 else "media")
    if not candidatos:
        return (prefijo, "sin_catalogo")
    return ("AMBIGUO", "baja")


def decodificar_entidad_autorizadora(serie: pd.Series) -> pd.DataFrame:
    """Recupera el banco desde el prefijo visible de ``Entidad autorizadora``."""
    unicos = serie.astype("string")
    tabla = {v: _decodificar_entidad(v) for v in unicos.dropna().unique()}
    etiqueta = unicos.map(lambda v: tabla.get(v, ("DESCONOCIDO", "nula"))[0])
    confianza = unicos.map(lambda v: tabla.get(v, ("DESCONOCIDO", "nula"))[1])
    return pd.DataFrame({"entidad": etiqueta, "entidad_confianza": confianza}, index=serie.index)


CATALOGO_OBJETIVO: dict[str, list[str]] = {
    "Grupo Éxito": ["ALMACENES EXITO", "EXITO VIAJES"],
    "Cencosud Colombia": ["CENCOSUD", "JUMBO", "TIENDAS METRO"],
    "Colsubsidio": ["CAJA COLOMBIANA DE SUBSIDIO FAMILIAR COLSUBSIDIO"],
    "Compensar": ["CAJA DE COMPENSACION FAMILIAR COMPENSAR"],
    "Cafam": ["CAJA DE COMPENSACIÓN FAMILIAR CAFAM", "CAJA DE COMPENSACION FAMILIAR CAFAM"],
    "Bancolombia": ["BANCOLOMBIA"],
    "Davivienda": ["BANCO DAVIVIENDA"],
    "Colpensiones": ["ADMINISTRADORA COLOMBIANA DE PENSIONES COLPENSIONES"],
    "Porvenir": ["PORVENIR"],
}
GRUPO_OBJETIVO: dict[str, str] = {
    "Grupo Éxito": "Retail", "Cencosud Colombia": "Retail",
    "Colsubsidio": "Caja de compensación", "Compensar": "Caja de compensación",
    "Cafam": "Caja de compensación",
    "Bancolombia": "Banca", "Davivienda": "Banca",
    "Colpensiones": "Pensiones", "Porvenir": "Pensiones",
}


def _sin_tildes(texto: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFD", texto) if unicodedata.category(c) != "Mn")


def _decodificar_objetivo(valor) -> tuple[str, str]:
    """Dos pasadas: primero se busca una coincidencia de prefijo Y longitud (que es
    inequívoca), y solo si ninguna encaja se acepta una de prefijo. Sin ese orden,
    'CAJA DE COMPENSACIÓN FAMILI********' (Cafam) se llevaría la etiqueta de
    Compensar, que aparece antes en el catálogo y también encaja por prefijo."""
    texto = str(valor).upper().strip()
    visible = texto.rstrip("*").strip()
    if not visible:
        return ("No objetivo", "nula")
    largo_real = len(texto.rstrip("*")) + N_MASCARA if texto.endswith("*") else len(texto)
    v = _sin_tildes(visible)

    for entidad, canonicos in CATALOGO_OBJETIVO.items():
        for canon in canonicos:
            c = _sin_tildes(canon.upper())
            if c.startswith(v) and len(c) == largo_real:
                return (entidad, "alta")

    for entidad, canonicos in CATALOGO_OBJETIVO.items():
        for canon in canonicos:
            c = _sin_tildes(canon.upper())
            if v.startswith(c) or (c.startswith(v) and len(v) >= 10):
                return (entidad, "media")
    return ("No objetivo", "nula")


def decodificar_entidad_objetivo(serie: pd.Series) -> pd.DataFrame:
    """Identifica comercios de las entidades objetivo por prefijo y longitud."""
    unicos = serie.dropna().astype("string").unique()
    tabla = {c: _decodificar_objetivo(c) for c in unicos}
    entidad = serie.map(lambda c: tabla.get(c, ("No objetivo", "nula"))[0])
    confianza = serie.map(lambda c: tabla.get(c, ("No objetivo", "nula"))[1])
    return pd.DataFrame({"entidad_objetivo": entidad, "objetivo_confianza": confianza}, index=serie.index)


# --------------------------------------------------------------------------- #
# 4) Lectura del dataset curado                                                #
# --------------------------------------------------------------------------- #
def uri_curado(storage: Storage | None = None) -> str:
    storage = storage or get_storage()
    return storage.ruta(storage.settings.bucket_curated, RUTA_CURADO)


def leer_manifiesto(storage: Storage | None = None) -> dict:
    """Lee el ``_manifest.json`` del dataset curado. Devuelve {} si no existe."""
    storage = storage or get_storage()
    ruta = storage.ruta(storage.settings.bucket_curated, RUTA_CURADO, ARCHIVO_MANIFIESTO)
    if not storage.existe(ruta):
        log.warning("No hay manifiesto en %s", ruta)
        return {}
    return storage.leer_json(ruta)


def cargar_fuente(
    fuente: str,
    columnas: list[str] | None = None,
    storage: Storage | None = None,
    estrategia: EstrategiaLlave = "cedula",
    solo_ventana: bool = True,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Carga una fuente del dataset curado con ``person_id`` y ``periodo`` listos.

    Las filas duplicadas exactas ya vienen eliminadas por el job de procesamiento.
    """
    storage = storage or get_storage()
    settings = settings or storage.settings

    # Se lee la partición de la fuente, no la raíz del dataset: las tres fuentes tienen
    # esquemas distintos (31, 26 y 12 columnas) y pyarrow unificaría el esquema con el
    # del primer fragmento que encuentre, perdiendo columnas.
    ruta = f"{uri_curado(storage)}/fuente={fuente}"
    if not storage.existe(ruta):
        raise FileNotFoundError(
            f"No existe la partición {ruta}. ¿Corrió el job de procesamiento para la fuente {fuente!r}?"
        )

    necesarias = None
    if columnas is not None:
        necesarias = sorted({*columnas, COLUMNA_PERSONA[fuente], COLUMNA_PERIODO[fuente]})

    df = storage.leer_parquet(ruta, columnas=necesarias)
    if df.empty:
        raise ValueError(
            f"El dataset curado no tiene filas de la fuente {fuente!r} en {ruta}. "
            "¿Corrió el job de procesamiento?"
        )

    df["person_id"] = llave_persona(df, fuente, estrategia)
    df["periodo"] = df[COLUMNA_PERIODO[fuente]].astype("string").str.slice(0, 7)
    if solo_ventana:
        inicio, fin = settings.ventana
        dentro = df["periodo"].between(inicio, fin).fillna(False).to_numpy(dtype=bool)
        df = df.loc[dentro]
    return df.reset_index(drop=True)


# --------------------------------------------------------------------------- #
# 5) Tablas analíticas persona × mes                                           #
# --------------------------------------------------------------------------- #
def persona_mes_transferencias(df: pd.DataFrame) -> pd.DataFrame:
    """Flujos reales por persona y mes. Excluye cuenta propia del ingreso y del gasto:
    mover dinero entre cuentas propias no es ni ingreso ni gasto."""
    valor, cantidad = num(df["Valor"]), num(df["Cantidad de transferencias"])
    enviado = df["Tipo registro"].astype("string").str.upper().eq("ENVIADO")
    servicio = df["Servicio"].astype("string").str.upper()
    propio = df["Cuenta propia"].astype("string").str.upper().eq("SI")
    externo = ~propio.fillna(False)
    enviado = enviado.fillna(False)

    base = pd.DataFrame({
        "person_id": df["person_id"], "periodo": df["periodo"],
        "recibido": valor.where(~enviado & externo, 0.0),
        "enviado": valor.where(enviado & externo, 0.0),
        "cuenta_propia_total": valor.where(propio.fillna(False), 0.0),
        "n_recibidas": cantidad.where(~enviado & externo, 0.0),
        "n_enviadas": cantidad.where(enviado & externo, 0.0),
        "recibido_ach": valor.where(~enviado & externo & servicio.eq("ACH"), 0.0),
        "recibido_transfiya": valor.where(~enviado & externo & servicio.eq("TRANSFIYA"), 0.0),
    })
    return base.groupby(["person_id", "periodo"], as_index=False).sum()


def persona_mes_pagos(df: pd.DataFrame) -> pd.DataFrame:
    """Gasto PSE por persona, mes y categoría de comercio."""
    valor, cantidad = num(df["Valor"]), num(df["Cantidad"])
    categoria = categorizar_comercio(df["Comercio"])
    base = pd.DataFrame({
        "person_id": df["person_id"], "periodo": df["periodo"],
        "gasto": valor, "n_pagos": cantidad, "categoria": categoria,
        "comercio": df["Comercio"].astype("string"),
    })
    agregado = base.groupby(["person_id", "periodo"], as_index=False).agg(
        gasto_pse=("gasto", "sum"), n_pagos=("n_pagos", "sum"), n_comercios=("comercio", "nunique"))
    pivote = (base.pivot_table(index=["person_id", "periodo"], columns="categoria", values="gasto",
                               aggfunc="sum", fill_value=0.0)
              .add_prefix("gasto_").reset_index())
    return agregado.merge(pivote, on=["person_id", "periodo"], how="left")


def persona_mes_seguridad_social(df: pd.DataFrame) -> pd.DataFrame:
    """Ingreso declarado y días cotizados por persona y mes."""
    base = pd.DataFrame({
        "person_id": df["person_id"], "periodo": df["periodo"],
        "ibc_salud": num(df["Ingreso base salud"]),
        "ibc_pension": num(df["Ingreso base pensión"]),
        "salario_basico": num(df["Salario básico"]),
        "dias_pension": num(df["Días pensión"]),
        "dias_salud": num(df["Días salud"]),
    })
    return base.groupby(["person_id", "periodo"], as_index=False).agg(
        ibc_salud=("ibc_salud", "max"), ibc_pension=("ibc_pension", "max"),
        salario_basico=("salario_basico", "max"),
        dias_pension=("dias_pension", "sum"), dias_salud=("dias_salud", "sum"))


_CONSTRUCTORES = {
    "trf": persona_mes_transferencias,
    "pse": persona_mes_pagos,
    "ss": persona_mes_seguridad_social,
}


def tabla_persona_mes(
    fuente: str,
    storage: Storage | None = None,
    estrategia: EstrategiaLlave = "cedula",
    solo_ventana: bool = True,
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Tabla persona × mes de una fuente, leída del dataset curado."""
    if fuente not in _CONSTRUCTORES:
        raise ValueError(f"Fuente desconocida: {fuente!r}")
    df = cargar_fuente(fuente, storage=storage, estrategia=estrategia,
                       solo_ventana=solo_ventana, settings=settings)
    tabla = _CONSTRUCTORES[fuente](df)
    log.info("Tabla persona-mes %s: %s filas, %s personas",
             fuente, f"{len(tabla):,}", f"{tabla['person_id'].nunique():,}")
    return tabla


# --------------------------------------------------------------------------- #
# 6) Perfil de persona desde Seguridad Social                                  #
# --------------------------------------------------------------------------- #
def perfil_persona_ss(
    storage: Storage | None = None,
    estrategia: EstrategiaLlave = "cedula",
    settings: Settings | None = None,
) -> pd.DataFrame:
    """Clasificación de cada persona según PILA: tipo, cobertura de prestaciones,
    ingreso declarado. Alimenta a los tres modelos del Caso 05."""
    df = cargar_fuente("ss", storage=storage, estrategia=estrategia,
                       solo_ventana=False, settings=settings)

    relacion = df["Relación laboral"].astype("string").str.upper()
    aportante = df["Tipo aportante"].astype("string").str.upper()
    planilla = df["Tipo planilla"].astype("string").str.upper()

    tipo = pd.Series("Otro", index=df.index, dtype="object")
    tipo[relacion.eq("DEPENDIENTE") | aportante.eq("EMPLEADOR") | planilla.eq("E")] = "Empleado"
    tipo[relacion.str.startswith("INDEPENDIENTE", na=False) | aportante.eq("INDEPENDIENTE")
         | planilla.isin(["I", "Y"])] = "Independiente"
    tipo[relacion.str.contains("APRENDIC", na=False)] = "Aprendiz"
    tipo[relacion.str.contains("MAGISTERIO", na=False)
         | aportante.str.contains("MAGISTERIO", na=False)] = "Magisterio"
    tipo[relacion.str.contains("PENSIONADO", na=False) | planilla.eq("P")
         | aportante.str.contains("PENSIONES", na=False)] = "Pensionado"

    d = df.assign(
        _tipo=tipo,
        _pension=num(df["Días pensión"]) > 0, _salud=num(df["Días salud"]) > 0,
        _caja=num(df["Días caja compensación"]) > 0, _riesgos=num(df["Días riesgos"]) > 0,
        _ibc=num(df["Ingreso base salud"]), _sal=num(df["Salario básico"]),
    )

    conteo = d.groupby(["person_id", "_tipo"]).size().reset_index(name="n")
    tipo_dominante = (conteo.sort_values("n").drop_duplicates("person_id", keep="last")
                      .set_index("person_id")["_tipo"])

    g = d.groupby("person_id")
    perfil = pd.DataFrame({
        "tipo_persona": tipo_dominante,
        "n_meses_ss": g["periodo"].nunique(),
        "frac_pension": g["_pension"].mean(), "frac_salud": g["_salud"].mean(),
        "frac_caja": g["_caja"].mean(), "frac_riesgos": g["_riesgos"].mean(),
    })
    perfil["ibc_ss"] = d[d["_ibc"] > 0].groupby("person_id")["_ibc"].median()
    perfil["salario_ss"] = d[d["_sal"] > 0].groupby("person_id")["_sal"].median()
    perfil[["ibc_ss", "salario_ss"]] = perfil[["ibc_ss", "salario_ss"]].fillna(0.0)
    perfil["prestaciones_completas"] = (
        perfil[["frac_pension", "frac_salud", "frac_caja", "frac_riesgos"]] >= 0.5).all(axis=1)
    perfil["cotiza_pension"] = perfil["frac_pension"] >= 0.5
    return perfil.reset_index()


def meses_hasta(periodos: pd.Series, fin: str) -> pd.Series:
    """Meses entre cada periodo 'YYYY-MM' y ``fin``. Base de la recencia."""
    indice = pd.PeriodIndex(periodos.astype("string"), freq="M")
    limite = pd.Period(fin, freq="M")
    return pd.Series([(limite - p).n for p in indice], index=periodos.index)
