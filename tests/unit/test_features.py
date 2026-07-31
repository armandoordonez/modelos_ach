"""Tests de la capa de features compartida: llave de persona, taxonomía,
decodificadores y tablas persona-mes."""

from __future__ import annotations

import pandas as pd
import pytest

from common import features as ft
from common.config import Settings


# --------------------------------------------------------------------------- #
# Llave de persona                                                             #
# --------------------------------------------------------------------------- #
def test_llave_cedula_es_el_documento_tal_cual(pse_df):
    llave = ft.llave_persona(pse_df, "pse", "cedula")
    assert llave.iloc[0] == "10000001****"


def test_llave_cedula_cruza_las_tres_fuentes(ss_df, trf_df, pse_df):
    """La cédula llega enmascarada igual en las tres: es lo que permite el cruce."""
    en_ss = set(ft.llave_persona(ss_df, "ss", "cedula"))
    en_trf = set(ft.llave_persona(trf_df, "trf", "cedula"))
    en_pse = set(ft.llave_persona(pse_df, "pse", "cedula"))
    assert "10000001****" in en_ss & en_trf & en_pse


def test_llave_legacy_del_caso02_compone_nombre_y_documento(pse_df):
    llave = ft.llave_persona(pse_df, "pse", "nombre_documento")
    assert llave.iloc[0] == "***ANA PEREZ|10000001****"


def test_llave_legacy_del_caso04_normaliza_y_quita_asteriscos(pse_df):
    llave = ft.llave_persona(pse_df, "pse", "nombre_normalizado_documento_visible")
    assert llave.iloc[0] == "ANA PEREZ|10000001"


def test_las_llaves_legacy_no_cruzan_entre_si(pse_df):
    """Justifica la unificación: dos scripts con llaves distintas no comparten universo."""
    caso02 = set(ft.llave_persona(pse_df, "pse", "nombre_documento"))
    caso04 = set(ft.llave_persona(pse_df, "pse", "nombre_normalizado_documento_visible"))
    assert caso02.isdisjoint(caso04)


def test_estrategia_desde_config_respeta_el_linaje():
    produccion = Settings(_env_file=None, lineage="cedula-v1")
    legado = Settings(_env_file=None, lineage="legacy")
    assert ft.estrategia_desde_config("nombre_documento", produccion) == "cedula"
    assert ft.estrategia_desde_config("nombre_documento", legado) == "nombre_documento"
    assert ft.estrategia_desde_config(None, legado) == "cedula"


def test_estrategia_desconocida_falla(pse_df):
    with pytest.raises(ValueError, match="Estrategia de llave desconocida"):
        ft.llave_persona(pse_df, "pse", "telepatia")


# --------------------------------------------------------------------------- #
# Taxonomía de comercios                                                       #
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("comercio,categoria", [
    ("BANCOLOM********", "Financiero / créditos"),
    ("CLARO SOLUCIONES********", "Telco / servicios públicos"),
    ("ALMACENES EXITO-PRO********", "Comercio electrónico / retail"),
    ("COMPENSAR EPS********", "Seguridad social / nómina"),
    ("UN COMERCIO QUE NO EXISTE********", "Otros / no clasificado"),
    # La ofuscación corta el nombre antes de la marca: queda sin clasificar. Es la
    # limitación ya documentada del 39,5% del valor PSE sin categoría.
    ("CAJA COLOMBIANA DE SUBSIDIO FAMILIAR COL********", "Otros / no clasificado"),
])
def test_categorizacion_de_comercios(comercio, categoria):
    serie = pd.Series([comercio], dtype="string")
    assert ft.categorizar_comercio(serie).iloc[0] == categoria


def test_la_taxonomia_es_unica_para_todos_los_modelos():
    """Los scripts originales traían tres taxonomías distintas; ahora hay una sola."""
    assert len(ft.CATEGORIAS_COMERCIO) == 12
    assert ft.CATEGORIA_POR_DEFECTO not in ft.CATEGORIAS_COMERCIO


def test_categorizar_tolera_nulos():
    serie = pd.Series([None, "BANCOLOM********"], dtype="string")
    resultado = ft.categorizar_comercio(serie)
    assert resultado.iloc[0] == ft.CATEGORIA_POR_DEFECTO


# --------------------------------------------------------------------------- #
# Decodificadores                                                              #
# --------------------------------------------------------------------------- #
def test_decodifica_banco_desde_el_prefijo():
    serie = pd.Series(["BANCO DAVIV********", "*****", "BAN********", "NEQUI***"], dtype="string")
    salida = ft.decodificar_entidad_autorizadora(serie)
    assert salida.loc[0, "entidad"] == "DAVIVIENDA"
    assert salida.loc[0, "entidad_confianza"] == "alta"
    assert salida.loc[1, "entidad"] == "DESCONOCIDO", "enmascarada por completo, no se adivina"
    assert salida.loc[2, "entidad_confianza"] == "media", "prefijo ambiguo con override documentado"
    assert salida.loc[3, "entidad"] == "NEQUI"


def test_la_longitud_desambigua_compensar_de_cafam():
    """La ofuscación tapa los últimos 8 caracteres, así que la longitud real es
    recuperable y separa dos cajas que por prefijo serían idénticas."""
    serie = pd.Series([
        "CAJA DE COMPENSACION FAMILIAR C********",   # 31 + 8 = 39 -> COMPENSAR
        "CAJA DE COMPENSACIÓN FAMILI********",       # 27 + 8 = 35 -> CAFAM
    ], dtype="string")
    salida = ft.decodificar_entidad_objetivo(serie)
    assert salida.loc[0, "entidad_objetivo"] == "Compensar"
    assert salida.loc[0, "objetivo_confianza"] == "alta"
    assert salida.loc[1, "entidad_objetivo"] == "Cafam"


def test_comercio_fuera_del_catalogo_no_es_objetivo():
    serie = pd.Series(["PANADERIA DONA ROSA********"], dtype="string")
    assert ft.decodificar_entidad_objetivo(serie).loc[0, "entidad_objetivo"] == "No objetivo"


def test_todas_las_entidades_objetivo_tienen_grupo():
    assert set(ft.CATALOGO_OBJETIVO) == set(ft.GRUPO_OBJETIVO)


# --------------------------------------------------------------------------- #
# Tablas persona-mes                                                           #
# --------------------------------------------------------------------------- #
def test_transferencias_excluyen_cuenta_propia(trf_df):
    df = trf_df.assign(person_id=ft.llave_persona(trf_df, "trf", "cedula"),
                       periodo=trf_df["Periodo"])
    tabla = ft.persona_mes_transferencias(df)
    fila = tabla.iloc[0]
    assert fila["recibido"] == 1_200_000.0, "solo los recibidos externos"
    assert fila["enviado"] == 300_000.0
    assert fila["cuenta_propia_total"] == 5_000_000.0, "el movimiento interno se guarda aparte"
    assert fila["recibido_transfiya"] == 200_000.0


def test_pagos_agregan_por_categoria(pse_df):
    df = pse_df.assign(person_id=ft.llave_persona(pse_df, "pse", "cedula"),
                       periodo=pse_df["Periodo"])
    tabla = ft.persona_mes_pagos(df)
    enero = tabla[tabla["periodo"] == "2025-01"].iloc[0]
    assert enero["gasto_pse"] == 395_000.0
    assert enero["n_comercios"] == 3
    assert enero["gasto_Financiero / créditos"] == 250_000.0


def test_seguridad_social_toma_el_maximo_ibc_del_mes(ss_df):
    df = ss_df.assign(person_id=ft.llave_persona(ss_df, "ss", "cedula"),
                      periodo=ss_df["Periodo cotización"])
    tabla = ft.persona_mes_seguridad_social(df)
    assert set(tabla["ibc_salud"]) == {1_500_000.0, 900_000.0}


def test_meses_hasta_calcula_recencia():
    periodos = pd.Series(["2026-06", "2026-04", "2025-12"])
    assert list(ft.meses_hasta(periodos, "2026-06")) == [0, 2, 6]


# --------------------------------------------------------------------------- #
# Lectura del dataset curado                                                   #
# --------------------------------------------------------------------------- #
def test_cargar_fuente_agrega_llave_y_periodo(curated):
    df = ft.cargar_fuente("pse", storage=curated)
    assert "person_id" in df.columns and "periodo" in df.columns
    assert df["person_id"].iloc[0] == "10000001****"


def test_cargar_fuente_filtra_por_ventana(curated):
    ajustes = curated.settings.model_copy(update={"window_start": "2025-02", "window_end": "2025-02"})
    df = ft.cargar_fuente("pse", storage=curated, settings=ajustes)
    assert set(df["periodo"]) == {"2025-02"}


def test_fuente_sin_filas_falla_con_mensaje_claro(curated):
    ajustes = curated.settings.model_copy(update={"window_start": "2030-01", "window_end": "2030-12"})
    df = ft.cargar_fuente("pse", storage=curated, settings=ajustes)
    assert df.empty


def test_tabla_persona_mes_desde_curado(curated):
    tabla = ft.tabla_persona_mes("trf", storage=curated)
    assert len(tabla) == 1
    assert tabla["recibido"].iloc[0] == 1_200_000.0


def test_perfil_ss_clasifica_pensionado_y_empleado(curated):
    perfil = ft.perfil_persona_ss(storage=curated).set_index("person_id")
    assert perfil.loc["10000001****", "tipo_persona"] == "Empleado"
    assert perfil.loc["10000002****", "tipo_persona"] == "Pensionado"
    assert perfil.loc["10000002****", "ibc_ss"] == 900_000.0
    assert bool(perfil.loc["10000001****", "cotiza_pension"]) is True
