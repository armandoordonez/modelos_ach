"""Tests del diccionario de datos y sus validaciones."""

from __future__ import annotations

import pandas as pd
import pytest

from common import schema


@pytest.mark.parametrize("fuente,n_columnas", [("ss", 31), ("trf", 26), ("pse", 12)])
def test_el_diccionario_coincide_con_el_extracto_real(fuente, n_columnas):
    assert len(schema.nombres_columnas(fuente)) == n_columnas


def test_fuente_desconocida_falla_con_mensaje_util():
    with pytest.raises(schema.ErrorDeEsquema, match="Fuente desconocida"):
        schema.columnas("inventada")


def test_los_roles_de_persona_y_periodo_son_explicitos():
    # Este es el defecto que se corrige: en Seguridad Social el periodo de análisis
    # es 'Periodo cotización', NO 'Fecha de pago'.
    assert schema.COLUMNA_PERIODO["ss"] == "Periodo cotización"
    assert schema.COLUMNA_PERSONA["ss"] == "Número de documento"
    assert schema.COLUMNA_PERSONA["trf"] == "Número documento"
    assert "Fecha de pago" in schema.nombres_columnas("ss")


def test_llave_de_persona_no_es_nullable_en_ninguna_fuente():
    for fuente in ("ss", "trf", "pse"):
        columna = next(c for c in schema.columnas(fuente) if c.nombre == schema.COLUMNA_PERSONA[fuente])
        assert columna.nullable is False


def test_estructura_valida_no_lanza(ss_df):
    schema.validar_estructura(ss_df, "ss")


def test_falta_de_columna_falla_temprano_y_la_nombra(ss_df):
    incompleto = ss_df.drop(columns=["Ingreso base salud"])
    with pytest.raises(schema.ErrorDeEsquema) as error:
        schema.validar_estructura(incompleto, "ss")
    assert "Ingreso base salud" in str(error.value)
    assert "Seguridad Social" in str(error.value)


def test_columnas_de_mas_se_reportan_pero_no_rompen(ss_df):
    con_extra = ss_df.assign(columna_rara=1)
    schema.validar_estructura(con_extra, "ss")  # no falla: strict=False

    faltante = con_extra.drop(columns=["Días salud"])
    with pytest.raises(schema.ErrorDeEsquema) as error:
        schema.validar_estructura(faltante, "ss")
    assert "columna_rara" in str(error.value)


def test_aplicar_tipos_convierte_segun_el_diccionario(pse_df):
    crudo = pse_df.astype(str)
    tipado = schema.aplicar_tipos(crudo, "pse")
    assert str(tipado["Valor"].dtype) == "float64"
    assert str(tipado["Cantidad"].dtype) == "Int64"
    assert str(tipado["Comercio"].dtype) == "string"


def test_validar_devuelve_el_dataframe_tipado(trf_df):
    tipado = schema.validar(trf_df, "trf")
    assert str(tipado["Valor"].dtype) == "float64"
    assert len(tipado) == len(trf_df)


def test_periodo_con_formato_invalido_falla(pse_df):
    malo = pse_df.copy()
    malo.loc[0, "Periodo"] = "enero-2025"
    with pytest.raises(schema.ErrorDeEsquema, match="no cumple el diccionario"):
        schema.validar(malo, "pse")


def test_monto_negativo_falla(pse_df):
    malo = pse_df.copy()
    malo.loc[0, "Valor"] = -1.0
    with pytest.raises(schema.ErrorDeEsquema) as error:
        schema.validar(malo, "pse")
    assert "Valor" in str(error.value)


def test_esquema_arrow_tiene_todas_las_columnas():
    for fuente in ("ss", "trf", "pse"):
        esquema = schema.esquema_arrow(fuente)
        assert esquema.names == schema.nombres_columnas(fuente)


def test_hay_un_archivo_declarado_por_fuente():
    assert set(schema.ARCHIVO_FUENTE) == set(schema.DICCIONARIO)
    assert all(nombre.endswith(".xlsx") for nombre in schema.ARCHIVO_FUENTE.values())
