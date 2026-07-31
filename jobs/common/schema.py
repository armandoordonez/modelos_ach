"""Diccionario de datos: única fuente de verdad de columnas, tipos y roles.

Sustituye la detección de columnas por expresiones regulares que traían algunos de
los scripts originales. Ahí, por ejemplo, el patrón ``fecha`` enganchaba
``Fecha de pago`` antes que ``Periodo cotización`` en Seguridad Social, que son cosas
distintas (cuándo pagó el aportante vs. qué mes se cotizó). Aquí cada rol está
declarado explícitamente y no se adivina nada.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import pandas as pd
import pandera.pandas as pa
import pyarrow as pyarrow_types

Rol = Literal["identidad", "periodo", "monto", "conteo", "categoria", "texto", "agregado"]
Tipo = Literal["string", "int64", "float64"]


class ErrorDeEsquema(ValueError):
    """Se lanza cuando un archivo de entrada no cumple el diccionario de datos."""


@dataclass(frozen=True)
class Columna:
    nombre: str
    tipo: Tipo
    rol: Rol
    nullable: bool = False
    descripcion: str = ""


# --------------------------------------------------------------------------- #
# Diccionario de datos por fuente                                              #
# --------------------------------------------------------------------------- #
SEGURIDAD_SOCIAL: tuple[Columna, ...] = (
    Columna("Nombre", "string", "identidad", True, "Nombre ofuscado de la persona"),
    Columna("Tipo de documento", "string", "identidad", True, "Tipo de documento de la persona"),
    Columna("Número de documento", "string", "identidad", False, "Cédula ofuscada — LLAVE DE PERSONA"),
    Columna("Promedio salario básico", "int64", "agregado", True, "Agregado pre-calculado por persona"),
    Columna("Promedio ingreso base pensión", "int64", "agregado", True, ""),
    Columna("Promedio ingreso base salud", "int64", "agregado", True, ""),
    Columna("Promedio ingreso base caja compensación", "int64", "agregado", True, ""),
    Columna("Promedio ingreso base riesgos", "int64", "agregado", True, ""),
    Columna("Razón social", "string", "texto", True, "Aportante: empleador o pagador de pensión"),
    Columna("Tipo documento", "string", "identidad", True, "Tipo de documento del aportante"),
    Columna("Número documento", "string", "identidad", True, "Documento del aportante (NO de la persona)"),
    Columna("Código actividad económica", "int64", "categoria", True, "CIIU del aportante"),
    Columna("Actividad económica", "string", "categoria", True, ""),
    Columna("Clase aportante", "string", "categoria", True, ""),
    Columna("Tipo aportante", "string", "categoria", True, "Marca de pagador/administradora de pensiones"),
    Columna("Fecha de pago", "string", "texto", True, "Fecha en que el aportante pagó — NO es el periodo"),
    Columna("Periodo cotización", "string", "periodo", False, "Mes cotizado YYYY-MM — PERIODO DE ANÁLISIS"),
    Columna("Empleador", "string", "texto", True, ""),
    Columna("Tipo planilla", "string", "categoria", True, "Tipo PILA; 'P' identifica planilla de pensionados"),
    Columna("Relación laboral", "string", "categoria", True, "Dependiente/independiente/pensionado/aprendiz"),
    Columna("Salario básico", "int64", "monto", True, ""),
    Columna("Tipo salario", "string", "categoria", True, ""),
    Columna("Novedades", "string", "texto", True, "Multivaluado separado por comas; nulo = mes sin novedad"),
    Columna("Ingreso base pensión", "int64", "monto", True, ""),
    Columna("Días pensión", "int64", "conteo", True, "0 con salud activa = señal de informalidad"),
    Columna("Ingreso base salud", "int64", "monto", True, "IBC de salud: ingreso declarado"),
    Columna("Días salud", "int64", "conteo", True, ""),
    Columna("Ingreso base caja compensación", "int64", "monto", True, ""),
    Columna("Días caja compensación", "int64", "conteo", True, ""),
    Columna("Ingreso base riesgos", "int64", "monto", True, ""),
    Columna("Días riesgos", "int64", "conteo", True, ""),
)

TRANSFERENCIAS: tuple[Columna, ...] = (
    Columna("Nombre", "string", "identidad", True, ""),
    Columna("Tipo documento", "string", "identidad", True, ""),
    Columna("Número documento", "string", "identidad", False, "Cédula ofuscada — LLAVE DE PERSONA"),
    Columna("Promedio ACH recibidas", "int64", "agregado", True, ""),
    Columna("Promedio Transfiya recibidas", "int64", "agregado", True, ""),
    Columna("Promedio ACH enviadas", "int64", "agregado", True, ""),
    Columna("Promedio Transfiya enviadas", "int64", "agregado", True, ""),
    Columna("Periodo", "string", "periodo", False, "Mes YYYY-MM"),
    Columna("Total recibidas periodo", "float64", "agregado", True, ""),
    Columna("Cantidad recibidas periodo", "int64", "agregado", True, ""),
    Columna("Valor promedio recibidas periodo", "float64", "agregado", True, ""),
    Columna("Total enviadas periodo", "float64", "agregado", True, ""),
    Columna("Cantidad enviadas periodo", "int64", "agregado", True, ""),
    Columna("Valor promedio enviadas periodo", "float64", "agregado", True, ""),
    Columna("Servicio", "string", "categoria", True, "ACH o TRANSFIYA"),
    Columna("Tipo registro", "string", "categoria", True, "RECIBIDO o ENVIADO"),
    Columna("Tipo de cuenta", "string", "categoria", True, ""),
    Columna("Entidad origen", "string", "categoria", True, ""),
    Columna("Usuario originador", "string", "identidad", True, "Nulo estructural en registros ENVIADO"),
    Columna("Nombre usuario originador", "string", "texto", True, "Nulo estructural en registros ENVIADO"),
    Columna("Entidad destino", "string", "categoria", True, ""),
    Columna("Usuario receptor", "string", "identidad", True, "Nulo estructural en registros RECIBIDO"),
    Columna("Nombre usuario receptor", "string", "texto", True, "Nulo estructural en registros RECIBIDO"),
    Columna("Cuenta propia", "string", "categoria", True, "SI = movimiento interno, no es ingreso ni gasto"),
    Columna("Cantidad de transferencias", "int64", "conteo", True, ""),
    Columna("Valor", "float64", "monto", True, ""),
)

PAGOS_DIGITALES: tuple[Columna, ...] = (
    Columna("Nombre", "string", "identidad", True, ""),
    Columna("Tipo documento", "string", "identidad", True, ""),
    Columna("Número documento", "string", "identidad", False, "Cédula ofuscada — LLAVE DE PERSONA"),
    Columna("Promedio pagos digitales", "int64", "agregado", True, ""),
    Columna("Periodo", "string", "periodo", False, "Mes YYYY-MM"),
    Columna("Total pagos periodo", "float64", "agregado", True, ""),
    Columna("Cantidad pagos periodo", "int64", "agregado", True, ""),
    Columna("Valor promedio periodo", "float64", "agregado", True, ""),
    Columna("Entidad autorizadora", "string", "categoria", True, "Banco autorizador; llega 100% enmascarada"),
    Columna("Comercio", "string", "categoria", True, "Comercio ofuscado en sus últimos 8 caracteres"),
    Columna("Cantidad", "int64", "conteo", True, ""),
    Columna("Valor", "float64", "monto", True, ""),
)

DICCIONARIO: dict[str, tuple[Columna, ...]] = {
    "ss": SEGURIDAD_SOCIAL,
    "trf": TRANSFERENCIAS,
    "pse": PAGOS_DIGITALES,
}

NOMBRE_FUENTE: dict[str, str] = {
    "ss": "Seguridad Social (PILA)",
    "trf": "Transferencias ACH/Transfiya",
    "pse": "Pagos Digitales (PSE)",
}

# Nombre del archivo XLSX de cada fuente, tal como lo entrega ACH.
ARCHIVO_FUENTE: dict[str, str] = {
    "ss": "Conversion Seguridad Social 1 - Ofuscado.xlsx",
    "trf": "Conversion Transferencias ACH 1 - Ofuscado.xlsx",
    "pse": "Conversion Pagos Digitales 1 - Ofuscado.xlsx",
}

# Roles explícitos: aquí es donde se elimina la adivinación de columnas.
COLUMNA_PERSONA: dict[str, str] = {
    "ss": "Número de documento",
    "trf": "Número documento",
    "pse": "Número documento",
}
COLUMNA_PERIODO: dict[str, str] = {
    "ss": "Periodo cotización",
    "trf": "Periodo",
    "pse": "Periodo",
}
COLUMNA_NOMBRE: dict[str, str] = {"ss": "Nombre", "trf": "Nombre", "pse": "Nombre"}

_TIPOS_PANDAS: dict[Tipo, str] = {"string": "string", "int64": "Int64", "float64": "float64"}
_TIPOS_ARROW = {
    "string": pyarrow_types.string(),
    "int64": pyarrow_types.int64(),
    "float64": pyarrow_types.float64(),
}


# --------------------------------------------------------------------------- #
# Utilidades del diccionario                                                   #
# --------------------------------------------------------------------------- #
def columnas(fuente: str) -> tuple[Columna, ...]:
    if fuente not in DICCIONARIO:
        raise ErrorDeEsquema(f"Fuente desconocida {fuente!r}. Válidas: {sorted(DICCIONARIO)}")
    return DICCIONARIO[fuente]


def nombres_columnas(fuente: str) -> list[str]:
    return [c.nombre for c in columnas(fuente)]


def columnas_por_rol(fuente: str, rol: Rol) -> list[str]:
    return [c.nombre for c in columnas(fuente) if c.rol == rol]


def esquema_arrow(fuente: str) -> pyarrow_types.Schema:
    """Esquema pyarrow de la fuente, para escribir parquet tipado."""
    return pyarrow_types.schema(
        [pyarrow_types.field(c.nombre, _TIPOS_ARROW[c.tipo], nullable=True) for c in columnas(fuente)]
    )


def esquema_pandera(fuente: str) -> pa.DataFrameSchema:
    """Esquema Pandera con los chequeos de contenido de la fuente."""
    definicion: dict[str, pa.Column] = {}
    for col in columnas(fuente):
        checks = []
        if col.rol == "periodo":
            checks.append(pa.Check.str_matches(r"^\d{4}-\d{2}", error="El periodo debe empezar por YYYY-MM"))
        if col.rol in ("monto", "conteo"):
            checks.append(pa.Check.ge(0, error=f"{col.nombre} no puede ser negativo"))
        definicion[col.nombre] = pa.Column(
            _TIPOS_PANDAS[col.tipo],
            checks=checks or None,
            nullable=col.nullable,
            coerce=True,
            required=True,
        )
    return pa.DataFrameSchema(definicion, strict=False, coerce=True, name=NOMBRE_FUENTE[fuente])


def aplicar_tipos(df: pd.DataFrame, fuente: str) -> pd.DataFrame:
    """Castea las columnas al tipo del diccionario. No altera el orden de filas."""
    salida = df.copy()
    for col in columnas(fuente):
        if col.nombre not in salida.columns:
            continue
        if col.tipo == "string":
            salida[col.nombre] = salida[col.nombre].astype("string")
        else:
            numerico = pd.to_numeric(salida[col.nombre], errors="coerce")
            salida[col.nombre] = numerico.astype("Int64" if col.tipo == "int64" else "float64")
    return salida


def validar_estructura(df: pd.DataFrame, fuente: str) -> None:
    """Chequeo barato sobre las columnas. Falla temprano y con mensaje accionable."""
    esperadas = set(nombres_columnas(fuente))
    presentes = set(map(str, df.columns))
    faltantes = sorted(esperadas - presentes)
    if faltantes:
        sobrantes = sorted(presentes - esperadas)
        detalle = f" Columnas no esperadas en el archivo: {sobrantes}." if sobrantes else ""
        raise ErrorDeEsquema(
            f"[{NOMBRE_FUENTE[fuente]}] faltan {len(faltantes)} columnas del diccionario de datos: "
            f"{faltantes}.{detalle} Revisa que el archivo de entrada sea el extracto correcto."
        )


def validar(df: pd.DataFrame, fuente: str, muestra: int | None = None) -> pd.DataFrame:
    """Valida estructura y contenido. Devuelve el DataFrame ya tipado.

    ``muestra`` limita la validación de contenido a las primeras N filas; la
    validación de estructura siempre corre sobre el DataFrame completo.
    """
    validar_estructura(df, fuente)
    tipado = aplicar_tipos(df, fuente)
    a_validar = tipado.head(muestra) if muestra else tipado
    try:
        esquema_pandera(fuente).validate(a_validar, lazy=True)
    except pa.errors.SchemaErrors as exc:
        fallos = exc.failure_cases[["column", "check", "failure_case"]].head(10)
        raise ErrorDeEsquema(
            f"[{NOMBRE_FUENTE[fuente]}] el contenido no cumple el diccionario de datos.\n"
            f"Primeros fallos:\n{fallos.to_string(index=False)}"
        ) from exc
    return tipado
