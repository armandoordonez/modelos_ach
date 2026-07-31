"""Genera un extracto sintético y lo sube al bucket ``raw`` para probar de punta a punta.

    python scripts/seed_minio.py --personas 1500

Sirve para levantar el stack y ver la pipeline completa **sin los datos reales de ACH**,
que son confidenciales y no se versionan. Los datos que genera tienen la misma forma
que la entrega real —mismas columnas, mismos tipos, mismas ventanas— pero son
inventados: las métricas que salgan de aquí **no significan nada de negocio**, solo
demuestran que la pipeline funciona.

Genera XLSX y no CSV a propósito: es el formato en que ACH entrega los extractos y el
que consume el job de procesamiento. El día que la entrega cambie a CSV o parquet, este
script y el lector cambian juntos.

Las personas se construyen a partir de arquetipos de comportamiento para que los
modelos de segmentación encuentren estructura real y los de propensión tengan ambas
clases; con datos uniformes no habría nada que agrupar ni que predecir.
"""

from __future__ import annotations

import argparse
import logging
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ / "jobs") not in sys.path:
    sys.path.insert(0, str(RAIZ / "jobs"))

from common.logging_config import configurar_logging  # noqa: E402
from common.schema import ARCHIVO_FUENTE, nombres_columnas  # noqa: E402
from common.storage import get_storage  # noqa: E402

log = logging.getLogger(__name__)

SMLMV = 1_423_500

# Arquetipos de persona. Cada uno define el nivel y la forma de su actividad, que es
# lo que después encuentran los modelos de segmentación.
ARQUETIPOS = [
    # (nombre, peso, ingreso × SMLMV, actividad PSE, propensión salud, propensión viajes)
    ("asalariado_base",     0.34, 1.05, 0.55, 0.06, 0.03),
    ("asalariado_medio",    0.22, 2.20, 0.80, 0.12, 0.10),
    ("profesional_alto",    0.12, 5.50, 0.95, 0.20, 0.28),
    ("independiente",       0.18, 1.40, 0.65, 0.10, 0.08),
    ("pensionado_rutina",   0.06, 1.15, 0.75, 0.22, 0.05),
    ("pensionado_esencial", 0.04, 0.90, 0.45, 0.30, 0.02),
    ("bajo_uso",            0.04, 1.00, 0.20, 0.03, 0.01),
]

COMERCIOS = [
    # (nombre ofuscado, categoría a la que cae, peso, ticket medio)
    ("BANCOLOM********",                    "financiero", 0.16, 320_000),
    ("BANCO DAVIVIENDA S.A. (ZON********",  "financiero", 0.10, 280_000),
    ("NEQUI SERVICIOS FINANC********",      "financiero", 0.08, 90_000),
    ("CLARO SOLUCIONES MOVI********",       "telco",      0.14, 68_000),
    ("MOVISTAR COLOMBIA TELECOM********",   "telco",      0.08, 55_000),
    ("ACUEDUCTO Y ALCANTARILLADO DE********", "telco",    0.07, 74_000),
    ("DROGUERIA Y FARMACIA LA REB********",  "salud",     0.06, 62_000),
    ("CLINICA DEL NORTE SERVICIOS********",  "salud",     0.04, 210_000),
    ("EPS SANITAS MEDICINA PREPAG********",  "salud",     0.04, 155_000),
    ("AVIANCA SERVICIOS AEREOS********",     "viajes",    0.03, 780_000),
    ("HOTEL ESTELAR DE LA SABANA********",   "viajes",    0.02, 420_000),
    ("EXPRESO BRASILIA TRANSPO********",     "viajes",    0.02, 95_000),
    ("ALMACENES EXITO-PRO********",          "retail",    0.05, 180_000),
    ("FALABELLA.CO COLOM********",           "retail",    0.03, 240_000),
    ("SECRETARIA DE HACIENDA DISTRI********", "gobierno", 0.04, 130_000),
    ("UNIVERSIDAD NACIONAL DE COL********",  "educacion", 0.02, 950_000),
    ("PAYU COLOMB********",                  "pasarela",  0.02, 88_000),
]
PESOS_COMERCIO = np.array([c[2] for c in COMERCIOS])
PESOS_COMERCIO = PESOS_COMERCIO / PESOS_COMERCIO.sum()

PAGADORES_PENSION = [
    "ADMINISTRADORA COLOMBIANA DE PENSIONES COLP********",
    "PORV********",
    "FONDO DE PENSIONES OBLIGATORIAS PROTECCION RETIRO PR********",
    "SEGUROS DE VIDA ********",
    "CONSORCIO FO********",
]
RELACIONES_PENSION = [
    "Pensionado de regimen de prima media con tope maximo 25 SMLMV",
    "Pensionado de ahorro individual no aplica tope maximo de pension",
    "Pensionado por el empleador tope maximo de pension 25 SMLMV",
]
EMPLEADORES = [f"EMPRESA {letra}********" for letra in "ABCDEFGHIJKLMNOP"]


def _periodos(inicio: str, fin: str) -> list[str]:
    return [str(p) for p in pd.period_range(inicio, fin, freq="M")]


def _documento(indice: int) -> str:
    return f"{10_000_000 + indice * 7:08d}****"


def _nombre(indice: int, rng: np.random.Generator) -> str:
    nombres = ["ANA", "LUIS", "MARIA", "CARLOS", "SOFIA", "JORGE", "LAURA", "DIEGO", "PAULA", "ANDRES"]
    apellidos = ["GOMEZ", "RODRIGUEZ", "MARTINEZ", "LOPEZ", "PEREZ", "SANCHEZ", "RAMIREZ", "TORRES"]
    return f"****{nombres[indice % len(nombres)]} {apellidos[rng.integers(len(apellidos))]}"


def generar_personas(n: int, rng: np.random.Generator) -> pd.DataFrame:
    pesos = np.array([a[1] for a in ARQUETIPOS])
    pesos = pesos / pesos.sum()
    elegidos = rng.choice(len(ARQUETIPOS), size=n, p=pesos)
    return pd.DataFrame({
        "indice": range(n),
        "documento": [_documento(i) for i in range(n)],
        "nombre": [_nombre(i, rng) for i in range(n)],
        "arquetipo": [ARQUETIPOS[i][0] for i in elegidos],
        "ingreso": [ARQUETIPOS[i][2] * SMLMV * rng.lognormal(0, 0.18) for i in elegidos],
        "actividad": [np.clip(ARQUETIPOS[i][3] * rng.normal(1, 0.15), 0.05, 1.0) for i in elegidos],
        "p_salud": [ARQUETIPOS[i][4] for i in elegidos],
        "p_viajes": [ARQUETIPOS[i][5] for i in elegidos],
    })


def generar_seguridad_social(personas: pd.DataFrame, periodos: list[str],
                             rng: np.random.Generator) -> pd.DataFrame:
    filas = []
    fecha_base = date(2023, 7, 15)
    for persona in personas.itertuples():
        es_pensionado = persona.arquetipo.startswith("pensionado")
        # Los pensionados cotizan salud sobre su mesada; el resto, sobre su salario.
        pagador = (PAGADORES_PENSION[persona.indice % len(PAGADORES_PENSION)] if es_pensionado
                   else EMPLEADORES[persona.indice % len(EMPLEADORES)])
        relacion = (RELACIONES_PENSION[persona.indice % len(RELACIONES_PENSION)] if es_pensionado
                    else ("INDEPENDIENTE" if persona.arquetipo == "independiente" else "DEPENDIENTE"))
        planilla = "P" if es_pensionado else ("I" if relacion == "INDEPENDIENTE" else "E")
        aportante = ("PAGADOR DE PENSIONES" if es_pensionado
                     else ("INDEPENDIENTE" if relacion == "INDEPENDIENTE" else "EMPLEADOR"))

        for i, periodo in enumerate(periodos):
            if rng.random() > 0.94:      # meses sin cotización
                continue
            ibc = int(max(persona.ingreso * rng.normal(1, 0.05), SMLMV * 0.35))
            dias_pension = 0 if (es_pensionado or rng.random() < 0.5) else 30
            filas.append({
                "Nombre": persona.nombre, "Tipo de documento": "P",
                "Número de documento": persona.documento,
                "Promedio salario básico": ibc, "Promedio ingreso base pensión": 0,
                "Promedio ingreso base salud": ibc,
                "Promedio ingreso base caja compensación": 0, "Promedio ingreso base riesgos": ibc,
                "Razón social": pagador, "Tipo documento": "J",
                "Número documento": f"{90_000_000 + persona.indice % 40:08d}****",
                "Código actividad económica": 4771 if not es_pensionado else 6531,
                "Actividad económica": "Régimen de prima media" if es_pensionado else "Comercio al por menor",
                "Clase aportante": "EMPRESAS CON MAS DE 200 COTIZANTES",
                "Tipo aportante": aportante,
                "Fecha de pago": str(fecha_base + timedelta(days=30 * i)),
                "Periodo cotización": periodo, "Empleador": pagador,
                "Tipo planilla": planilla, "Relación laboral": relacion,
                "Salario básico": 0 if es_pensionado else ibc,
                "Tipo salario": "F", "Novedades": None,
                "Ingreso base pensión": 0 if es_pensionado else ibc,
                "Días pensión": dias_pension, "Ingreso base salud": ibc, "Días salud": 30,
                "Ingreso base caja compensación": 0, "Días caja compensación": 0,
                "Ingreso base riesgos": 0 if es_pensionado else ibc,
                "Días riesgos": 0 if es_pensionado else 30,
            })
    return pd.DataFrame(filas)[nombres_columnas("ss")]


def generar_transferencias(personas: pd.DataFrame, periodos: list[str],
                           rng: np.random.Generator) -> pd.DataFrame:
    filas = []
    for persona in personas.itertuples():
        for periodo in periodos:
            if rng.random() > persona.actividad:
                continue
            # El piso en cero no es cosmético: la validación del diccionario de datos
            # rechaza montos negativos, y una normal con esta dispersión los produce.
            recibido = max(float(persona.ingreso * rng.normal(0.85, 0.2)), 0.0)
            base = {
                "Nombre": persona.nombre, "Tipo documento": "P", "Número documento": persona.documento,
                "Promedio ACH recibidas": 0, "Promedio Transfiya recibidas": 0,
                "Promedio ACH enviadas": 0, "Promedio Transfiya enviadas": 0, "Periodo": periodo,
                "Total recibidas periodo": 0.0, "Cantidad recibidas periodo": 0,
                "Valor promedio recibidas periodo": 0.0, "Total enviadas periodo": 0.0,
                "Cantidad enviadas periodo": 0, "Valor promedio enviadas periodo": 0.0,
                "Tipo de cuenta": "AHORROS", "Entidad origen": "BAN********",
                "Usuario originador": "9********", "Nombre usuario originador": "EMPRESA********",
                "Entidad destino": None, "Usuario receptor": None, "Nombre usuario receptor": None,
            }
            filas.append({**base, "Servicio": "ACH", "Tipo registro": "RECIBIDO",
                          "Cuenta propia": "NO", "Cantidad de transferencias": 1,
                          "Valor": round(max(recibido, 0), 2)})
            if rng.random() < 0.7:
                filas.append({**base, "Servicio": "TRANSFIYA", "Tipo registro": "ENVIADO",
                              "Cuenta propia": "NO",
                              "Cantidad de transferencias": int(rng.integers(1, 6)),
                              "Valor": round(float(recibido * rng.uniform(0.1, 0.4)), 2)})
            if rng.random() < 0.15:      # movimiento entre cuentas propias
                filas.append({**base, "Servicio": "ACH", "Tipo registro": "RECIBIDO",
                              "Cuenta propia": "SI", "Cantidad de transferencias": 1,
                              "Valor": round(float(recibido * rng.uniform(0.5, 2.0)), 2)})
    return pd.DataFrame(filas)[nombres_columnas("trf")]


def generar_pagos(personas: pd.DataFrame, periodos: list[str],
                  rng: np.random.Generator) -> pd.DataFrame:
    indices_salud = [i for i, c in enumerate(COMERCIOS) if c[1] == "salud"]
    indices_viajes = [i for i, c in enumerate(COMERCIOS) if c[1] == "viajes"]
    filas = []
    for persona in personas.itertuples():
        for periodo in periodos:
            if rng.random() > persona.actividad:
                continue
            n_pagos = int(rng.integers(1, 7))
            elegidos = list(rng.choice(len(COMERCIOS), size=n_pagos, p=PESOS_COMERCIO))
            # La propensión del arquetipo decide si además aparece salud o viajes:
            # así los modelos de clasificación tienen señal real que aprender.
            if rng.random() < persona.p_salud:
                elegidos.append(int(rng.choice(indices_salud)))
            if rng.random() < persona.p_viajes:
                elegidos.append(int(rng.choice(indices_viajes)))

            for indice in elegidos:
                comercio, _, _, ticket = COMERCIOS[indice]
                filas.append({
                    "Nombre": persona.nombre, "Tipo documento": "P",
                    "Número documento": persona.documento, "Promedio pagos digitales": 0,
                    "Periodo": periodo, "Total pagos periodo": 0.0, "Cantidad pagos periodo": 0,
                    "Valor promedio periodo": 0.0, "Entidad autorizadora": "*****",
                    "Comercio": comercio, "Cantidad": 1,
                    "Valor": round(float(ticket * rng.lognormal(0, 0.35)), 2),
                })
    return pd.DataFrame(filas)[nombres_columnas("pse")]


def escribir_xlsx(df: pd.DataFrame, destino: Path) -> Path:
    """Escribe en modo write_only para no cargar el libro completo en memoria."""
    from openpyxl import Workbook

    libro = Workbook(write_only=True)
    hoja = libro.create_sheet()
    hoja.append(list(df.columns))
    for fila in df.itertuples(index=False, name=None):
        hoja.append([None if pd.isna(v) else v for v in fila])
    libro.save(destino)
    return destino


def sembrar(personas: int, fecha: str, inicio: str, fin: str, semilla: int) -> dict[str, int]:
    rng = np.random.default_rng(semilla)
    periodos_comunes = _periodos(inicio, fin)
    # Seguridad Social cubre más historia que las fuentes transaccionales, igual que
    # en la entrega real (36 meses contra 18).
    periodos_ss = _periodos(str(pd.Period(inicio, freq="M") - 18), fin)

    log.info("Generando %s personas · %d periodos transaccionales · %d de PILA",
             f"{personas:,}", len(periodos_comunes), len(periodos_ss))
    base = generar_personas(personas, rng)
    log.info("Arquetipos: %s", base["arquetipo"].value_counts().to_dict())

    tablas = {
        "ss": generar_seguridad_social(base, periodos_ss, rng),
        "trf": generar_transferencias(base, periodos_comunes, rng),
        "pse": generar_pagos(base, periodos_comunes, rng),
    }

    storage = get_storage()
    conteos = {}
    with tempfile.TemporaryDirectory(prefix="ach_seed_") as temporal:
        for fuente, df in tablas.items():
            nombre = ARCHIVO_FUENTE[fuente]
            local = escribir_xlsx(df, Path(temporal) / nombre)
            remoto = storage.ruta(storage.settings.bucket_raw, fecha, nombre)
            storage.crear_directorio(remoto.rsplit("/", 1)[0])
            with open(local, "rb") as origen, storage.fs.open(remoto, "wb") as destino:
                destino.write(origen.read())
            conteos[fuente] = len(df)
            log.info("%s: %s filas → %s", nombre, f"{len(df):,}", remoto)

    log.info("Carga sintética %s lista. Dispara el DAG con: make trigger", fecha)
    return conteos


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="seed_minio",
        description="Genera un extracto sintético con la forma de la entrega de ACH y lo sube a raw/.")
    parser.add_argument("--personas", type=int, default=1500, help="Cuántas personas generar.")
    parser.add_argument("--fecha", default=date.today().isoformat(), help="Carpeta dentro de raw/.")
    parser.add_argument("--inicio", default="2025-01", help="Primer periodo transaccional.")
    parser.add_argument("--fin", default="2026-06", help="Último periodo transaccional.")
    parser.add_argument("--semilla", type=int, default=42, help="Semilla de reproducibilidad.")
    args = parser.parse_args(argv)

    configurar_logging()
    log.warning("Estos datos son SINTÉTICOS: sirven para probar la pipeline, no para "
                "sacar conclusiones de negocio.")
    sembrar(args.personas, args.fecha, args.inicio, args.fin, args.semilla)
    return 0


if __name__ == "__main__":
    sys.exit(main())
