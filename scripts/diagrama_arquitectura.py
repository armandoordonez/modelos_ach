"""Dibuja el diagrama de arquitectura de la pipeline.

    python scripts/diagrama_arquitectura.py

Genera PDF (vectorial, para imprimir o adjuntar), SVG (para incrustar en las
diapositivas) y PNG. Se dibuja con matplotlib para que el diagrama sea código
versionado y no una imagen que alguien tenga que rehacer a mano cada vez que cambie
la arquitectura.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # sin display: el script corre en CI o por consola

import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch  # noqa: E402

RAIZ = Path(__file__).resolve().parents[1]
DESTINO = RAIZ / "Docs" / "presentacion"

# Paleta sobria y de alto contraste, coherente con la del tablero.
AZUL, VERDE, NARANJA, MORADO = "#2a78d6", "#008300", "#e9683b", "#4a3aa7"
GRIS_TEXTO, GRIS_SUAVE, BORDE = "#3b3b38", "#8d8d85", "#d5d5cf"
FONDO_CAJA = "#ffffff"


def caja(ax, x, y, ancho, alto, titulo, subtitulo="", color=AZUL, relleno=FONDO_CAJA,
         tam_titulo=10.5, tam_sub=8.2):
    """Caja con título y subtítulo. Las posiciones del texto son relativas al alto de
    la caja, no desplazamientos fijos: si no, el subtítulo se sale en las cajas bajas."""
    ax.add_patch(FancyBboxPatch(
        (x, y), ancho, alto, boxstyle="round,pad=0.006,rounding_size=0.018",
        linewidth=1.6, edgecolor=color, facecolor=relleno, zorder=3))
    centro_x = x + ancho / 2
    if subtitulo:
        ax.text(centro_x, y + alto * 0.66, titulo, ha="center", va="center",
                fontsize=tam_titulo, fontweight="semibold", color=GRIS_TEXTO, zorder=4,
                linespacing=1.35)
        ax.text(centro_x, y + alto * 0.27, subtitulo, ha="center", va="center",
                fontsize=tam_sub, color=GRIS_SUAVE, zorder=4, linespacing=1.45)
    else:
        ax.text(centro_x, y + alto / 2, titulo, ha="center", va="center",
                fontsize=tam_titulo, fontweight="semibold", color=GRIS_TEXTO, zorder=4)


def zona(ax, x, y, ancho, alto, etiqueta, color, alpha=0.05):
    """Zona con su etiqueta en la banda superior, reservada para que no pise las cajas."""
    ax.add_patch(FancyBboxPatch(
        (x, y), ancho, alto, boxstyle="round,pad=0.006,rounding_size=0.02",
        linewidth=1.1, edgecolor=color, facecolor=color, alpha=alpha,
        linestyle=(0, (5, 3)), zorder=1))
    ax.text(x + 0.016, y + alto - 0.030, etiqueta, ha="left", va="center",
            fontsize=8.8, fontweight="semibold", color=color, zorder=2)


def flecha(ax, desde, hasta, color=GRIS_SUAVE, texto="", estilo="-|>", curva=0.0,
           ancho=1.5, desplazar=(0, 0.028)):
    ax.add_patch(FancyArrowPatch(
        desde, hasta, arrowstyle=estilo, mutation_scale=13, linewidth=ancho,
        color=color, zorder=5, connectionstyle=f"arc3,rad={curva}",
        shrinkA=3, shrinkB=3))
    if texto:
        mx, my = (desde[0] + hasta[0]) / 2, (desde[1] + hasta[1]) / 2
        ax.text(mx + desplazar[0], my + desplazar[1], texto, ha="center", va="bottom",
                fontsize=7.8, color=color, zorder=6, fontweight="medium")


def construir() -> plt.Figure:
    fig, ax = plt.subplots(figsize=(13.5, 7.4))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("#fcfcfb")

    ax.text(0.5, 0.965, "Pipeline de modelos analíticos · ACH Colombia",
            ha="center", fontsize=15, fontweight="bold", color=GRIS_TEXTO)
    ax.text(0.5, 0.928,
            "Airflow orquesta · los modelos corren aislados en contenedores · "
            "el tablero solo habla con la API",
            ha="center", fontsize=9.2, color=GRIS_SUAVE)

    # --- Orquestación -------------------------------------------------------
    zona(ax, 0.030, 0.680, 0.940, 0.190, "ORQUESTACIÓN · Airflow 3 (LocalExecutor)", MORADO)
    y_orq, h_orq = 0.700, 0.108
    caja(ax, 0.055, y_orq, 0.175, h_orq, "procesamiento", "1 reintento", MORADO, tam_titulo=9.6)
    caja(ax, 0.275, y_orq, 0.245, h_orq, "modelo × N",
         "dynamic task mapping · pool de 2", MORADO, tam_titulo=9.6, tam_sub=7.5)
    caja(ax, 0.565, y_orq, 0.175, h_orq, "consolidación", "results/index.json", MORADO,
         tam_titulo=9.6, tam_sub=7.5)
    caja(ax, 0.785, y_orq, 0.160, h_orq, "DockerOperator", "vía docker.sock", MORADO,
         tam_titulo=9.2, tam_sub=7.5)

    centro_orq = y_orq + h_orq / 2
    flecha(ax, (0.230, centro_orq), (0.275, centro_orq), MORADO)
    flecha(ax, (0.520, centro_orq), (0.565, centro_orq), MORADO)
    flecha(ax, (0.740, centro_orq), (0.785, centro_orq), MORADO, ancho=1.1)

    # --- Ejecución ----------------------------------------------------------
    zona(ax, 0.030, 0.360, 0.940, 0.250, "EJECUCIÓN · imagen ach-jobs (pandas, scikit-learn)", AZUL)
    y_eje, h_eje = 0.408, 0.135
    caja(ax, 0.055, y_eje, 0.175, h_eje, "Job de procesamiento",
         "XLSX → parquet\nstreaming + validación", AZUL, tam_titulo=9.2, tam_sub=7.6)

    etiquetas = [("#6", "regresión"), ("#15", "clustering"), ("#17", "clasif."),
                 ("#55", "clasif."), ("#4", "clustering"), ("#46", "clustering"),
                 ("#101", "clustering")]
    ancho_m, hueco = 0.0925, 0.011
    x0 = 0.275
    for i, (ref, tipo) in enumerate(etiquetas):
        caja(ax, x0 + i * (ancho_m + hueco), y_eje, ancho_m, h_eje, ref, tipo, AZUL,
             tam_titulo=11, tam_sub=7.2)
    ax.text(0.622, 0.382, "7 modelos · 6 módulos · declarados en models_config.yml",
            ha="center", fontsize=8.4, color=AZUL, fontweight="medium")

    centro_eje = y_eje + h_eje / 2
    flecha(ax, (0.230, centro_eje), (0.275, centro_eje), AZUL)

    # Orquestación → ejecución: cada tarea lanza su contenedor
    for x in (0.142, 0.397, 0.652):
        flecha(ax, (x, y_orq), (x, y_eje + h_eje), MORADO, ancho=1.1)

    # --- Almacenamiento -----------------------------------------------------
    zona(ax, 0.030, 0.075, 0.600, 0.215, "ALMACENAMIENTO · MinIO (S3)", VERDE)
    y_alm, h_alm = 0.105, 0.118
    caja(ax, 0.055, y_alm, 0.160, h_alm, "raw/", "XLSX crudos", VERDE, tam_titulo=10, tam_sub=7.6)
    caja(ax, 0.240, y_alm, 0.180, h_alm, "curated/", "parquet particionado\n+ _manifest.json",
         VERDE, tam_titulo=10, tam_sub=7.6)
    caja(ax, 0.445, y_alm, 0.160, h_alm, "results/", "un JSON por\nmodelo y corrida",
         VERDE, tam_titulo=10, tam_sub=7.6)

    centro_alm = y_alm + h_alm / 2
    flecha(ax, (0.215, centro_alm), (0.240, centro_alm), VERDE, ancho=1.2)
    flecha(ax, (0.420, centro_alm), (0.445, centro_alm), VERDE, ancho=1.2)

    # Lecturas y escrituras entre ejecución y almacenamiento
    flecha(ax, (0.100, y_eje), (0.100, y_alm + h_alm), VERDE, "lee", desplazar=(-0.024, -0.012))
    flecha(ax, (0.180, y_alm + h_alm), (0.180, y_eje), VERDE, "escribe", desplazar=(0.032, -0.012))
    flecha(ax, (0.310, y_eje), (0.310, y_alm + h_alm), VERDE, "lee", desplazar=(-0.024, -0.012))
    flecha(ax, (0.520, y_eje), (0.520, y_alm + h_alm), VERDE, "escribe", desplazar=(0.034, -0.012))

    # --- Aplicación ---------------------------------------------------------
    zona(ax, 0.660, 0.075, 0.310, 0.215, "APLICACIÓN", NARANJA)
    caja(ax, 0.690, y_alm, 0.120, h_alm, "backend", "FastAPI\ncaché TTL", NARANJA,
         tam_titulo=10, tam_sub=7.6)
    caja(ax, 0.835, y_alm, 0.115, h_alm, "tablero", "React\n+ Recharts", NARANJA,
         tam_titulo=10, tam_sub=7.6)

    flecha(ax, (0.605, centro_alm), (0.690, centro_alm), NARANJA, "lee JSON", ancho=1.4,
           desplazar=(0, 0.014))
    flecha(ax, (0.810, centro_alm), (0.835, centro_alm), NARANJA, ancho=1.4)

    # --- Pie ----------------------------------------------------------------
    ax.text(0.035, 0.028,
            "Separación de dependencias: Airflow no tiene pandas ni scikit-learn — solo lanza "
            "contenedores. El backend tampoco — solo lee JSON.",
            ha="left", fontsize=8.2, color=GRIS_SUAVE, style="italic")
    ax.text(0.965, 0.028, "make up · make seed · make trigger",
            ha="right", fontsize=8.2, color=GRIS_SUAVE, family="monospace")

    fig.tight_layout(pad=0.4)
    return fig


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Genera el diagrama de arquitectura.")
    parser.add_argument("--destino", type=Path, default=DESTINO)
    parser.add_argument("--nombre", default="arquitectura")
    args = parser.parse_args(argv)

    args.destino.mkdir(parents=True, exist_ok=True)
    figura = construir()
    salidas = []
    for extension in ("pdf", "svg", "png"):
        ruta = args.destino / f"{args.nombre}.{extension}"
        figura.savefig(ruta, format=extension, dpi=200,
                       facecolor=figura.get_facecolor(), bbox_inches="tight")
        salidas.append(ruta)
    plt.close(figura)

    for ruta in salidas:
        print(f"  {ruta.relative_to(RAIZ)}  ({ruta.stat().st_size / 1024:,.0f} KB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
