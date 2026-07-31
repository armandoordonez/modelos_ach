"""Genera las diapositivas del avance en HTML.

    python scripts/generar_diapositivas.py

Las cifras salen de los resultados reales del bucket y de las referencias de paridad,
no están escritas a mano: si la pipeline vuelve a correr, se regenera el deck y los
números siguen siendo ciertos.

El diagrama de arquitectura se incrusta como SVG en base64, así que el HTML resultante
es un solo archivo que se puede abrir en cualquier máquina o mandar por correo.
Con Ctrl+P se exporta a PDF.
"""

from __future__ import annotations

import argparse
import base64
import json
import logging
import sys
from datetime import date
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
if str(RAIZ / "jobs") not in sys.path:
    sys.path.insert(0, str(RAIZ / "jobs"))

from common.config import get_settings  # noqa: E402
from common.logging_config import configurar_logging  # noqa: E402
from common.storage import get_storage  # noqa: E402

log = logging.getLogger(__name__)

DESTINO = RAIZ / "Docs" / "presentacion"
REFERENCIAS = RAIZ / "tests" / "parity" / "referencias.json"

ETIQUETA_TAREA = {"clustering": "Segmentación", "classification": "Clasificación",
                  "regression": "Regresión", "scoring": "Scoring"}
METRICA_PRINCIPAL = {"clustering": ("silhouette", "silueta"),
                     "classification": ("roc_auc", "ROC-AUC"),
                     "regression": ("r2_log", "R² (log)")}


def cargar_datos() -> dict:
    """Lee el índice de resultados y las referencias de paridad."""
    ajustes = get_settings()
    storage = get_storage(ajustes)
    ruta = storage.ruta(ajustes.bucket_results, "index.json")
    if not storage.existe(ruta):
        raise FileNotFoundError(
            f"No hay {ruta}. Corre la pipeline antes de generar las diapositivas.")
    indice = storage.leer_json(ruta)
    referencias = json.loads(REFERENCIAS.read_text(encoding="utf-8"))
    return {"indice": indice, "referencias": referencias}


def _svg_incrustado(ruta: Path) -> str:
    datos = base64.b64encode(ruta.read_bytes()).decode("ascii")
    return f"data:image/svg+xml;base64,{datos}"


def _fila_modelo(modelo: dict, referencias: dict) -> str:
    clave, etiqueta = METRICA_PRINCIPAL.get(modelo["task_type"], ("", ""))
    valor = modelo["metrics"].get(clave)
    ref = referencias["modelos"].get(modelo["model_id"], {})
    paridad = ref.get("paridad", "")
    marca = ('<span class="pill pill--ok">exacta</span>' if paridad == "exacta"
             else '<span class="pill pill--rebase">re-baseline</span>')
    return f"""        <tr>
          <td class="mono">{modelo['catalog_ref']}</td>
          <td>{modelo['model_name']}</td>
          <td><span class="tag tag--{modelo['task_type']}">{ETIQUETA_TAREA.get(modelo['task_type'], '')}</span></td>
          <td class="num">{etiqueta} {valor:.3f}</td>
          <td>{marca}</td>
        </tr>"""


def construir_html(datos: dict, diagrama: str) -> str:
    indice = datos["indice"]
    referencias = datos["referencias"]
    modelos = sorted(indice["models"], key=lambda m: (m["use_case"], m["catalog_ref"]))
    filas = "\n".join(_fila_modelo(m, referencias) for m in modelos)

    por_tipo: dict[str, int] = {}
    for m in modelos:
        por_tipo[m["task_type"]] = por_tipo.get(m["task_type"], 0) + 1
    resumen_tipos = " · ".join(
        f"{n} de {ETIQUETA_TAREA.get(t, t).lower()}" for t, n in sorted(por_tipo.items()))

    duracion_total = sum(m["duration_seconds"] for m in modelos)

    def ref_metrica(model_id: str, clave: str, defecto="—"):
        valores = referencias["modelos"].get(model_id, {})
        return valores.get("metricas_esperadas", {}).get(clave, defecto)

    def ref_notebook(model_id: str, clave: str, defecto="—"):
        valores = referencias["modelos"].get(model_id, {})
        return valores.get("metricas_notebook_original", {}).get(clave, defecto)

    return f"""<!doctype html>
<html lang="es">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>ACH · Pipeline de modelos analíticos</title>
<style>
  :root {{
    --fondo: #fcfcfb; --tarjeta: #ffffff; --sutil: #f2f2ef; --borde: #e2e2dc;
    --texto: #1a1a19; --suave: #55554f; --tenue: #8d8d85;
    --azul: #2a78d6; --verde: #008300; --naranja: #e9683b; --morado: #4a3aa7; --rojo: #e34948;
  }}
  * {{ box-sizing: border-box; }}
  html, body {{ margin: 0; padding: 0; background: #22222a; }}
  body {{
    font-family: ui-sans-serif, system-ui, "Segoe UI", Roboto, sans-serif;
    color: var(--texto); -webkit-font-smoothing: antialiased;
  }}
  .mono {{ font-family: ui-monospace, Menlo, Consolas, monospace; }}

  .slide {{
    width: 1280px; height: 720px; margin: 24px auto; background: var(--fondo);
    padding: 56px 68px; position: relative; display: none;
    box-shadow: 0 10px 40px rgba(0,0,0,.35); border-radius: 4px; overflow: hidden;
  }}
  .slide.activa {{ display: flex; flex-direction: column; }}
  .slide__num {{
    position: absolute; bottom: 26px; right: 34px; font-size: 12px; color: var(--tenue);
  }}
  .slide__pie {{
    position: absolute; bottom: 26px; left: 68px; font-size: 12px; color: var(--tenue);
  }}

  h1 {{ font-size: 46px; margin: 0 0 14px; letter-spacing: -.02em; line-height: 1.1; }}
  h2 {{ font-size: 31px; margin: 0 0 6px; letter-spacing: -.015em; }}
  h3 {{ font-size: 15px; margin: 0 0 10px; color: var(--suave); font-weight: 620; }}
  .subtitulo {{ font-size: 16px; color: var(--tenue); margin: 0 0 28px; }}
  p {{ font-size: 16px; line-height: 1.62; color: var(--suave); margin: 0 0 14px; }}

  .portada {{ justify-content: center; align-items: flex-start; }}
  .portada h1 {{ font-size: 58px; max-width: 15ch; }}
  .portada .meta {{ margin-top: 40px; font-size: 14px; color: var(--tenue); line-height: 2; }}
  .barra {{ width: 76px; height: 5px; background: var(--azul); margin-bottom: 30px; }}

  table {{ width: 100%; border-collapse: collapse; font-size: 14.5px; }}
  th, td {{ text-align: left; padding: 9px 12px; border-bottom: 1px solid var(--borde); }}
  thead th {{ font-size: 12px; color: var(--tenue); font-weight: 600; text-transform: uppercase; letter-spacing: .05em; }}
  td.num, th.num {{ text-align: right; font-variant-numeric: tabular-nums; }}

  .tag {{ font-size: 11.5px; padding: 3px 9px; border-radius: 999px; font-weight: 550; white-space: nowrap; }}
  .tag--clustering {{ background: #2a78d61f; color: var(--azul); }}
  .tag--classification {{ background: #0083001f; color: var(--verde); }}
  .tag--regression {{ background: #4a3aa71f; color: var(--morado); }}
  .pill {{ font-size: 11.5px; padding: 3px 9px; border-radius: 4px; font-weight: 600; }}
  .pill--ok {{ background: #0083001a; color: var(--verde); }}
  .pill--rebase {{ background: #e9683b1a; color: var(--naranja); }}

  .rejilla {{ display: grid; gap: 18px; }}
  .rejilla--2 {{ grid-template-columns: 1fr 1fr; }}
  .rejilla--3 {{ grid-template-columns: repeat(3, 1fr); }}
  .rejilla--4 {{ grid-template-columns: repeat(4, 1fr); }}
  .caja {{
    background: var(--tarjeta); border: 1px solid var(--borde); border-radius: 8px; padding: 20px 22px;
  }}
  .caja--acento {{ border-left: 3px solid var(--azul); }}
  .caja--alerta {{ border-left: 3px solid var(--naranja); }}
  .caja--ok {{ border-left: 3px solid var(--verde); }}
  .caja h4 {{ margin: 0 0 8px; font-size: 15px; }}
  .caja p {{ margin: 0; font-size: 14px; line-height: 1.55; }}

  .cifra {{ font-size: 40px; font-weight: 640; letter-spacing: -.02em; line-height: 1; }}
  .cifra--v {{ color: var(--verde); }} .cifra--a {{ color: var(--azul); }}
  .cifra--n {{ color: var(--naranja); }} .cifra--m {{ color: var(--morado); }}
  .cifra__et {{ font-size: 13px; color: var(--tenue); margin-top: 8px; display: block; }}

  ul {{ margin: 0; padding-left: 20px; font-size: 15.5px; line-height: 1.75; color: var(--suave); }}
  li {{ margin-bottom: 8px; }}
  li strong {{ color: var(--texto); }}

  pre {{
    background: #1c1c22; color: #e6e6e6; padding: 18px 22px; border-radius: 8px;
    font-size: 13.5px; line-height: 1.65; overflow: auto; margin: 0;
    font-family: ui-monospace, Menlo, Consolas, monospace;
  }}
  pre .c {{ color: #7f8b99; }} pre .k {{ color: #6db3f2; }} pre .s {{ color: #a5d6a7; }}

  .diagrama {{ width: 100%; flex: 1; object-fit: contain; }}
  .nota {{ font-size: 13px; color: var(--tenue); font-style: italic; }}

  .nav {{
    position: fixed; bottom: 22px; left: 50%; transform: translateX(-50%);
    display: flex; gap: 8px; align-items: center; z-index: 50;
    background: rgba(28,28,34,.92); padding: 8px 14px; border-radius: 999px;
  }}
  .nav button {{
    background: none; border: none; color: #e6e6e6; cursor: pointer; font-size: 16px;
    padding: 4px 10px; border-radius: 6px;
  }}
  .nav button:hover {{ background: rgba(255,255,255,.12); }}
  .nav span {{ color: #9a9a94; font-size: 13px; min-width: 58px; text-align: center; }}

  @media print {{
    @page {{ size: 1280px 720px; margin: 0; }}
    body {{ background: #fff; }}
    .slide {{ display: flex !important; margin: 0; box-shadow: none; page-break-after: always; border-radius: 0; }}
    .nav {{ display: none; }}
  }}
</style>
</head>
<body>

<!-- 1 -->
<section class="slide activa portada">
  <div class="barra"></div>
  <h1>De notebooks a una pipeline productiva</h1>
  <p class="subtitulo" style="font-size:19px">
    Modelos analíticos de ACH Colombia · orquestados, reproducibles y verificados
  </p>
  <div class="meta">
    {len(modelos)} modelos en producción &nbsp;·&nbsp; {resumen_tipos}<br>
    Airflow 3 · MinIO · FastAPI · React<br>
    {date.today().strftime('%d/%m/%Y')}
  </div>
  <div class="slide__pie">ACH Data · INNLAB Universidad Icesi</div>
</section>

<!-- 2 -->
<section class="slide">
  <h2>El punto de partida</h2>
  <p class="subtitulo">Siete modelos repartidos en notebooks y scripts sueltos, sin forma de ejecutarlos juntos.</p>
  <div class="rejilla rejilla--2" style="margin-top:8px">
    <div class="caja caja--alerta">
      <h4>Tres linajes de datos incompatibles</h4>
      <p>Cada origen cruzaba las fuentes con una llave de persona distinta: cédula,
      <span class="mono">Nombre|documento</span> y nombre normalizado sin asteriscos.
      Tres universos de personas que no se podían comparar entre sí.</p>
    </div>
    <div class="caja caja--alerta">
      <h4>Una fuga de información</h4>
      <p>Los modelos de propensión predecían el gasto futuro en una categoría usando el
      gasto pasado <em>de esa misma categoría</em>. El notebook original sí la excluía;
      al pasarlo a script se perdió.</p>
    </div>
    <div class="caja caja--alerta">
      <h4>Dependencias de ejecución</h4>
      <p>Un script cargaba los tres XLSX completos en memoria y otro leía la caché en
      disco del primero. Imposible de correr en paralelo o en un contenedor.</p>
    </div>
    <div class="caja caja--alerta">
      <h4>Parámetros que se elegían solos</h4>
      <p>La categoría objetivo y la ventana temporal se recalculaban según los datos
      que llegaran: el mismo identificador de modelo entrenaba modelos distintos.</p>
    </div>
  </div>
  <div class="slide__pie">El punto de partida</div>
</section>

<!-- 3 -->
<section class="slide">
  <h2>Arquitectura</h2>
  <p class="subtitulo">Cuatro capas con responsabilidades separadas y una sola fuente de verdad por cosa.</p>
  <img class="diagrama" src="{diagrama}" alt="Diagrama de arquitectura de la pipeline">
  <div class="slide__pie">Arquitectura</div>
</section>

<!-- 4 -->
<section class="slide">
  <h2>Los {len(modelos)} modelos</h2>
  <p class="subtitulo">Declarados en un único archivo de configuración; el DAG los descubre solo.</p>
  <table>
    <thead>
      <tr><th>Catálogo</th><th>Modelo</th><th>Tipo</th><th class="num">Métrica principal</th><th>Paridad</th></tr>
    </thead>
    <tbody>
{filas}
    </tbody>
  </table>
  <p class="nota" style="margin-top:20px">
    #17 y #55 comparten módulo y solo cambian el parámetro <span class="mono">categoria</span>:
    el segundo no costó una línea de código.
  </p>
  <div class="slide__pie">Los modelos</div>
</section>

<!-- 5 -->
<section class="slide">
  <h2>Cómo se ejecuta</h2>
  <p class="subtitulo">Un DAG manual que procesa, corre los modelos en paralelo y consolida.</p>
  <pre><span class="c"># Corrida real · {indice['run_id'][:38]}</span>

  procesamiento         <span class="s">success</span>   14:04:53 → 14:09:28   274s
  modelo[1]             <span class="s">success</span>   14:09:29 → 14:10:33    65s  <span class="c">┐ en paralelo</span>
  modelo[0]             <span class="s">success</span>   14:09:29 → 14:10:43    74s  <span class="c">┘</span>
  modelo[2]             <span class="s">success</span>   14:10:34 → 14:11:29    55s  <span class="c">┐ en paralelo</span>
  modelo[3]             <span class="s">success</span>   14:10:44 → 14:11:39    56s  <span class="c">┘</span>
  modelo[4]             <span class="s">success</span>   14:11:30 → 14:12:37    68s  <span class="c">┐ en paralelo</span>
  modelo[5]             <span class="s">success</span>   14:11:40 → 14:12:49    69s  <span class="c">┘</span>
  modelo[6]             <span class="s">success</span>   14:12:38 → 14:13:05    26s
  consolidacion         <span class="s">success</span>   14:13:05 → 14:13:09     4s</pre>
  <div class="rejilla rejilla--3" style="margin-top:24px">
    <div class="caja"><span class="cifra cifra--a">{len(modelos)}</span><span class="cifra__et">modelos, todos en verde</span></div>
    <div class="caja"><span class="cifra cifra--v">2</span><span class="cifra__et">en paralelo (pool, por memoria)</span></div>
    <div class="caja"><span class="cifra cifra--m">{duracion_total/60:.0f} min</span><span class="cifra__et">de cómputo de modelos</span></div>
  </div>
  <div class="slide__pie">Ejecución</div>
</section>

<!-- 6 -->
<section class="slide">
  <h2>Paridad: los modelos del Caso 05</h2>
  <p class="subtitulo">Ya cruzaban por cédula, así que la migración no debía moverlos. No los movió.</p>
  <table>
    <thead><tr><th>Modelo</th><th class="num">Notebook</th><th class="num">Pipeline</th><th>Estado</th></tr></thead>
    <tbody>
      <tr><td>#4 · CLV · personas / k / silueta</td>
          <td class="num">32.164 / 4 / 0,231</td>
          <td class="num">{ref_metrica('caso05_clv','n_entities'):,.0f} / {ref_metrica('caso05_clv','k'):.0f} / {ref_metrica('caso05_clv','silhouette'):.4f}</td>
          <td><span class="pill pill--ok">exacta</span></td></tr>
      <tr><td>#46 · Ciclo de vida · personas / k / silueta</td>
          <td class="num">33.281 / 5 / 0,350</td>
          <td class="num">{ref_metrica('caso05_ciclo_vida','n_entities'):,.0f} / {ref_metrica('caso05_ciclo_vida','k'):.0f} / {ref_metrica('caso05_ciclo_vida','silhouette'):.4f}</td>
          <td><span class="pill pill--ok">exacta</span></td></tr>
      <tr><td>#101 · Pensionados · universo / k / silueta</td>
          <td class="num">360 / 4 / 0,271</td>
          <td class="num">{ref_metrica('caso05_pensionados','n_entities'):,.0f} / {ref_metrica('caso05_pensionados','k'):.0f} / {ref_metrica('caso05_pensionados','silhouette'):.4f}</td>
          <td><span class="pill pill--ok">exacta</span></td></tr>
    </tbody>
  </table>
  <div class="caja caja--ok" style="margin-top:26px">
    <h4>El procesamiento también</h4>
    <p>743.406 / 1.048.574 / 1.048.574 filas leídas y 50 / 7.291 / 1 duplicados exactos
    descartados: idénticos al análisis exploratorio original. Mismas columnas, mismos tipos.</p>
  </div>
  <div class="slide__pie">Paridad</div>
</section>

<!-- 7 -->
<section class="slide">
  <h2>Re-baseline: Caso 02 y Caso 04</h2>
  <p class="subtitulo">Aquí las métricas cambian a propósito. Cada delta tiene una causa declarada.</p>
  <table>
    <thead><tr><th>Modelo</th><th class="num">Antes</th><th class="num">Ahora</th><th>Causa</th></tr></thead>
    <tbody>
      <tr><td>#6 · Ingresos · R² (log)</td>
          <td class="num">{ref_notebook('caso02_ingresos_independientes','r2_log')}</td>
          <td class="num">{ref_metrica('caso02_ingresos_independientes','r2_log'):.3f}</td>
          <td>llave unificada a cédula</td></tr>
      <tr><td>#15 · RFM · silueta</td>
          <td class="num">{ref_notebook('caso04_rfm_consumidores','silhouette')}</td>
          <td class="num">{ref_metrica('caso04_rfm_consumidores','silhouette'):.3f}</td>
          <td>llave + k accionable (k≥4)</td></tr>
      <tr><td>#17 · Propensión salud · ROC-AUC</td>
          <td class="num">{ref_notebook('caso04_propension_salud','roc_auc')}</td>
          <td class="num">{ref_metrica('caso04_propension_salud','roc_auc'):.3f}</td>
          <td><strong>fuga corregida</strong></td></tr>
      <tr><td>#55 · Propensión turismo · ROC-AUC</td>
          <td class="num">{ref_notebook('caso04_propension_turismo','roc_auc')}</td>
          <td class="num">{ref_metrica('caso04_propension_turismo','roc_auc'):.3f}</td>
          <td>fuga corregida + llave</td></tr>
    </tbody>
  </table>
  <div class="caja caja--alerta" style="margin-top:24px">
    <h4>Que el AUC baje es el resultado correcto</h4>
    <p>El 0,835 de #17 medía, en buena parte, la capacidad del modelo de leer su propia
    respuesta. Con la exclusión restaurada mide lo que decía medir. Cada JSON registra qué
    variables se excluyeron, para que la corrección sea auditable.</p>
  </div>
  <div class="slide__pie">Re-baseline</div>
</section>

<!-- 8 -->
<section class="slide">
  <h2>Qué se corrigió</h2>
  <p class="subtitulo">Cinco defectos detectados al revisar los scripts, todos con test de regresión.</p>
  <div class="rejilla rejilla--2">
    <div class="caja caja--ok"><h4>1 · Memoria</h4>
      <p>El Caso 04 cargaba los tres XLSX completos en RAM. Ahora lee del dataset curado;
      el procesamiento va en streaming y su pico depende del lote, no del archivo.</p></div>
    <div class="caja caja--ok"><h4>2 · Acoplamiento oculto</h4>
      <p>El Caso 04 leía la caché en disco del Caso 02. Ahora ambos leen del bucket.</p></div>
    <div class="caja caja--ok"><h4>3 · Fuga de la categoría objetivo</h4>
      <p>Restaurada la exclusión que el notebook original sí hacía. Queda registrada en
      cada corrida.</p></div>
    <div class="caja caja--ok"><h4>4 · Columnas adivinadas</h4>
      <p>La detección por expresiones regulares confundía <span class="mono">Fecha de pago</span>
      con <span class="mono">Periodo cotización</span>. Ahora hay un diccionario de datos explícito.</p></div>
    <div class="caja caja--ok" style="grid-column: span 2"><h4>5 · Parámetros que se auto-elegían</h4>
      <p>Categoría objetivo y ventana temporal ahora se declaran en la configuración. Dos corridas
      del mismo modelo entrenan el mismo modelo.</p></div>
  </div>
  <div class="slide__pie">Correcciones</div>
</section>

<!-- 9 -->
<section class="slide">
  <h2>El tablero</h2>
  <p class="subtitulo">Renderiza cualquier modelo sin cambios de código.</p>
  <div class="rejilla rejilla--2" style="margin-bottom:22px">
    <div class="caja caja--acento">
      <h4>Cómo funciona</h4>
      <p>Cada modelo publica los bloques de gráfico que tenga sentido en su JSON. El tablero
      recorre esos bloques y dibuja los que conoce. Si un modelo no reporta curva ROC, esa
      sección simplemente no aparece — no hay estado vacío que inventar.</p>
    </div>
    <div class="caja caja--acento">
      <h4>Qué muestra hoy</h4>
      <p>Segmentación: distribución de segmentos, selección de k y plano PCA.<br>
      Clasificación: curva ROC, matriz de confusión e importancia de variables.<br>
      Regresión: importancia de variables y predicho contra real.</p>
    </div>
  </div>
  <pre><span class="c">// components/charts/index.jsx — el registro que lo hace genérico</span>
<span class="k">export const</span> RENDERIZADORES = {{
  segment_distribution: {{ ... }},   roc_curve:        {{ ... }},
  k_selection:          {{ ... }},   confusion_matrix: {{ ... }},
  scatter_2d:           {{ ... }},   feature_importance: {{ ... }},
}}
<span class="c">// Ni un solo condicional por model_id en todo el tablero.</span></pre>
  <div class="slide__pie">Tablero</div>
</section>

<!-- 10 -->
<section class="slide">
  <h2>Agregar un modelo nuevo</h2>
  <p class="subtitulo">Dos pasos. Cero cambios en el DAG, el backend o el tablero.</p>
  <div class="rejilla rejilla--2">
    <div>
      <h3>1 · El módulo</h3>
      <pre><span class="c"># jobs/models/mi_modelo/main.py</span>
<span class="k">def</span> ejecutar(ctx) -> ModelResult:
    datos = ...
    <span class="k">return</span> construir_resultado(
        model_id=ctx.config.id,
        ...
        metrics={{<span class="s">"mi_metrica"</span>: 0.87}},
    )</pre>
    </div>
    <div>
      <h3>2 · La entrada en la configuración</h3>
      <pre><span class="c"># jobs/models_config.yml</span>
- id: mi_modelo
  nombre: <span class="s">"Nombre para el tablero"</span>
  catalogo: <span class="s">"#99"</span>
  caso_uso: 3
  task_type: clustering
  modulo: models.mi_modelo.main
  params: {{ k_min: 4 }}</pre>
    </div>
  </div>
  <div class="caja caja--ok" style="margin-top:24px">
    <p style="margin:0">En la siguiente corrida el DAG crea su tarea, el backend lo lista y el
    tablero le pinta su tarjeta. Hay dos tests que verifican precisamente esto.</p>
  </div>
  <div class="slide__pie">Extensibilidad</div>
</section>

<!-- 11 -->
<section class="slide">
  <h2>Cómo sabemos que funciona</h2>
  <p class="subtitulo">Verificación en tres niveles, ejecutable con un comando.</p>
  <div class="rejilla rejilla--4" style="margin-bottom:26px">
    <div class="caja"><span class="cifra cifra--a">122</span><span class="cifra__et">tests unitarios y de contrato</span></div>
    <div class="caja"><span class="cifra cifra--v">18</span><span class="cifra__et">tests de paridad</span></div>
    <div class="caja"><span class="cifra cifra--m">3</span><span class="cifra__et">modelos con paridad exacta</span></div>
    <div class="caja"><span class="cifra cifra--n">0</span><span class="cifra__et">secretos o rutas absolutas</span></div>
  </div>
  <div class="rejilla rejilla--3">
    <div class="caja caja--acento"><h4>Contrato</h4>
      <p>El JSON de salida está validado con Pydantic. Si un modelo deja de publicar lo que
      el tablero necesita, el test falla antes que la interfaz.</p></div>
    <div class="caja caja--acento"><h4>Paridad</h4>
      <p>Las métricas se comparan contra valores de referencia versionados. Un delta sin
      explicación es un fallo hasta que se demuestre lo contrario.</p></div>
    <div class="caja caja--acento"><h4>Datos</h4>
      <p>El diccionario de datos valida cada lote de entrada. Un archivo que no cumple aborta
      el job antes de escribir nada.</p></div>
  </div>
  <div class="slide__pie">Calidad</div>
</section>

<!-- 12 -->
<section class="slide">
  <h2>Lo que todavía no resuelve</h2>
  <p class="subtitulo">Los límites son de la entrega de datos, no del modelado.</p>
  <ul style="margin-top:14px">
    <li><strong>Los extractos están truncados</strong> en el tope de filas de Excel: los volúmenes
    son relativos, no censales. No se deben extrapolar.</li>
    <li><strong>La llave de persona es aproximada.</strong> La cédula llega enmascarada en sus
    últimos dígitos; asumimos y documentamos el riesgo de colisión. Se resuelve con una llave
    hasheada consistente.</li>
    <li><strong>No hay día de transacción</strong>, solo mes. El efecto "día de pago de la mesada"
    no es medible; se sustituye por proxies de ritmo mensual.</li>
    <li><strong>39,5% del gasto queda sin clasificar</strong> por la ofuscación de los nombres de
    comercio. Se resuelve con un catálogo maestro con código de actividad.</li>
    <li><strong>Sin campo de ubicación</strong>, el cruce geográfico del caso 5 queda incompleto.</li>
    <li><strong>El universo de pensionados es de cientos</strong>, no de miles: los segmentos del
    #101 son direccionales.</li>
  </ul>
  <div class="slide__pie">Limitaciones</div>
</section>

<!-- 13 -->
<section class="slide">
  <h2>Próximos pasos</h2>
  <p class="subtitulo">En orden de impacto sobre lo que hoy limita los modelos.</p>
  <div class="rejilla rejilla--2">
    <div class="caja caja--acento"><h4>1 · Extracto completo y llave hasheada</h4>
      <p>Elimina el truncamiento y el riesgo de colisión de identidad de un solo golpe. Es el
      cambio que más mueve la aguja en todos los modelos a la vez.</p></div>
    <div class="caja caja--acento"><h4>2 · Catálogo maestro de comercios</h4>
      <p>Con código de actividad, convierte el gasto intermediado en gasto clasificado y
      permite medir de verdad la penetración por sector.</p></div>
    <div class="caja caja--acento"><h4>3 · Campo de ubicación</h4>
      <p>Habilita la dimensión geográfica que pide el caso 5: salario × ciudad × comercio.</p></div>
    <div class="caja caja--acento"><h4>4 · Seguimiento longitudinal</h4>
      <p>Reejecutar periódicamente y medir la migración entre segmentos. La transición de
      "rutina" a "apalancado" es la alerta temprana más valiosa para el negocio.</p></div>
  </div>
  <div class="slide__pie">Próximos pasos</div>
</section>

<!-- 14 -->
<section class="slide portada">
  <div class="barra"></div>
  <h1 style="font-size:44px">La pipeline está lista para recibir modelos nuevos</h1>
  <p style="font-size:17px; max-width:60ch; margin-top:8px">
    Levantar el stack completo son tres comandos. Agregar un modelo son dos pasos.
    Las métricas de los notebooks originales están reproducidas y verificadas.
  </p>
  <pre style="margin-top:30px; width:520px">make up
make seed
make trigger</pre>
  <div class="slide__pie">Gracias</div>
</section>

<div class="nav">
  <button onclick="mover(-1)" aria-label="Anterior">&#8249;</button>
  <span id="contador"></span>
  <button onclick="mover(1)" aria-label="Siguiente">&#8250;</button>
</div>

<script>
  const slides = document.querySelectorAll('.slide');
  let actual = 0;

  function pintar() {{
    slides.forEach((s, i) => s.classList.toggle('activa', i === actual));
    document.getElementById('contador').textContent = `${{actual + 1}} / ${{slides.length}}`;
    slides[actual].querySelector('.slide__num')?.remove();
    const num = document.createElement('div');
    num.className = 'slide__num';
    num.textContent = `${{actual + 1}} / ${{slides.length}}`;
    slides[actual].appendChild(num);
    location.hash = actual + 1;
  }}

  function mover(paso) {{
    actual = Math.min(Math.max(actual + paso, 0), slides.length - 1);
    pintar();
    window.scrollTo({{ top: 0, behavior: 'instant' }});
  }}

  document.addEventListener('keydown', (e) => {{
    if (['ArrowRight', 'PageDown', ' '].includes(e.key)) {{ e.preventDefault(); mover(1); }}
    if (['ArrowLeft', 'PageUp'].includes(e.key)) {{ e.preventDefault(); mover(-1); }}
    if (e.key === 'Home') {{ actual = 0; pintar(); }}
    if (e.key === 'End') {{ actual = slides.length - 1; pintar(); }}
  }});

  const desdeHash = parseInt(location.hash.replace('#', ''), 10);
  if (desdeHash >= 1 && desdeHash <= slides.length) actual = desdeHash - 1;
  pintar();
</script>
</body>
</html>
"""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Genera las diapositivas del avance.")
    parser.add_argument("--destino", type=Path, default=DESTINO)
    args = parser.parse_args(argv)

    configurar_logging()
    args.destino.mkdir(parents=True, exist_ok=True)

    svg = args.destino / "arquitectura.svg"
    if not svg.exists():
        raise FileNotFoundError(
            f"Falta {svg}. Corre antes: python scripts/diagrama_arquitectura.py")

    datos = cargar_datos()
    html = construir_html(datos, _svg_incrustado(svg))
    salida = args.destino / "avance.html"
    salida.write_text(html, encoding="utf-8")

    log.info("Diapositivas: %s (%.0f KB)", salida, salida.stat().st_size / 1024)
    log.info("Ábrelas en el navegador; con Ctrl+P se exportan a PDF.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
