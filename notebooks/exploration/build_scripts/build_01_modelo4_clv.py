# -*- coding: utf-8 -*-
"""Construye Models/01_Modelo4_CLV_Clustering.ipynb — Caso de uso 5, Modelo #4:
Segmentación de clientes por valor (CLV). Iteración 2: llave por cédula (cruza las
3 fuentes), recibido/enviado sin cuenta propia, enriquecido con IBC declarado."""
import nbformat as nbf
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
nb = nbf.v4.new_notebook()
cells = []
md = lambda s: cells.append(nbf.v4.new_markdown_cell(s))
code = lambda s: cells.append(nbf.v4.new_code_cell(s))

md("""# 01 · Modelo #4 — Segmentación de clientes por valor (CLV)

**Caso de uso 5** · **Sector:** Banca · **Analítica:** Clustering · **Solo ACH:** Sí
**Metodología:** CRISP-DM · Fases 3–4 · **Depende de:** `00_EDA_Preparacion_Datos.ipynb`

### Objetivo de negocio
Agrupar a las personas por su **valor transaccional** para que un banco identifique a sus clientes más
valiosos (retención, cross-sell) y a los de bajo valor o en fuga. Se construye un **CLV-proxy** sobre la señal
que ACH observa: dinero **realmente** movido (recibido + enviado + gasto PSE), frecuencia, permanencia y
recencia — enriquecido con el **ingreso declarado** (IBC de Seguridad Social).

**Empresas objetivo:** Bancolombia, Davivienda, Banco de Bogotá, BBVA Colombia, Nequi, Daviplata,
Banco Popular, Banco Caja Social, Lulo Bank.

### Novedades de esta iteración
- **Llave por cédula** → cruza las 3 fuentes (entra el IBC declarado como feature).
- **Cuenta propia excluida:** las transferencias entre cuentas de la misma persona **no** cuentan como
  ingreso ni gasto (antes inflaban el valor).
- **K = 4** para segmentos más específicos (alto valor formal vs informal, gastador solo PSE, bajo valor).

> ⚠️ Limitaciones: truncamiento de Transferencias/PSE (volúmenes relativos, no censales); sin campo de ciudad.""")

md("## 1 · Configuración")
code('''import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

ROOT = Path.cwd()
if not (ROOT / "scripts").exists():
    ROOT = ROOT.parent
sys.path.insert(0, str(ROOT / "scripts"))
import ach_pipeline as ap   # noqa: E402

sns.set_theme(style="whitegrid", palette="deep")
plt.rcParams.update({"figure.dpi": 100, "figure.figsize": (11, 4.2),
                     "axes.titlesize": 12, "axes.titleweight": "bold"})
pd.set_option("display.max_columns", 50)
pd.set_option("display.float_format", lambda v: f"{v:,.2f}")
RNG = 0
cop = mticker.FuncFormatter(lambda x, _: f"${x/1e6:,.1f}M")
print("Listo")''')

md("""## 2 · Construcción de features (nivel persona)

`ap.build_clv_features()` arma la tabla persona × features de valor desde las 3 fuentes cruzadas por cédula.""")
code('''clv = ap.build_clv_features()
print(f"Personas: {len(clv):,}")
cols_modelo = ["recibido_total", "enviado_total", "gasto_total", "throughput",
               "n_transacciones", "meses_activos", "recencia_meses",
               "transfiya_share", "gasto_financiero_share", "ibc_ss", "clv_proxy"]
display(clv[cols_modelo].describe(percentiles=[.25, .5, .75, .95]).T)
print("Composición por tipo de persona (SS):")
print(clv["tipo_persona"].value_counts().to_string())''')

md("### 2.1 · Distribuciones")
code('''fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
axes[0].hist(np.log1p(clv["throughput"]), bins=60, color="#4c72b0", edgecolor="white")
axes[0].set_title("log(throughput) — dinero real movido"); axes[0].set_xlabel("log(1+COP)")
axes[1].hist(np.log1p(clv["ibc_ss"]), bins=60, color="#55a868", edgecolor="white")
axes[1].set_title("log(IBC declarado)"); axes[1].set_xlabel("log(1+COP)")
axes[2].hist(clv["recencia_meses"], bins=18, color="#dd8452", edgecolor="white")
axes[2].set_title("Recencia (meses desde última actividad)"); axes[2].set_xlabel("meses")
plt.tight_layout(); plt.show()''')

md("""## 3 · Preprocesamiento
Montos y conteos con `log1p`; estandarización con `StandardScaler`.""")
code('''feat_log = ["recibido_total", "enviado_total", "gasto_total", "throughput",
            "n_transacciones", "ibc_ss", "clv_proxy"]
X = clv[cols_modelo].copy()
for c in feat_log:
    X[c] = np.log1p(X[c])
Xs = StandardScaler().fit_transform(X.fillna(0.0))
print("Matriz de modelado:", Xs.shape)''')

md("""## 4 · Selección del número de segmentos (k)
Se explora **k ≥ 4**: k=2–3 maximizan la silueta pero son cortes gruesos; el negocio pide segmentos más
específicos, así que se parte de 4.""")
code('''ks = range(4, 9)
inercia, sil = [], []
muestra = np.random.RandomState(RNG).choice(len(Xs), min(8000, len(Xs)), replace=False)
for k in ks:
    km = KMeans(n_clusters=k, n_init=10, random_state=RNG).fit(Xs)
    inercia.append(km.inertia_); sil.append(silhouette_score(Xs[muestra], km.labels_[muestra]))
fig, axes = plt.subplots(1, 2, figsize=(13, 3.8))
axes[0].plot(list(ks), inercia, "o-", color="#4c72b0"); axes[0].set_title("Codo (inercia)"); axes[0].set_xlabel("k")
axes[1].plot(list(ks), sil, "o-", color="#55a868"); axes[1].set_title("Silueta media"); axes[1].set_xlabel("k")
plt.tight_layout(); plt.show()
K = list(ks)[int(np.argmax(sil))]
print(f"k con mejor silueta (k>=4): {K}  (silueta = {max(sil):.3f})")''')

md("## 5 · Segmentación final (K-Means)")
code('''km = KMeans(n_clusters=K, n_init=20, random_state=RNG).fit(Xs)
clv["cluster"] = km.labels_
perfil = clv.groupby("cluster").agg(
    personas=("person_id", "size"),
    recibido=("recibido_total", "median"), gasto_pse=("gasto_total", "median"),
    throughput=("throughput", "median"), ibc=("ibc_ss", "median"),
    n_tx=("n_transacciones", "median"), meses_activos=("meses_activos", "median"),
    recencia=("recencia_meses", "median"), clv_proxy=("clv_proxy", "median"))
display(perfil.round(0))''')

md("""### 5.1 · Nombres de negocio
Etiqueta por regla sobre el perfil (no por número de cluster). Distingue alto valor **con ingreso formal**
(IBC>0) vs **informal** (sin PILA), gastador solo por PSE, y bajo valor / en fuga.""")
code('''def _es_pse(r):  return r["gasto_pse"] > 0 and r["recibido"] < 0.15 * r["gasto_pse"]
def _es_fuga(r): return r["recencia"] >= 2 or r["meses_activos"] <= 6

# Clusters "activos" (ni solo-PSE ni en fuga), ordenados por throughput:
# el de mayor volumen es el de ALTO valor; el resto, valor medio.
activos = [cl for cl in perfil.index if not _es_pse(perfil.loc[cl]) and not _es_fuga(perfil.loc[cl])]
orden = perfil.loc[activos].sort_values("throughput", ascending=False).index.tolist()

def nombrar(cl):
    r = perfil.loc[cl]
    if _es_pse(r):
        return "Gastador digital (solo PSE)"
    if _es_fuga(r):
        return "Bajo valor / en fuga"
    nivel = "Alto valor" if orden and cl == orden[0] else "Valor medio"
    return f"{nivel} · {'ingreso formal' if r['ibc'] > 0 else 'informal'}"

nombres = {cl: nombrar(cl) for cl in perfil.index}
vistos = {}
for cl in perfil.sort_values("throughput", ascending=False).index:
    n = nombres[cl]
    if n in vistos.values():
        n = f"{n} ({perfil.loc[cl, 'personas']:,})"
    vistos[cl] = n
clv["segmento"] = clv["cluster"].map(vistos)
perfil["segmento"] = perfil.index.map(vistos)
display(perfil.set_index("segmento").round(0))''')

md("## 6 · Visualización")
code('''Xstd = pd.DataFrame(Xs, columns=cols_modelo); Xstd["segmento"] = clv["segmento"].values
heat = Xstd.groupby("segmento").mean()
fig, ax = plt.subplots(figsize=(11, 0.7 * len(heat) + 1.5))
sns.heatmap(heat, annot=True, fmt=".2f", cmap="RdBu_r", center=0, ax=ax,
            cbar_kws={"label": "media estandarizada (z)"})
ax.set_title("Perfil comparativo de segmentos (features estandarizadas)"); ax.set_ylabel("")
plt.tight_layout(); plt.show()''')
code('''pca = PCA(n_components=2, random_state=RNG).fit(Xs); proj = pca.transform(Xs)
idx = np.random.RandomState(RNG).choice(len(proj), min(12000, len(proj)), replace=False)
fig, ax = plt.subplots(figsize=(8, 6))
for seg in sorted(clv["segmento"].unique()):
    m = (clv["segmento"].values == seg)[idx]
    ax.scatter(proj[idx][m, 0], proj[idx][m, 1], s=6, alpha=.35, label=seg)
ax.set_title(f"Segmentos CLV en el plano PCA ({pca.explained_variance_ratio_.sum():.0%} var.)")
ax.set_xlabel("PC1"); ax.set_ylabel("PC2"); ax.legend(markerscale=2, fontsize=8)
plt.tight_layout(); plt.show()''')

md("""### 6.1 · Formalidad por segmento
Qué tan formal es cada segmento (composición por tipo de persona de Seguridad Social).""")
code('''comp = (pd.crosstab(clv["segmento"], clv["tipo_persona"], normalize="index") * 100).round(1)
display(comp)''')

md("""## 7 · Perfilamiento comercial: ¿con qué banco opera cada segmento?
Con el decodificador de `Entidad autorizadora` (solo pagos con banco identificable).""")
code('''pse_raw = ap.add_person_key(ap.load_raw("pse").drop_duplicates(), "pse")
ent = ap.decode_entidad_autorizadora(pse_raw["Entidad autorizadora"])
pse_raw = pse_raw.assign(entidad=ent["entidad"].values, conf=ent["entidad_confianza"].values)
pse_raw = pse_raw.merge(clv[["person_id", "segmento"]], on="person_id", how="inner")
usable = pse_raw[pse_raw["conf"].isin(["alta", "media"])]
top = (usable.groupby(["segmento", "entidad"]).size().rename("n").reset_index()
       .sort_values(["segmento", "n"], ascending=[True, False]))
for seg in sorted(usable["segmento"].unique()):
    t = top[top["segmento"] == seg].head(4)
    print(f"{seg:32} -> " + " · ".join(f"{r.entidad} ({r.n:,})" for r in t.itertuples()))''')

md("## 8 · Valor por segmento y conclusiones")
code('''resumen = clv.groupby("segmento").agg(
    personas=("person_id", "size"), clv_total=("clv_proxy", "sum"),
    clv_medio=("clv_proxy", "median"), throughput_medio=("throughput", "median"))
resumen["%_personas"] = (resumen["personas"] / len(clv) * 100).round(1)
resumen["%_clv_total"] = (resumen["clv_total"] / resumen["clv_total"].sum() * 100).round(1)
display(resumen.sort_values("clv_total", ascending=False).round(0))
fig, ax = plt.subplots(figsize=(9, 4)); r = resumen.sort_values("clv_total")
ax.barh(r.index, r["clv_total"], color="#4c72b0")
ax.set_title("CLV-proxy total capturado por segmento"); ax.xaxis.set_major_formatter(cop)
plt.tight_layout(); plt.show()''')

md("""### Lectura de negocio y acción

- **Alto valor · ingreso formal** — clientes premium con salario declarado alto: retención prioritaria,
  inversión y crédito de alto cupo.
- **Alto valor · informal** — mueven mucho pero sin ingreso formal declarado: candidatos a formalización y a
  productos de ingreso variable.
- **Gastador digital (solo PSE)** — gastan mucho por PSE sin transferencias entrantes visibles: captar nómina.
- **Valor medio** — masa estable; automatización y up-sell.
- **Bajo valor / en fuga** — reactivación de bajo costo o des-priorización.

**Próximos pasos:** resolver el truncamiento (volúmenes censales); reponderar el CLV-proxy con márgenes
reales; cruzar con el modelo de ciclo de vida para una matriz **valor × etapa**.""")
code('''sal = ROOT / "Data" / "processed" / "segmentos_clv.parquet"
clv[["person_id", "cluster", "segmento", "tipo_persona", "clv_proxy", "throughput",
     "ibc_ss", "meses_activos", "recencia_meses"]].to_parquet(sal, index=False)
print("Guardado:", sal)''')

nb["cells"] = cells
nb["metadata"]["kernelspec"] = {"display_name": "Python 3", "language": "python", "name": "python3"}
nb["metadata"]["language_info"] = {"name": "python", "version": "3.13"}
out = ROOT / "Models" / "01_Modelo4_CLV_Clustering.ipynb"
out.parent.mkdir(exist_ok=True)
nbf.write(nb, out)
print(f"Notebook escrito: {out} ({len(cells)} celdas)")
