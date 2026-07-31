# Contrato de resultados

Lo que la pipeline escribe y lo que el backend y el frontend consumen. Mientras un
modelo respete este esquema, el tablero lo renderiza **sin cambios de código**.

La definición ejecutable está en `jobs/common/results.py` (modelos Pydantic). Este
documento es su lectura para quien construye la API y el tablero.

---

## Rutas en el bucket

```
results/index.json                        resumen de la última corrida
results/<model_id>/latest.json            último resultado del modelo
results/<model_id>/<run_id>.json          histórico por corrida
results/<model_id>/<run_id>/model.joblib  modelo entrenado
results/<model_id>/<run_id>/assignments.parquet   persona → segmento o score
```

`run_id` es el `dag_run_id` de Airflow saneado (sin `:` ni `+`). Es el mismo para todos
los modelos de una corrida: eso es lo que permite `GET /api/runs/{run_id}`.

---

## `results/<model_id>/<run_id>.json`

```jsonc
{
  "schema_version": "1.0",
  "model_id": "caso05_pensionados",
  "model_name": "Segmentación de pensionados por consumo",
  "catalog_ref": "#101",
  "use_case": 5,
  "task_type": "clustering",        // clustering | regression | classification | scoring

  "run_id": "manual__2026-07-31T010000Z",
  "status": "success",              // success | failed
  "started_at": "2026-07-31T01:00:00Z",
  "finished_at": "2026-07-31T01:00:42Z",
  "duration_seconds": 42.1,

  "dataset": {
    "uri": "curated/dataset.parquet",
    "manifest_hash": "acfe795f...",
    "rows": 360,
    "window": ["2025-01", "2026-06"],
    "lineage": "cedula-v1"
  },

  "params":  { "k": 4, "algoritmo": "KMeans", "escalador": "RobustScaler", "...": "..." },
  "metrics": { "silhouette": 0.2713, "davies_bouldin": 1.2014, "n_entities": 360 },

  "segments": [
    { "id": 2, "label": "Pensionado de alto consumo (supera su mesada)",
      "n": 44, "share": 0.1222, "profile": { "gasto_mensual": 3077034.4, "tasa_consumo": 4.91 } }
  ],

  "charts":    { /* ver abajo */ },
  "artifacts": { "model_uri": "results/...", "assignments_uri": "results/..." },
  "notes":     ["Limitaciones y supuestos, en texto plano"],
  "error":     null
}
```

### Reglas que el frontend puede dar por hechas

1. **`metrics` es plano**: `nombre → número`, siempre finito. Nunca `NaN` ni `null`.
   Se puede renderizar como tarjetas sin conocer los nombres de antemano.
2. **`segments` solo viene en clustering.** La suma de `n` es `dataset.rows` y la de
   `share` es 1.
3. **`charts` es opcional y por bloques.** Si un bloque no está, esa sección **no se
   dibuja**. No hay estado vacío que inventar.
4. **`status: "failed"`** trae `error` y `metrics` vacío. El tablero debe mostrar el
   fallo, no un hueco.
5. **Campos nuevos no rompen nada**: `charts` admite bloques desconocidos y el frontend
   ignora lo que no sabe dibujar.

### Bloques de `charts` y quién los emite hoy

| Bloque | Forma | Lo emiten |
|---|---|---|
| `segment_distribution` | `[{id, label, n, share, profile}]` | los 4 de clustering |
| `k_selection` | `[{k, silhouette, inertia, davies_bouldin}]` | los 4 de clustering |
| `scatter_2d` | `{points: [{x, y, label}], explained_variance}` | los 4 de clustering |
| `feature_importance` | `[{feature, importance}]` | #6, #17, #55 |
| `residuals` | `{actual: [...], predicted: [...]}` | #6 |
| `roc_curve` | `{fpr: [...], tpr: [...], auc}` | #17, #55 |
| `confusion_matrix` | `{labels: [...], matrix: [[...]]}` | #17, #55 |

`scatter_2d` viene submuestreado a 2.000 puntos: el JSON no debe cargarle 30.000 puntos
al navegador.

---

## `results/index.json`

Lo genera la tarea de consolidación. **`GET /api/models` debería leer esto y no recorrer
el bucket**: así el endpoint es una lectura y no N.

```jsonc
{
  "schema_version": "1.0",
  "run_id": "manual__2026-07-31T010000Z",
  "generated_at": "2026-07-31T01:05:00Z",
  "total_models": 7, "successful": 7, "failed": 0,
  "models": [
    { "model_id": "caso05_pensionados", "model_name": "...", "catalog_ref": "#101",
      "use_case": 5, "task_type": "clustering", "status": "success",
      "run_id": "...", "finished_at": "...", "duration_seconds": 42.1,
      "metrics": { "silhouette": 0.2713 },
      "latest_uri": "results/caso05_pensionados/latest.json", "error": null }
  ]
}
```

---

## Notas para el backend (Fase 6)

- **Leer del bucket con `common/storage.py`.** Ya funciona igual contra MinIO y contra
  AWS; solo cambia el endpoint.
- **Caché en memoria con TTL corto** (`ACH_CACHE_TTL`, 30 s por defecto). Los resultados
  solo cambian cuando corre el DAG.
- **Ninguna credencial de MinIO llega al navegador.** Los artefactos (`.joblib`,
  `.parquet`) se sirven por el backend o con URL prefirmada de vida corta.
- **`GET /api/models/{id}/runs`** se resuelve listando `results/<model_id>/*.json` y
  descartando `latest.json`.

## Notas para el frontend (Fase 7)

- Iterar `charts` y renderizar lo que exista. Nada de `if (model_id === ...)`.
- Los siete modelos de hoy cubren los tres `task_type`, así que las secciones de ROC y
  matriz de confusión **sí tienen datos** (vía #17 y #55).
- Si `index.json` no existe todavía, mostrar el estado vacío: la pipeline aún no ha
  corrido. No inventar valores de ejemplo.
