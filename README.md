# modelos_ach

Pipeline de modelos analíticos sobre datos de **ACH Colombia**, orquestada con Airflow y
almacenamiento tipo S3 (MinIO).

Los notebooks de exploración viven en `notebooks/exploration/` y **no se ejecutan en producción**:
son la referencia metodológica. Lo que corre es el paquete `jobs/`.

## Arquitectura

```
XLSX crudos ──▶ job de procesamiento ──▶ curated/dataset.parquet ──▶ N jobs de modelo ──▶ results/*.json
   (raw/)          (streaming)              (particionado)              (en paralelo)        (backend)
```

| Componente | Rol |
|---|---|
| **MinIO** | Almacenamiento S3 (`raw`, `curated`, `results`) |
| **Airflow** | Orquestación (`pipeline_modelos`), LocalExecutor |
| **jobs** | Imagen con el paquete de procesamiento y modelos |
| **backend** | FastAPI que sirve los resultados del bucket |
| **frontend** | Tablero de métricas |

## Arranque

```bash
cp .env.example .env      # revisar credenciales locales
make up                   # levanta todo el stack
make seed                 # sube los XLSX a raw/ (requiere ACH_DATA_DIR)
```

Airflow queda en <http://localhost:8080>, MinIO en <http://localhost:9001>.
Disparar el DAG `pipeline_modelos` desde la UI ejecuta la pipeline completa.

## Comandos

| Comando | Qué hace |
|---|---|
| `make up` / `make down` | Levanta / apaga el stack |
| `make seed` | Sube los XLSX locales al bucket `raw` |
| `make test` | Tests unitarios y de contrato |
| `make parity` | Tests de paridad contra los notebooks originales |
| `make logs` | Logs de todos los servicios |

## Modelos

Los modelos se declaran en un único archivo: `jobs/models_config.yml`.
**Agregar un modelo = crear su módulo en `jobs/models/` y añadir una entrada ahí.**
No hay que tocar el DAG, ni el backend, ni el frontend.

| id | Catálogo | Tipo | Origen |
|---|---|---|---|
| `caso02_ingresos_independientes` | #6 | regresión | Caso 02 |
| `caso04_rfm_consumidores` | #15 | clustering | Caso 04 |
| `caso04_propension_salud` | #17 | clasificación | Caso 04 |
| `caso04_propension_turismo` | #55 | clasificación | Caso 04 |
| `caso05_clv` | #4 | clustering | Caso 05 |
| `caso05_ciclo_vida` | #46 | clustering | Caso 05 |
| `caso05_pensionados` | #101 | clustering | Caso 05 |

## Rutas en el bucket

```
raw/<fecha>/<archivo>.xlsx
curated/dataset.parquet/fuente=<ss|trf|pse>/periodo=<YYYY-MM>/*.parquet
curated/dataset.parquet/_manifest.json
results/<model_id>/<run_id>.json
results/<model_id>/latest.json
results/index.json
```

## Linaje de datos

Las tres fuentes se cruzan por la **cédula ofuscada** (`Número documento`), que llega enmascarada
de forma idéntica en las tres. Es la única llave del proyecto (`lineage = cedula-v1`).

Los scripts originales de los casos 02 y 04 usaban llaves compuestas distintas; el paquete conserva
ese comportamiento bajo `lineage = legacy` **solo para verificar la paridad de la migración**.
Producción usa siempre `cedula-v1`.

## Documentación

- `Docs/Entendimiento_Negocio_y_Datos.md` — contexto de negocio y hallazgos del EDA
- `docs/CONTRATO_RESULTADOS.md` — esquema del JSON que consume el backend
- `docs/PARIDAD.md` — métricas de referencia y deltas del re-baseline
