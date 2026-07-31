# modelos_ach

Pipeline de modelos analíticos sobre datos de **ACH Colombia**, orquestada con Airflow,
con almacenamiento tipo S3 (MinIO), API en FastAPI y un tablero de métricas.

Los notebooks de exploración viven en `notebooks/exploration/` y **no se ejecutan en
producción**: son la referencia metodológica. Lo que corre es el paquete `jobs/`.

---

## Arquitectura

```
                        ┌──────────── Airflow (solo orquesta) ────────────┐
                        │  procesamiento ─▶ modelo × N ─▶ consolidación   │
                        └───────────────────────┬─────────────────────────┘
                                                │ lanza contenedores
                                                ▼
   XLSX crudos ──────▶  job de procesamiento  ──────▶  N jobs de modelo
      (raw/)            (streaming, valida)            (en paralelo, pool de 2)
                                 │                              │
                                 ▼                              ▼
                    curated/dataset.parquet          results/<model_id>/*.json
                       (particionado)                          │
                                                               ▼
                                                     backend  ──▶  tablero
                                                    (FastAPI)      (React)
```

| Servicio | Rol | Puerto |
|---|---|---|
| **frontend** | Tablero de métricas | 5173 |
| **backend** | API que sirve los resultados del bucket | 8000 |
| **airflow** | Orquestación (`api-server`, `scheduler`, `dag-processor`) | 8080 |
| **minio** | Almacenamiento S3 (`raw`, `curated`, `results`) | 9000 / 9001 |
| **postgres** | Metadatos de Airflow | interno |

**Separación deliberada de dependencias:** Airflow no tiene pandas ni scikit-learn —
solo lanza contenedores. El backend tampoco — solo lee JSON. Las librerías de datos
viven únicamente en la imagen `jobs`.

---

## Arranque

```bash
cp .env.example .env      # revisar credenciales locales
make up                   # levanta todo el stack
make seed-demo            # sube un extracto SINTÉTICO (no requiere los datos reales)
make trigger              # dispara la pipeline
```

| Servicio | URL | Credenciales |
|---|---|---|
| Tablero | <http://localhost:5173> | — |
| API | <http://localhost:8000/docs> | — |
| Airflow | <http://localhost:8080> | `admin` / `AIRFLOW_ADMIN_PASSWORD` del `.env` |
| MinIO | <http://localhost:9001> | `MINIO_ROOT_USER` / `MINIO_ROOT_PASSWORD` del `.env` |

Con los datos reales de ACH, en vez de `seed-demo`:

```bash
# ACH_DATA_DIR en .env apunta a la carpeta con los 3 XLSX
make seed
```

> **Windows sin GNU make:** usa `make.cmd` (mismo juego de comandos) desde cmd o
> PowerShell, o instala make con `choco install make`. En Git Bash, WSL, macOS y Linux
> se usa el `Makefile` normal.

---

## Comandos

| Comando | Qué hace |
|---|---|
| `make up` / `make down` | Levanta / apaga el stack |
| `make seed` | Sube los XLSX reales de `ACH_DATA_DIR` al bucket `raw` |
| `make seed-demo` | Genera y sube un extracto sintético para probar sin datos reales |
| `make trigger` | Dispara el DAG `pipeline_modelos` |
| `make logs` / `make ps` | Logs y estado de los servicios |
| `make test` | Tests unitarios y de contrato |
| `make parity` | Tests de paridad contra los notebooks originales |
| `make lint` | Estilo del código |
| `make exportar` | Exporta los resultados como JSON estáticos |
| `make demo-estatico` | Genera `dist-demo/`: tablero autocontenido para publicar |
| `make clean` | Borra artefactos locales de Python |
| `make reset` | Apaga y borra los volúmenes |

---

## Modelos

Los modelos se declaran en un único archivo: `jobs/models_config.yml`.

| id | Catálogo | Tipo | Origen |
|---|---|---|---|
| `caso02_ingresos_independientes` | #6 | regresión | Caso 02 |
| `caso04_rfm_consumidores` | #15 | clustering | Caso 04 |
| `caso04_propension_salud` | #17 | clasificación | Caso 04 |
| `caso04_propension_turismo` | #55 | clasificación | Caso 04 |
| `caso05_clv` | #4 | clustering | Caso 05 |
| `caso05_ciclo_vida` | #46 | clustering | Caso 05 |
| `caso05_pensionados` | #101 | clustering | Caso 05 |

### Agregar un modelo nuevo

**Dos pasos. Cero cambios en el DAG, el backend o el frontend.**

**1.** Crear `jobs/models/mi_modelo/main.py` con una función `ejecutar`:

```python
from common.results import ModelResult, construir_resultado
from models.base import ContextoModelo, cli_para, limpiar_metricas

def ejecutar(ctx: ContextoModelo) -> ModelResult:
    datos = ...  # leer con common.features
    return construir_resultado(
        model_id=ctx.config.id, model_name=ctx.config.nombre,
        catalog_ref=ctx.config.catalogo, use_case=ctx.config.caso_uso,
        task_type=ctx.config.task_type, run_id=ctx.run_id,
        started_at=ctx.started_at, dataset=ctx.dataset_info(len(datos)),
        metrics=limpiar_metricas({"mi_metrica": 0.87}),
    )

main = cli_para("mi_modelo")
```

**2.** Añadir su entrada al registro:

```yaml
- id: mi_modelo
  nombre: "Nombre para el tablero"
  catalogo: "#99"
  caso_uso: 3
  task_type: clustering          # clustering | regression | classification | scoring
  modulo: models.mi_modelo.main
  params: { k_min: 4 }
```

En la siguiente corrida el DAG crea su tarea, el backend lo lista y el tablero le pinta
su tarjeta. `#17` y `#55` son la prueba: comparten módulo y solo cambian el parámetro
`categoria` — el segundo no costó una línea de código.

Para publicar un gráfico, se rellena el bloque correspondiente de `charts`
(`roc_curve`, `feature_importance`, `segment_distribution`…). El tablero dibuja lo que
encuentre; lo que no venga, no se muestra. El contrato completo está en
`Docs/CONTRATO_RESULTADOS.md`.

---

## Rutas en el bucket

```
raw/<fecha>/<archivo>.xlsx
curated/dataset.parquet/fuente=<ss|trf|pse>/periodo=<YYYY-MM>/*.parquet
curated/dataset.parquet/_manifest.json
results/index.json
results/<model_id>/latest.json
results/<model_id>/<run_id>.json
results/<model_id>/<run_id>/model.joblib
```

## Linaje de datos

Las tres fuentes se cruzan por la **cédula ofuscada** (`Número documento`), que llega
enmascarada de forma idéntica en las tres. Es la única llave del proyecto
(`ACH_LINEAGE=cedula-v1`).

Los scripts originales de los casos 02 y 04 usaban llaves compuestas distintas; el
paquete conserva ese comportamiento bajo `ACH_LINEAGE=legacy` **solo para verificar la
paridad de la migración**. Producción usa siempre `cedula-v1`.

---

## Estado verificado

Corrida completa sobre los datos reales (2,83 M de filas):

| Etapa | Resultado |
|---|---|
| Procesamiento | 336 s · 743.406 / 1.048.574 / 1.048.574 filas, idénticas al EDA original |
| 7 modelos | todos en verde, 27-63 s cada uno, 2 en paralelo |
| Consolidación | 6 s · `results/index.json` con los 7 |
| Paridad | exacta en los 3 modelos del Caso 05; re-baseline documentado en Caso 02 y 04 |
| Tests | 122 unitarios + 18 de paridad |

---

## Troubleshooting

**`make up` falla construyendo la imagen de Airflow: "image already exists".**
Cuatro servicios comparten esa imagen y Compose intentaba construirla en paralelo. El
`Makefile` ya la construye una sola vez antes (`docker compose build airflow-init`); si
lo ejecutas a mano, hazlo en ese orden.

**Las tareas del DAG fallan con `httpx.ConnectError: Connection refused`.**
En Airflow 3 las tareas hablan con la API de ejecución del `api-server`. Con los
componentes en contenedores distintos hay que apuntarla por nombre de servicio; el
compose ya define `AIRFLOW__CORE__EXECUTION_API_SERVER_URL`. Si lo cambias, revisa que
apunte a `http://airflow-apiserver:8080/execution/`.

**El `DockerOperator` no puede lanzar contenedores.**
El contenedor de Airflow monta `/var/run/docker.sock` y corre como `50000:0` para tener
permiso sobre él. En Docker Desktop funciona tal cual; en Linux puede hacer falta
ajustar el gid del grupo `docker`.

**Los contenedores de jobs no encuentran `minio`.**
El `network_mode` del `DockerOperator` tiene que ser la red del compose
(`ACH_DOCKER_NETWORK`, por defecto `ach_net`); con la red por defecto no se resuelven
los nombres de servicio.

**No sé la contraseña de Airflow.**
Es `AIRFLOW_ADMIN_PASSWORD` del `.env` (por defecto `ach-demo-2026`), usuario `admin`.
Ojo: el `AIRFLOW__CORE__SIMPLE_AUTH_MANAGER_USERS: admin:admin` del compose es
*usuario:rol*, no usuario:contraseña. `airflow-init` escribe la contraseña en
`/opt/airflow/auth/passwords.json`; si se deja vacía la variable, Airflow genera una
aleatoria que cambia en cada recreación.

**El tablero muestra "No se pudo hablar con la API".**
Comprueba `make ps` y <http://localhost:8000/health>. Si el backend responde pero el
navegador no, revisa `ACH_CORS_ORIGINS`: debe incluir el origen del tablero.

**El tablero dice "Todavía no hay resultados".**
Es el estado vacío correcto: el bucket no tiene nada. Corre `make seed-demo` y
`make trigger`.

**El procesamiento aborta con "no cumple el diccionario de datos".**
Es el comportamiento buscado: el archivo de entrada no es el extracto esperado. El
mensaje dice qué columna falla y por qué. El diccionario está en `jobs/common/schema.py`.

**Se queda sin memoria con varios modelos a la vez.**
El pool `modelos` limita a 2 simultáneos (`ACH_POOL_SLOTS`). Con 8 GB asignados a
Docker es el máximo sensato: cada job carga las tablas persona-mes de las tres fuentes.

**En Git Bash los montajes de Docker apuntan a rutas raras** (`C:/Program Files/Git/...`).
Es la conversión de rutas de MSYS. El `Makefile` ya exporta `MSYS_NO_PATHCONV=1`.

---

## Demo para cliente

Para enseñar el tablero sin montar infraestructura:

```bash
make demo-estatico        # deja dist-demo/: HTML + JSON, sin backend ni credenciales
npx serve dist-demo       # probarlo en local
npx netlify deploy --dir=dist-demo --prod   # publicarlo
```

Los datos quedan congelados en el momento de la exportación y los artefactos binarios
no se exportan. **Antes de publicar en una URL abierta:** lo que se exporta son métricas
reales sobre datos ofuscados de ACH. Si la demo va a ser pública, exporta desde una
corrida con `make seed-demo` (datos sintéticos).

### Presentación

```bash
python scripts/diagrama_arquitectura.py    # PDF, SVG y PNG del diagrama
python scripts/generar_diapositivas.py     # Docs/presentacion/avance.html
```

Las diapositivas leen las cifras de los resultados reales, así que se regeneran después
de cada corrida y siguen siendo ciertas. Con Ctrl+P se exportan a PDF.

## Documentación

- `Docs/Entendimiento_Negocio_y_Datos.md` — contexto de negocio y hallazgos del EDA
- `Docs/CONTRATO_RESULTADOS.md` — esquema del JSON que consumen backend y tablero
- `Docs/PARIDAD.md` — métricas de referencia y deltas del re-baseline
- `Docs/presentacion/` — diapositivas del avance y diagrama de arquitectura
