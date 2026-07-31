.DEFAULT_GOAL := help
SHELL := /bin/bash

COMPOSE := docker compose
JOBS_IMAGE ?= ach-jobs:latest
PYTHON ?= python
DAG ?= pipeline_modelos

# Las rutas de los montajes no se convierten en Git Bash (Windows).
export MSYS_NO_PATHCONV := 1

.PHONY: help up up-app down restart reset ps logs build build-jobs \
        seed seed-demo trigger dag exportar demo-estatico test test-unit parity lint clean

help:  ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------------------------------- #
# Infraestructura                                                              #
# --------------------------------------------------------------------------- #
up: build-jobs  ## Levanta todo el stack: Airflow, MinIO, backend y frontend
	@test -f .env || (echo "Falta .env — copiando desde .env.example" && cp .env.example .env)
	@# La imagen de Airflow la comparten cuatro servicios: se construye una sola vez
	@# para que no colisionen construyéndola en paralelo.
	$(COMPOSE) build airflow-init
	$(COMPOSE) up -d
	@echo ""
	@echo "  Tablero  → http://localhost:$${FRONTEND_PORT:-5173}"
	@echo "  API      → http://localhost:$${BACKEND_PORT:-8000}/docs"
	@echo "  Airflow  → http://localhost:$${AIRFLOW_PORT:-8080}"
	@echo "  MinIO    → http://localhost:$${MINIO_CONSOLE_PORT:-9001}"
	@echo ""
	@echo "  Siguiente paso: make seed-demo (datos de ejemplo) y make trigger"

up-app: up  ## Alias de up (backend y frontend ya vienen incluidos)

down:  ## Apaga el stack conservando los datos
	$(COMPOSE) down

restart: down up  ## Reinicia el stack

reset:  ## Apaga y BORRA los volúmenes (Postgres y MinIO incluidos)
	$(COMPOSE) down -v

ps:  ## Estado de los servicios
	$(COMPOSE) ps

logs:  ## Logs de todos los servicios
	$(COMPOSE) logs -f --tail=100

# --------------------------------------------------------------------------- #
# Imágenes                                                                     #
# --------------------------------------------------------------------------- #
build-jobs:  ## Construye la imagen de los jobs
	docker build -t $(JOBS_IMAGE) ./jobs

build: build-jobs  ## Construye todas las imágenes
	$(COMPOSE) build

# --------------------------------------------------------------------------- #
# Datos                                                                        #
# --------------------------------------------------------------------------- #
seed:  ## Sube los XLSX reales de ACH_DATA_DIR al bucket raw
	@test -n "$$ACH_DATA_DIR" || grep -q '^ACH_DATA_DIR=' .env 2>/dev/null \
		|| (echo "Define ACH_DATA_DIR en .env o en el entorno" && exit 1)
	docker run --rm --network $${ACH_DOCKER_NETWORK:-ach_net} \
		--env-file .env \
		-e ACH_S3_ENDPOINT=http://minio:9000 \
		-e ACH_S3_ACCESS_KEY=$${MINIO_ROOT_USER:-minioadmin} \
		-e ACH_S3_SECRET_KEY=$${MINIO_ROOT_PASSWORD:-minioadmin} \
		-e ACH_DATA_DIR=/datos \
		-v "$${ACH_DATA_DIR}:/datos:ro" \
		$(JOBS_IMAGE) processing.seed

seed-demo:  ## Sube un extracto SINTÉTICO al bucket raw (no requiere los datos reales)
	docker run --rm --network $${ACH_DOCKER_NETWORK:-ach_net} \
		-e ACH_S3_ENDPOINT=http://minio:9000 \
		-e ACH_S3_ACCESS_KEY=$${MINIO_ROOT_USER:-minioadmin} \
		-e ACH_S3_SECRET_KEY=$${MINIO_ROOT_PASSWORD:-minioadmin} \
		-v "$$(pwd)/scripts:/scripts:ro" \
		--entrypoint python $(JOBS_IMAGE) /scripts/seed_minio.py $(ARGS)

# --------------------------------------------------------------------------- #
# Pipeline                                                                     #
# --------------------------------------------------------------------------- #
trigger:  ## Dispara el DAG desde la consola
	$(COMPOSE) exec -T airflow-scheduler airflow dags unpause $(DAG)
	$(COMPOSE) exec -T airflow-scheduler airflow dags trigger $(DAG)
	@echo "DAG disparado. Sigue el avance en http://localhost:$${AIRFLOW_PORT:-8080}"

dag: trigger  ## Alias de trigger

# --------------------------------------------------------------------------- #
# Demo para cliente                                                            #
# --------------------------------------------------------------------------- #
exportar:  ## Exporta los resultados del bucket como JSON estáticos
	ACH_S3_ENDPOINT=http://localhost:$${MINIO_API_PORT:-9000} \
	ACH_S3_ACCESS_KEY=$${MINIO_ROOT_USER:-minioadmin} \
	ACH_S3_SECRET_KEY=$${MINIO_ROOT_PASSWORD:-minioadmin} \
	PYTHONPATH=jobs $(PYTHON) scripts/exportar_estatico.py

demo-estatico: exportar  ## Genera dist-demo/: tablero autocontenido, listo para publicar
	docker run --rm -v "$$(pwd)/frontend:/app" -w /app node:22-alpine \
		sh -c "npm install --no-audit --no-fund --silent && npm run build"
	rm -rf dist-demo && cp -r frontend/dist dist-demo
	@git checkout -- frontend/public/config.js 2>/dev/null || true
	@echo ""
	@echo "  Listo: dist-demo/ — carpeta estática, sin backend ni credenciales."
	@echo "  Probar:    npx serve dist-demo"
	@echo "  Publicar:  npx netlify deploy --dir=dist-demo --prod"

# --------------------------------------------------------------------------- #
# Calidad                                                                      #
# --------------------------------------------------------------------------- #
test: test-unit  ## Ejecuta la batería de tests

test-unit:  ## Tests unitarios y de contrato (no requieren Docker ni datos)
	$(PYTHON) -m pytest tests/unit -q

parity:  ## Tests de paridad contra los notebooks originales (requiere resultados)
	$(PYTHON) -m pytest tests/parity -q -m parity

lint:  ## Revisa el estilo del código
	$(PYTHON) -m ruff check jobs backend scripts tests

clean:  ## Borra artefactos locales de Python y del almacén de desarrollo
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	find . -type d -name '*.egg-info' -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache jobs/.ruff_cache .localstore
	@echo "Limpio. Los volúmenes de Docker no se tocan: para eso está 'make reset'."
