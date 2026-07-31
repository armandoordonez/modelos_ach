.DEFAULT_GOAL := help
SHELL := /bin/bash

COMPOSE := docker compose
JOBS_IMAGE ?= ach-jobs:latest
PYTHON ?= python

.PHONY: help up down restart logs ps build build-jobs seed dag test test-unit parity lint clean reset

help:  ## Muestra esta ayuda
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) \
		| awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-14s\033[0m %s\n", $$1, $$2}'

# --------------------------------------------------------------------------- #
# Infraestructura                                                              #
# --------------------------------------------------------------------------- #
up: build-jobs  ## Levanta todo el stack (Airflow, MinIO, Postgres)
	@test -f .env || (echo "Falta .env — copiando desde .env.example" && cp .env.example .env)
	$(COMPOSE) up -d --build
	@echo ""
	@echo "  Airflow  → http://localhost:$${AIRFLOW_PORT:-8080}"
	@echo "  MinIO    → http://localhost:$${MINIO_CONSOLE_PORT:-9001}"
	@echo ""
	@echo "  Siguiente paso: make seed (sube los XLSX) y dispara el DAG pipeline_modelos."

up-app: build-jobs  ## Levanta el stack incluyendo backend y frontend
	$(COMPOSE) --profile app up -d --build

down:  ## Apaga el stack (conserva los datos)
	$(COMPOSE) --profile app down

restart: down up  ## Reinicia el stack

reset:  ## Apaga y BORRA los volúmenes (Postgres y MinIO incluidos)
	$(COMPOSE) --profile app down -v

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
# Pipeline                                                                     #
# --------------------------------------------------------------------------- #
seed:  ## Sube los XLSX de ACH_DATA_DIR al bucket raw
	@test -n "$$ACH_DATA_DIR" || (grep -q '^ACH_DATA_DIR=' .env 2>/dev/null) \
		|| (echo "Define ACH_DATA_DIR en .env o en el entorno" && exit 1)
	docker run --rm --network $${ACH_DOCKER_NETWORK:-ach_net} \
		--env-file .env \
		-e ACH_S3_ENDPOINT=http://minio:9000 \
		-e ACH_S3_ACCESS_KEY=$${MINIO_ROOT_USER:-minioadmin} \
		-e ACH_S3_SECRET_KEY=$${MINIO_ROOT_PASSWORD:-minioadmin} \
		-e ACH_DATA_DIR=/datos \
		-v "$${ACH_DATA_DIR}:/datos:ro" \
		$(JOBS_IMAGE) processing.seed

dag:  ## Dispara el DAG pipeline_modelos desde la consola
	$(COMPOSE) exec airflow-scheduler airflow dags trigger pipeline_modelos

# --------------------------------------------------------------------------- #
# Calidad                                                                      #
# --------------------------------------------------------------------------- #
test: test-unit  ## Ejecuta la batería de tests

test-unit:  ## Tests unitarios y de contrato (no requieren Docker ni datos)
	$(PYTHON) -m pytest tests/unit -q

parity:  ## Tests de paridad contra los notebooks originales (requiere dataset curado)
	$(PYTHON) -m pytest tests/parity -q -m parity

lint:  ## Revisa el estilo del código
	$(PYTHON) -m ruff check jobs tests

clean:  ## Borra artefactos locales de Python
	find . -type d -name __pycache__ -prune -exec rm -rf {} + 2>/dev/null || true
	rm -rf .pytest_cache .ruff_cache
