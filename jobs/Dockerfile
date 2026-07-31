# Imagen de los jobs de la pipeline: procesamiento y modelos.
# Multi-stage para que la imagen final no cargue con las herramientas de build.
# NO lleva Jupyter ni nada de notebooks: en producción no se ejecutan cuadernos.

# --- Etapa 1: construcción del wheel ---------------------------------------
FROM python:3.12-slim AS builder

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /build
RUN pip install --no-cache-dir build==1.2.2

COPY pyproject.toml ./
COPY common ./common
COPY processing ./processing
COPY models ./models

RUN python -m build --wheel --outdir /dist


# --- Etapa 2: runtime -------------------------------------------------------
FROM python:3.12-slim AS runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ACH_MODELS_CONFIG=/app/models_config.yml

# libgomp lo necesitan scikit-learn y el gradient boosting para el paralelismo.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 50000 ach

COPY --from=builder /dist/*.whl /tmp/
RUN pip install --no-cache-dir /tmp/*.whl && rm -f /tmp/*.whl

# El registro de modelos viaja aparte del wheel: es configuración, no código, y así
# se puede sobrescribir montándolo sin reconstruir la imagen.
COPY models_config.yml /app/models_config.yml

USER ach
WORKDIR /home/ach

# Los jobs se invocan como módulos:
#   docker run ach-jobs:latest processing.main --fecha 2026-07-30
#   docker run ach-jobs:latest models.runner --model-id caso05_clv --run-id <run>
ENTRYPOINT ["python", "-m"]
CMD ["processing.main", "--help"]
