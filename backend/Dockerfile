# API de resultados.
#
# El contexto de build es la raíz del repo porque la imagen reutiliza `jobs/common`:
# el backend lee el bucket con el mismo cliente que la pipeline, en vez de tener una
# copia propia que se desincronice.
#
# No instala pandas ni scikit-learn: solo lee JSON.

FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    ACH_MODELS_CONFIG=/app/models_config.yml \
    PYTHONPATH=/app

RUN apt-get update \
    && apt-get install -y --no-install-recommends curl \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --create-home --uid 50000 ach

WORKDIR /app

COPY backend/requirements.txt ./requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

COPY jobs/common ./common
COPY jobs/models_config.yml ./models_config.yml
COPY backend/app ./app

USER ach
EXPOSE 8000

HEALTHCHECK --interval=15s --timeout=5s --start-period=20s --retries=5 \
    CMD curl --fail http://localhost:8000/health || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
