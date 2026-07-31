# Imagen de Airflow: SOLO orquesta.
#
# Aquí no se instala scikit-learn, pandas ni ninguna librería de modelos: eso vive en
# la imagen de jobs. Airflow lanza contenedores, no entrena. Mezclar ambas cosas es lo
# que hace que actualizar una versión de sklearn rompa el scheduler.

ARG AIRFLOW_VERSION=3.0.3
FROM apache/airflow:${AIRFLOW_VERSION}

USER airflow

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
