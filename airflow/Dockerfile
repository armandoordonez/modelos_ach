# Imagen de Airflow: SOLO orquesta.
#
# Aquí no se instala scikit-learn, pandas ni ninguna librería de modelos: eso vive en
# la imagen de jobs. Airflow lanza contenedores, no entrena. Mezclar ambas cosas es lo
# que hace que actualizar una versión de sklearn rompa el scheduler.

ARG AIRFLOW_VERSION=3.0.3
FROM apache/airflow:${AIRFLOW_VERSION}

# El directorio de credenciales se crea aquí, no en el compose: un volumen nombrado
# hereda el dueño y los permisos de la ruta que existe en la imagen. Si no existe,
# Docker lo crea como root y el contenedor —que corre como 50000:0— no puede escribir.
USER root
RUN mkdir -p /opt/airflow/auth \
    && chown airflow:root /opt/airflow/auth \
    && chmod 775 /opt/airflow/auth

USER airflow

COPY requirements.txt /tmp/requirements.txt
RUN pip install --no-cache-dir -r /tmp/requirements.txt
