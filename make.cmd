@echo off
REM Equivalente del Makefile para Windows sin GNU make.
REM
REM   make up      levanta el stack
REM   make seed    sube los XLSX al bucket raw
REM   make test    corre los tests
REM
REM En Git Bash, WSL, macOS o Linux se usa el Makefile normal.
REM Para tener 'make' nativo en Windows: choco install make  (o scoop install make)

setlocal enabledelayedexpansion
if "%~1"=="" goto :help

if exist ".env" (
    for /f "usebackq tokens=1,* delims==" %%a in (`findstr /v /r "^#" ".env"`) do (
        if not "%%a"=="" set "%%a=%%b"
    )
)
if "%ACH_DOCKER_NETWORK%"=="" set "ACH_DOCKER_NETWORK=ach_net"
if "%JOBS_IMAGE%"=="" set "JOBS_IMAGE=ach-jobs:latest"

if /i "%~1"=="up"          goto :up
if /i "%~1"=="up-app"      goto :up_app
if /i "%~1"=="down"        goto :down
if /i "%~1"=="reset"       goto :reset
if /i "%~1"=="ps"          goto :ps
if /i "%~1"=="logs"        goto :logs
if /i "%~1"=="build"       goto :build
if /i "%~1"=="build-jobs"  goto :build_jobs
if /i "%~1"=="seed"        goto :seed
if /i "%~1"=="dag"         goto :dag
if /i "%~1"=="test"        goto :test
if /i "%~1"=="parity"      goto :parity
if /i "%~1"=="lint"        goto :lint
goto :help

:up
if not exist .env copy .env.example .env >nul
call :build_jobs || exit /b 1
docker compose build airflow-init || exit /b 1
docker compose up -d || exit /b 1
echo.
echo   Airflow  ^-^> http://localhost:8080
echo   MinIO    ^-^> http://localhost:9001
echo.
echo   Siguiente paso: make seed y disparar el DAG pipeline_modelos.
exit /b 0

:up_app
call :build_jobs || exit /b 1
docker compose build airflow-init || exit /b 1
docker compose --profile app up -d
exit /b %errorlevel%

:down
docker compose --profile app down
exit /b %errorlevel%

:reset
docker compose --profile app down -v
exit /b %errorlevel%

:ps
docker compose ps
exit /b %errorlevel%

:logs
docker compose logs -f --tail=100
exit /b %errorlevel%

:build_jobs
docker build -t %JOBS_IMAGE% .\jobs
exit /b %errorlevel%

:build
call :build_jobs || exit /b 1
docker compose build
exit /b %errorlevel%

:seed
if "%ACH_DATA_DIR%"=="" (
    echo Define ACH_DATA_DIR en .env
    exit /b 1
)
docker run --rm --network %ACH_DOCKER_NETWORK% ^
    -e ACH_S3_ENDPOINT=http://minio:9000 ^
    -e ACH_S3_ACCESS_KEY=%MINIO_ROOT_USER% ^
    -e ACH_S3_SECRET_KEY=%MINIO_ROOT_PASSWORD% ^
    -e ACH_DATA_DIR=/datos ^
    -v "%ACH_DATA_DIR%:/datos:ro" ^
    %JOBS_IMAGE% processing.seed
exit /b %errorlevel%

:dag
docker compose exec airflow-scheduler airflow dags trigger pipeline_modelos
exit /b %errorlevel%

:test
python -m pytest tests/unit -q
exit /b %errorlevel%

:parity
python -m pytest tests/parity -q -m parity
exit /b %errorlevel%

:lint
python -m ruff check jobs tests
exit /b %errorlevel%

:help
echo   up          Levanta todo el stack
echo   up-app      Levanta el stack incluyendo backend y frontend
echo   down        Apaga el stack
echo   reset       Apaga y borra los volumenes
echo   ps          Estado de los servicios
echo   logs        Logs de todos los servicios
echo   build       Construye todas las imagenes
echo   build-jobs  Construye la imagen de los jobs
echo   seed        Sube los XLSX de ACH_DATA_DIR al bucket raw
echo   dag         Dispara el DAG pipeline_modelos
echo   test        Tests unitarios
echo   parity      Tests de paridad
echo   lint        Revisa el estilo del codigo
exit /b 0
