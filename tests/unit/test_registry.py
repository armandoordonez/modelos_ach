"""Tests del registro de modelos: es el contrato de 'agregar un modelo = una entrada'."""

from __future__ import annotations

import pytest
import yaml
from pydantic import ValidationError

from common.registry import RegistroModelos, cargar_registro, ruta_config


def test_el_registro_del_repo_carga_y_valida():
    registro = cargar_registro()
    assert len(registro.modelos) == 7
    assert registro.schema_version == "1.0"


def test_los_siete_modelos_acordados_estan_registrados():
    ids = set(cargar_registro().ids())
    assert ids == {
        "caso02_ingresos_independientes",
        "caso04_rfm_consumidores",
        "caso04_propension_salud",
        "caso04_propension_turismo",
        "caso05_clv",
        "caso05_ciclo_vida",
        "caso05_pensionados",
    }


def test_hay_los_tres_tipos_de_tarea():
    """El tablero necesita los tres para poder renderizar todas sus secciones."""
    tipos = {m.task_type for m in cargar_registro().modelos}
    assert tipos == {"clustering", "regression", "classification"}


def test_dos_entradas_comparten_modulo_y_solo_cambian_por_parametro():
    """#17 y #55 son el mismo código con distinta categoría: agregar el segundo
    fue una entrada en el YAML, cero código nuevo."""
    registro = cargar_registro()
    salud = registro.obtener("caso04_propension_salud")
    turismo = registro.obtener("caso04_propension_turismo")
    assert salud.modulo == turismo.modulo
    assert salud.params["categoria"] != turismo.params["categoria"]


def test_la_categoria_objetivo_esta_fijada_en_config():
    """El script original la re-elegía solo si no la encontraba en los datos."""
    for model_id in ("caso04_propension_salud", "caso04_propension_turismo"):
        modelo = cargar_registro().obtener(model_id)
        assert modelo.params.get("categoria"), f"{model_id} debe declarar su categoría"


def test_cada_modelo_declara_su_llave_legacy():
    for modelo in cargar_registro().modelos:
        assert modelo.legacy_key in (
            "cedula", "nombre_documento", "nombre_normalizado_documento_visible")


def test_el_comando_es_uniforme_para_todos():
    for modelo in cargar_registro().modelos:
        assert modelo.comando[:3] == ["python", "-m"] + [modelo.modulo]
        assert modelo.comando[-2:] == ["--model-id", modelo.id]


def test_modelo_inexistente_falla_listando_los_registrados():
    with pytest.raises(KeyError, match="no está en models_config.yml"):
        cargar_registro().obtener("modelo_fantasma")


def test_ids_repetidos_fallan(tmp_path):
    contenido = {
        "schema_version": "1.0",
        "modelos": [
            {"id": "repetido", "nombre": "A", "caso_uso": 5, "task_type": "clustering",
             "modulo": "models.a.main"},
            {"id": "repetido", "nombre": "B", "caso_uso": 5, "task_type": "clustering",
             "modulo": "models.b.main"},
        ],
    }
    destino = tmp_path / "models_config.yml"
    destino.write_text(yaml.safe_dump(contenido), encoding="utf-8")
    with pytest.raises(ValueError, match="ids de modelo repetidos"):
        cargar_registro(destino)


def test_config_inexistente_falla_con_mensaje_util(tmp_path):
    with pytest.raises(FileNotFoundError, match="ACH_MODELS_CONFIG"):
        cargar_registro(tmp_path / "no_existe.yml")


def test_la_ruta_por_defecto_vive_junto_al_paquete():
    assert ruta_config().name == "models_config.yml"
    assert ruta_config().exists()


def test_agregar_un_modelo_es_solo_una_entrada(tmp_path):
    """Prueba del criterio de aceptación: nada más que el YAML cambia."""
    base = yaml.safe_load(ruta_config().read_text(encoding="utf-8"))
    base["modelos"].append({
        "id": "modelo_nuevo", "nombre": "Modelo nuevo", "catalogo": "#999", "caso_uso": 1,
        "task_type": "clustering", "modulo": "models.caso05_clv.main", "params": {},
    })
    destino = tmp_path / "models_config.yml"
    destino.write_text(yaml.safe_dump(base, allow_unicode=True), encoding="utf-8")

    registro = cargar_registro(destino)
    assert len(registro.modelos) == 8
    assert registro.obtener("modelo_nuevo").caso_uso == 1


def test_task_type_invalido_falla(tmp_path):
    contenido = {"schema_version": "1.0", "modelos": [
        {"id": "x", "nombre": "X", "caso_uso": 5, "task_type": "adivinacion",
         "modulo": "models.x.main"}]}
    destino = tmp_path / "models_config.yml"
    destino.write_text(yaml.safe_dump(contenido), encoding="utf-8")
    with pytest.raises(ValidationError):
        cargar_registro(destino)


def test_registro_vacio_es_valido_pero_no_tiene_modelos():
    registro = RegistroModelos(modelos=[])
    assert registro.ids() == []
