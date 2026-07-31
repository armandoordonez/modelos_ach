# Paridad de la migración

Este documento responde a una sola pregunta: **¿la pipeline reproduce lo que hacían los
notebooks y scripts originales?**

La respuesta se separa en dos partes, porque son dos cosas distintas:

1. **Paridad de migración** — ¿el código migrado calcula lo mismo que el original?
2. **Re-baseline** — ¿qué cambió a propósito, y cuánto movió las métricas?

Mezclarlas es lo que hace imposible saber si una diferencia es un error de migración o una
decisión de diseño. Por eso el linaje es un parámetro (`ACH_LINEAGE`): con `legacy` cada
modelo usa la llave de persona de su script original, y con `cedula-v1` —producción— los
siete cruzan por la cédula ofuscada.

---

## 1 · Procesamiento de datos

Verificado contra los datos reales de ACH el 2026-07-31.

| Fuente | Filas leídas | Esperado (EDA) | Duplicados exactos | Esperado | Filas curadas |
|---|---:|---:|---:|---:|---:|
| Seguridad Social | 743.406 | 743.406 | 50 | 50 | 743.356 |
| Transferencias | 1.048.574 | 1.048.574 | 7.291 | 7.291 | 1.041.283 |
| Pagos Digitales | 1.048.574 | 1.048.574 | 1 | 1 | 1.048.573 |

**Columnas:** 31 + 26 + 12 = 69, idénticas al diccionario de datos.
**Tipos:** aplicados desde `common/schema.py`, validados con Pandera en cada lote.

✅ **Paridad exacta.** Mismas filas, mismas columnas, mismos tipos, mismos duplicados
descartados que los notebooks originales.

---

## 2 · Modelos del Caso 05 — paridad exacta

Estos tres ya cruzaban por cédula, así que la unificación de linaje no los toca. Se
comparan contra sus notebooks ejecutados.

| Modelo | Métrica | Notebook | Pipeline | Δ |
|---|---|---:|---:|---|
| **#4 CLV** | personas | 32.164 | 32.164 | — |
| | k | 4 | 4 | — |
| | silueta | 0,231 | 0,2307 | ✅ |
| **#46 Ciclo de vida** | personas | 33.281 | 33.281 | — |
| | k | 5 | 5 | — |
| | silueta | 0,350 | 0,3496 | ✅ |
| | % con transaccional | 73,8% | 73,77% | ✅ |
| **#101 Pensionados** | identificados en PILA | 583 | 583 | — |
| | universo de modelado | 360 | 360 | — |
| | escalador · k | Robust · 4 | Robust · 4 | — |
| | silueta | 0,271 | 0,2713 | ✅ |
| | Davies-Bouldin | 1,201 | 1,2014 | ✅ |

---

## 3 · Modelos del Caso 02 y 04 — re-baseline

Aquí las métricas **cambian a propósito**. Cada cambio está aislado y justificado.

| Modelo | Métrica | Script original | Pipeline | Causa del delta |
|---|---|---:|---:|---|
| **#6 Ingresos** | R² (log) | 0,473 | 0,445 | Llave unificada a cédula |
| | MAPE | 25,5% | 25,4% | — |
| **#15 RFM** | silueta | 0,498 | 0,405 | Llave + `k ≥ 4` por accionabilidad |
| | k | 2 | 4 | Ver nota (b) |
| **#17 Propensión salud** | ROC-AUC | 0,835 | 0,698 | **Fuga corregida** |
| | AUC nuevos adoptantes | no reportado | 0,678 | — |
| **#55 Propensión turismo** | ROC-AUC | 0,741 | 0,749 | Fuga corregida + llave |
| | AUC nuevos adoptantes | no reportado | 0,744 | — |

### Notas sobre cada cambio

**(a) Llave de persona unificada.** Los tres orígenes traían tres llaves distintas:

| Origen | Llave |
|---|---|
| Caso 05 | `Número documento` (cédula ofuscada) |
| Caso 02 | `Nombre \| Número documento` |
| Caso 04 | `nombre_normalizado \| documento_sin_asteriscos` |

La cédula llega enmascarada de forma idéntica en las tres fuentes, así que es la única
que permite cruzarlas. Cambiar la llave cambia el universo de personas —una llave
compuesta separa homónimos que la cédula junta— y por eso mueve las métricas. Es el
costo consciente de tener un linaje único.

**(b) `k ≥ 4` en RFM.** Con el rango original (desde k=2) el modelo elige **k=2** con
silueta 0,514: técnicamente mejor, comercialmente inútil, porque el corte es
"gasta / no gasta". Se aplicó el mismo criterio de accionabilidad que ya usaban los
modelos del Caso 05. Está declarado en `models_config.yml` y es reversible.

**(c) Fuga de la categoría objetivo — el cambio más importante.** El notebook original
del Caso 04 excluía a propósito la categoría a predecir de las variables predictoras
(`excluir_categoria`, 4 usos). Al migrarlo a script, esa exclusión se perdió: el gasto
pasado *en salud* entraba a predecir el gasto futuro *en salud*, de modo que el modelo
en buena parte se predecía a sí mismo.

La pipeline restaura la exclusión y lo deja registrado en el JSON de cada corrida:

```json
"variables_excluidas_por_fuga": ["gasto_Salud_med", "gasto_Salud_max"]
```

El AUC de #17 baja de 0,835 a 0,698. **Ese es el resultado correcto**: 0,835 medía la
capacidad del modelo de leer su propia respuesta.

El de turismo (#55) no baja —sube ligeramente— porque su señal nunca dependió tanto de
la persistencia: ya en el notebook original se observaba que predecía bien incluso sobre
personas sin gasto previo en la categoría.

**(d) AUC en nuevos adoptantes.** Métrica nueva, no estaba en el script. Es la que
distingue un modelo que encuentra clientes nuevos de uno que solo confirma a los que ya
estaban. Turismo la mantiene casi igual al AUC general (0,744 vs 0,749): encuentra gente
nueva de verdad.

---

## 4 · Cómo reproducirlo

```bash
# Paridad de la migración: cada modelo con la llave de su script original
ACH_LINEAGE=legacy make parity

# Producción: linaje unificado
ACH_LINEAGE=cedula-v1 make parity
```

Los valores de referencia viven en `tests/parity/referencias.json`. Cuando una decisión
de modelado cambie a propósito, se actualiza ese archivo **y** esta tabla, en el mismo
commit: un delta sin explicación es un bug hasta que se demuestre lo contrario.
