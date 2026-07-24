# Entendimiento del Negocio y de los Datos — Proyecto ACH Data

**Metodología:** CRISP-DM · Fases 1 y 2 · **Fecha:** julio 2026
**Artefactos relacionados:** `EDA_ACH.ipynb` (análisis exploratorio completo) · `Docs/` (contexto fuente)

---

## 1. Entendimiento del negocio

### 1.1 La empresa
**ACH Colombia** es la cámara de compensación automatizada del sistema financiero colombiano: una fintech de
infraestructura (operando desde 1997, vigilada por la Superintendencia Financiera) que mueve las transferencias
interbancarias del país a través de cuatro rieles: **ACH Transferencias**, **Transfiya** (transferencias
inmediatas), **PSE** (botón de pagos en línea) y **SOI** (pago de aportes a seguridad social / PILA).
Procesa millones de operaciones al mes; si un colombiano paga por PSE o transfiere entre bancos, pasa por ACH.

### 1.2 El reto (ACH Data · INNLAB Universidad Icesi)
Monetizar analíticamente los datos transaccionales de ACH. Con una metodología *data-driven model discovery*
(partir del dato disponible y preguntar qué pregunta de negocio responde, en lugar de partir de una lista de
deseos sectorial), el equipo construyó:

- Un inventario de **~40 señales latentes** extraídas de la lectura semántica del diccionario de datos
  (ej.: `diasPension = 0` + transferencias activas → señal de empleo informal).
- Un catálogo de **105 modelos analíticos validados** con 3 filtros (fricción observable, señal alineada,
  factibilidad regulatoria — Ley 1581 de Habeas Data), distribuidos en 12+ sectores (Banca la mayor, con 14).
- **18 modelos que cubren los 5 casos de uso solicitados por el cliente**, 15 de ellos operables solo con
  datos ACH:

| # | Caso de uso | Idea central | Analítica |
|---|---|---|---|
| 1 | **Monitoreo y alertas** | Alerta roja (deterioro: cae IBC, retiro sin reingreso, caen flujos) + alerta verde (oportunidad: sube salario, aporte voluntario) | Series de tiempo · Clasificación |
| 2 | **Estimador de ingresos de independientes** | El IBC declarado puede ser 1 SMLV cuando transferencias y PSE apuntan a 3–5× ese valor → triangular las 3 fuentes | Regresión · Scoring |
| 3 | **Flujo de caja** | FCM = ingresos (SS + recibido) − pagos PSE − enviados; FCM positivo estable + AVP → perfil de inversión | Series de tiempo · Regresión |
| 4 | **Comportamientos de consumo** | El campo `Comercio` de PSE es el mapa de consumo sin encuestas | Clustering (RFM) · Clasificación |
| 5 | **Análisis y segmentos** | Salario × ciudad × comercio → perfiles geoeconómicos | Clustering |

### 1.3 Objetivo de esta fase del proyecto
Entender y validar los tres extractos de datos entregados (ofuscados), evaluar su calidad y confirmar si
contienen la señal que los casos de uso necesitan, antes de pasar a preparación de datos y modelado.

---

## 2. Entendimiento de los datos

### 2.1 Inventario de datos

| Dataset | Archivo | Filas | Columnas | Grano |
|---|---|---|---|---|
| Seguridad Social (PILA) | `Conversion Seguridad Social 1 - Ofuscado.xlsx` | 743.406 | 31 | persona × período × aportante |
| Transferencias ACH/Transfiya | `Conversion Transferencias ACH 1 - Ofuscado.xlsx` | 1.048.574 ⚠️ | 26 | persona × período × contraparte |
| Pagos Digitales (PSE) | `Conversion Pagos Digitales 1 - Ofuscado.xlsx` | 1.048.574 ⚠️ | 12 | persona × período × comercio |

⚠️ = exactamente el tope de filas de una hoja de Excel → **extractos truncados**.

Los tres datasets comparten un mismo patrón de diseño: columnas de **resumen por persona** (promedios
históricos, repetidos en cada fila de la persona) + columnas de **resumen por persona-período**
(`Total/Cantidad … periodo`) + columnas de **detalle**. Estos agregados deben de-duplicarse antes de usarse.

**Ventana temporal real** (verificada en el EDA): Seguridad Social cubre **36 meses (2023-07 → 2026-06)**;
Transferencias y PSE cubren **18 meses (2025-01 → 2026-06)**. El análisis cruzado entre fuentes solo es
válido en la ventana común 2025-01 → 2026-06.

**Cobertura de personas** (llave enmascarada): 33.281 en Seguridad Social, 26.904 en Transferencias,
29.856 en PSE; universo combinado de 40.892 personas.

### 2.2 Hallazgos principales del EDA

1. **La señal de informalidad es masiva (caso 2):** el **57,6%** de los registros PILA cotiza **salud sin
   pensión** (`Días pensión = 0` con salud activa) — exactamente el marcador de independientes/ingreso no
   declarado que anticipó la lectura semántica del diccionario.
2. **La brecha "declarado vs. real" existe y es cuantificable:** entre las **14.917 personas** presentes en
   las tres fuentes, la correlación (Spearman) entre IBC salud declarado y dinero recibido por ACH/Transfiya
   es de solo **ρ = 0,24**, y el **11,8% recibe más del doble de lo que declara**. Es la validación empírica
   del Estimador de Ingresos para Independientes.
3. **Rotación y volatilidad laboral visibles (caso 1):** 62,4% de los registros PILA trae novedades; 10,1%
   con Ingreso (ING), 8,9% con Retiro (RET), 42,4% con variación transitoria de salario y 6,4% con
   incapacidad — los insumos directos del motor de alertas rojas/verdes.
4. **Dos regímenes de transferencia:** Transfiya domina la frecuencia (64,8% de los registros, tickets
   bajos, uso P2P) y ACH concentra los valores altos; un modelo debe tratarlos por separado. El **19,3%**
   de los registros va a **cuenta propia** — la señal de estructuración/AML del catálogo.
5. **El gasto PSE es sobre todo servicio de deuda (caso 4):** Financiero/crédito 38,2% + pasarelas de pago
   13,3% del valor total; el consumo discrecional (retail, viajes, streaming) es minoritario en valor.
   El IBC mediano de la muestra es ≈ $1,75 M COP, con el pico esperado en el salario mínimo.
6. **El cruce entre fuentes es viable hoy:** **18.319 personas (44,8%)** del universo combinado aparecen en
   las tres fuentes — masa suficiente para prototipar los casos 2, 3 y 4 sin datos externos.

**Viabilidad por caso de uso con estos extractos:**

| Caso | Estado | Nota |
|---|---|---|
| 1 · Monitoreo y alertas | ✅ Viable | Novedades + series mensuales ya calculables |
| 2 · Ingresos independientes | ✅ Viable | El mejor respaldado por la evidencia (ρ=0,24; 11,8% >2×) |
| 3 · Flujo de caja | ✅ Viable con cautela | El truncamiento sesga la suma de egresos |
| 4 · Comportamientos de consumo | ⚠ Parcial | Falta catálogo maestro de comercios (39,5% sin clasificar) |
| 5 · Análisis y segmentos | ⚠ Parcial | Estos extractos no traen campo de ubicación |

### 2.3 Calidad de los datos y limitaciones

- **Truncamiento (crítico):** Transferencias y PSE están cortados en el tope de Excel → son muestras, no el
  universo. Evidencia: los agregados pre-calculados solo cuadran con el detalle en el **88,4%** de los pares
  persona-período. **No extrapolar volúmenes.**
- **Ofuscación con costo analítico:** la llave de persona pierde los últimos dígitos (identidad aproximada,
  posibles colisiones al cruzar fuentes); `Entidad autorizadora` (PSE) está 100% enmascarada (columna sin
  valor); los nombres de comercio truncados dejan **39,5%** del valor PSE sin clasificar en la taxonomía
  por palabras clave.
- **Nulos estructurales, no de calidad:** `Novedades` 37,6% nulo (= mes sin novedad); `Tipo salario` 14,7%;
  en Transferencias, originador/receptor se llenan según el sentido del registro (100% estructural).
  Fuera de eso, la completitud es buena.
- **Duplicados exactos marginales:** 50 filas en Seguridad Social (~0,01%) y 7.291 en Transferencias (0,7%).
- **Regulatorio:** el uso debe permanecer dentro de la Ley 1581 (Habeas Data); la trazabilidad
  campo → señal → modelo del catálogo es la herramienta de compliance.

---

## 3. Conclusión y próximos pasos

**Conclusión:** los tres extractos contienen la señal que los 5 casos de uso necesitan — la informalidad,
la brecha ingreso declarado/real, la rotación laboral y el mapa de gasto son visibles y cuantificables ya.
Las limitaciones dominantes no son de señal sino de **entrega** (truncamiento, ofuscación de la llave,
ventanas desiguales).

**Próximos pasos (CRISP-DM → Fase 3, preparación de datos):**
1. Re-solicitar los extractos completos en **CSV/parquet** (sin límite de Excel) con **llave de persona
   hasheada de forma consistente** entre las tres fuentes.
2. Conseguir el **diccionario de datos formal** (códigos de tipo planilla, novedades, clase aportante) y un
   **catálogo maestro de comercios PSE** con categoría/CIIU.
3. Construir la **tabla analítica persona × mes** (IBC, flujo recibido/enviado, gasto por categoría) sobre
   la ventana común 2025-01 → 2026-06 — base de los casos 1, 2 y 3.
4. Definir con negocio los **umbrales del motor de alertas** (caída de IBC, RET sin ING, caída de flujos) y
   los targets de los primeros modelos (estimador de ingreso real, FCM).
