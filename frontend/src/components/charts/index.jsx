// Registro de renderizadores de gráficos.
//
// ESTA ES LA PIEZA QUE HACE GENÉRICO AL TABLERO: la vista de detalle no sabe qué
// modelo está mostrando, solo recorre las claves de `charts` del JSON y pinta las que
// encuentra aquí. Un modelo que no reporte un bloque simplemente no lo muestra, y uno
// que reporte un bloque desconocido no rompe nada.
//
// Agregar un tipo de gráfico nuevo = agregar una entrada a este mapa.

import ConfusionMatrix from './ConfusionMatrix.jsx'
import Distribution from './Distribution.jsx'
import FeatureImportance from './FeatureImportance.jsx'
import KSelection from './KSelection.jsx'
import Residuals from './Residuals.jsx'
import RocCurve from './RocCurve.jsx'
import Scatter2D from './Scatter2D.jsx'
import SegmentDistribution from './SegmentDistribution.jsx'

export const RENDERIZADORES = {
  segment_distribution: {
    titulo: 'Distribución de segmentos',
    descripcion: 'Tamaño de cada grupo encontrado y su peso sobre la base.',
    componente: SegmentDistribution,
  },
  k_selection: {
    titulo: 'Selección del número de segmentos',
    descripcion: 'Cómo se comportan las métricas de calidad al variar k.',
    componente: KSelection,
  },
  scatter_2d: {
    titulo: 'Segmentos en el plano principal',
    descripcion: 'Proyección PCA de las personas, coloreada por segmento.',
    componente: Scatter2D,
  },
  feature_importance: {
    titulo: 'Variables más influyentes',
    descripcion: 'Qué pesa más en la predicción del modelo.',
    componente: FeatureImportance,
  },
  roc_curve: {
    titulo: 'Curva ROC',
    descripcion: 'Capacidad de separar las dos clases a lo largo de todos los umbrales.',
    componente: RocCurve,
  },
  confusion_matrix: {
    titulo: 'Matriz de confusión',
    descripcion: 'Aciertos y errores del modelo sobre el conjunto de prueba.',
    componente: ConfusionMatrix,
  },
  residuals: {
    titulo: 'Predicho frente a real',
    descripcion: 'Dispersión de la predicción contra el valor observado.',
    componente: Residuals,
  },
  distribution: {
    titulo: 'Distribución',
    descripcion: null,
    componente: Distribution,
  },
}

/** Bloques del JSON que sabemos dibujar, en el orden en que conviene leerlos. */
export const ORDEN = [
  'segment_distribution',
  'scatter_2d',
  'k_selection',
  'roc_curve',
  'confusion_matrix',
  'feature_importance',
  'residuals',
  'distribution',
]

export function bloquesRenderizables(charts = {}) {
  const presentes = Object.entries(charts)
    .filter(([clave, valor]) => valor != null && RENDERIZADORES[clave])
    .map(([clave]) => clave)
  return ORDEN.filter((clave) => presentes.includes(clave))
}

export function bloquesDesconocidos(charts = {}) {
  return Object.keys(charts).filter((clave) => charts[clave] != null && !RENDERIZADORES[clave])
}
