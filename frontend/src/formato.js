// Formateo de números y etiquetas. Nada aquí inventa datos: si un valor no viene,
// se muestra un guion.

const NUMERO = new Intl.NumberFormat('es-CO')
const DECIMAL = new Intl.NumberFormat('es-CO', { minimumFractionDigits: 3, maximumFractionDigits: 3 })

/** Métricas que son proporciones y se leen mejor como porcentaje. */
const PORCENTAJES = new Set(['mape', 'tasa_positiva', 'share'])

/** Métricas que son conteos y no llevan decimales. */
const CONTEOS = /^(n_|k$|.*_identificados$|personas|total)/

export function formatearMetrica(nombre, valor) {
  if (valor === null || valor === undefined || Number.isNaN(valor)) return '—'
  if (typeof valor !== 'number') return String(valor)

  if (PORCENTAJES.has(nombre)) return `${(valor * 100).toFixed(1)}%`
  if (nombre.startsWith('pct_')) return `${valor.toFixed(1)}%`
  if (CONTEOS.test(nombre) || Number.isInteger(valor)) return NUMERO.format(valor)
  if (Math.abs(valor) >= 100000) return abreviarMonto(valor)
  return DECIMAL.format(valor)
}

export function abreviarMonto(valor) {
  if (valor === null || valor === undefined) return '—'
  const absoluto = Math.abs(valor)
  if (absoluto >= 1e12) return `${(valor / 1e12).toFixed(1)} B`
  if (absoluto >= 1e9) return `${(valor / 1e9).toFixed(1)} MM`
  if (absoluto >= 1e6) return `${(valor / 1e6).toFixed(1)} M`
  if (absoluto >= 1e3) return `${(valor / 1e3).toFixed(1)} k`
  return NUMERO.format(Math.round(valor))
}

export function etiquetaMetrica(nombre) {
  return nombre
    .replace(/_/g, ' ')
    .replace(/\bpct\b/g, '%')
    .replace(/^./, (c) => c.toUpperCase())
}

export function formatearFecha(iso) {
  if (!iso) return '—'
  const fecha = new Date(iso)
  if (Number.isNaN(fecha.getTime())) return String(iso)
  return fecha.toLocaleString('es-CO', {
    day: '2-digit', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit',
  })
}

export function formatearDuracion(segundos) {
  if (!segundos && segundos !== 0) return '—'
  if (segundos < 60) return `${segundos.toFixed(1)} s`
  const minutos = Math.floor(segundos / 60)
  return `${minutos} min ${Math.round(segundos % 60)} s`
}

/** Etiqueta legible del tipo de tarea. */
export const ETIQUETA_TAREA = {
  clustering: 'Segmentación',
  classification: 'Clasificación',
  regression: 'Regresión',
  scoring: 'Scoring',
}

/**
 * Métricas que encabezan la tarjeta de cada modelo, por tipo de tarea.
 * Si un modelo no reporta ninguna de estas, se muestran las tres primeras que traiga:
 * el tablero no supone qué métricas existen.
 */
const DESTACADAS = {
  clustering: ['silhouette', 'k', 'n_entities'],
  classification: ['roc_auc', 'average_precision', 'n_entities'],
  regression: ['r2_log', 'mape', 'n_entities'],
  scoring: ['n_entities'],
}

export function metricasDestacadas(taskType, metrics = {}) {
  const preferidas = (DESTACADAS[taskType] || []).filter((m) => m in metrics)
  const resto = Object.keys(metrics).filter((m) => !preferidas.includes(m))
  return [...preferidas, ...resto].slice(0, 3).map((nombre) => ({ nombre, valor: metrics[nombre] }))
}

/** Paleta categórica fija: el mismo segmento conserva su color entre gráficos. */
export const PALETA = [
  '#2a78d6', '#008300', '#e87ba4', '#eda100',
  '#1baf7a', '#eb6834', '#4a3aa7', '#e34948',
]

export const colorPorIndice = (i) => PALETA[i % PALETA.length]
