// Cliente de la API. La URL llega en tiempo de arranque desde /config.js, nunca
// horneada en el bundle ni con credenciales: el navegador solo habla con el backend.

const BASE = (window.__ACH_CONFIG__?.apiUrl || 'http://localhost:8000').replace(/\/$/, '')

async function pedir(ruta) {
  const respuesta = await fetch(`${BASE}${ruta}`)
  if (!respuesta.ok) {
    const detalle = await respuesta.json().catch(() => ({}))
    const error = new Error(detalle.detail || `La API respondió ${respuesta.status}`)
    error.status = respuesta.status
    throw error
  }
  return respuesta.json()
}

export const api = {
  base: BASE,
  salud: () => pedir('/health'),
  modelos: () => pedir('/api/models'),
  ultimo: (modelId) => pedir(`/api/models/${encodeURIComponent(modelId)}/latest`),
  historico: (modelId) => pedir(`/api/models/${encodeURIComponent(modelId)}/runs`),
  corrida: (runId) => pedir(`/api/runs/${encodeURIComponent(runId)}`),
  urlArtefacto: (modelId, runId, nombre) =>
    `${BASE}/api/models/${encodeURIComponent(modelId)}/runs/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(nombre)}`,
}
