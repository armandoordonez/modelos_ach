// Cliente de la API. La URL llega en tiempo de arranque desde /config.js, nunca
// horneada en el bundle ni con credenciales: el navegador solo habla con el backend.
//
// Modo estático: con `modo: "estatico"` el tablero lee archivos JSON exportados en vez
// de llamar al backend. Sirve para publicar una demo en cualquier hosting de estáticos
// (Netlify, Vercel, GitHub Pages, un bucket) sin levantar infraestructura. Los datos
// quedan congelados en el momento de la exportación.

const CONFIG = window.__ACH_CONFIG__ || {}
const BASE = (CONFIG.apiUrl || 'http://localhost:8000').replace(/\/$/, '')
const ESTATICO = CONFIG.modo === 'estatico'

/** En modo estático cada ruta de la API tiene su archivo exportado. */
function rutaEstatica(ruta) {
  const [camino, busqueda = ''] = ruta.split('?')
  const query = new URLSearchParams(busqueda)

  if (camino === '/api/use-cases/4/dashboard') {
    const runId = query.get('run_id')
    return runId
      ? `./datos/casos-uso/4/dashboard-${runId}.json`
      : './datos/casos-uso/4/dashboard.json'
  }
  const modelo = camino.match(/^\/api\/models\/([^/]+)\/(latest|runs)$/)
  if (modelo) {
    const [, id, recurso] = modelo
    return recurso === 'latest'
      ? `./datos/modelos/${id}.json`
      : `./datos/modelos/${id}-runs.json`
  }
  if (camino === '/api/models') return './datos/models.json'
  if (camino === '/health') return './datos/health.json'
  const corrida = camino.match(/^\/api\/runs\/(.+)$/)
  if (corrida) return `./datos/corridas/${corrida[1]}.json`
  return `./datos${camino.replace(/^\/api/, '')}.json`
}

async function pedir(ruta) {
  const destino = ESTATICO ? rutaEstatica(ruta) : `${BASE}${ruta}`
  const respuesta = await fetch(destino)
  if (!respuesta.ok) {
    const detalle = await respuesta.json().catch(() => ({}))
    const error = new Error(detalle.detail || `La API respondió ${respuesta.status}`)
    error.status = respuesta.status
    throw error
  }
  return respuesta.json()
}

export const api = {
  base: ESTATICO ? 'exportación estática' : BASE,
  estatico: ESTATICO,
  salud: () => pedir('/health'),
  modelos: () => pedir('/api/models'),
  dashboardCaso04: (runId) =>
    pedir(runId ? `/api/use-cases/4/dashboard?run_id=${encodeURIComponent(runId)}` : '/api/use-cases/4/dashboard'),
  ultimo: (modelId) => pedir(`/api/models/${encodeURIComponent(modelId)}/latest`),
  historico: (modelId) => pedir(`/api/models/${encodeURIComponent(modelId)}/runs`),
  corrida: (runId) => pedir(`/api/runs/${encodeURIComponent(runId)}`),
  // Los artefactos binarios no se exportan: en modo estático no hay de dónde servirlos.
  urlArtefacto: (modelId, runId, nombre) =>
    ESTATICO
      ? null
      : `${BASE}/api/models/${encodeURIComponent(modelId)}/runs/${encodeURIComponent(runId)}/artifacts/${encodeURIComponent(nombre)}`,
}
