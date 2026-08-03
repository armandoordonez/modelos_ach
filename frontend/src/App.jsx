import { useCallback, useEffect, useState } from 'react'
import { api } from './api.js'
import Case04Dashboard from './components/Case04Dashboard.jsx'
import ModelDetail from './components/ModelDetail.jsx'
import { formatearFecha } from './formato.js'

function modeloDelHash() {
  const hash = window.location.hash.replace(/^#\/?/, '')
  return hash || null
}

export default function App() {
  const [datos, setDatos] = useState(null)
  const [error, setError] = useState(null)
  const [cargando, setCargando] = useState(true)
  const [corrida, setCorrida] = useState(null)
  const [seleccionado, setSeleccionado] = useState(modeloDelHash)

  const cargar = useCallback((runId = corrida) => {
    setCargando(true)
    api.dashboardCaso04(runId)
      .then((respuesta) => {
        setDatos(respuesta)
        setError(null)
      })
      .catch((e) => setError(e.message))
      .finally(() => setCargando(false))
  }, [corrida])

  useEffect(() => {
    cargar(corrida)
  }, [cargar, corrida])

  useEffect(() => {
    const alCambiarHash = () => setSeleccionado(modeloDelHash())
    window.addEventListener('hashchange', alCambiarHash)
    return () => window.removeEventListener('hashchange', alCambiarHash)
  }, [])

  const seleccionar = (modelId) => {
    window.location.hash = `/${modelId}`
    setSeleccionado(modelId)
  }

  const volver = () => {
    window.location.hash = ''
    setSeleccionado(null)
  }

  return (
    <div className="aplicacion">
      <header className="cabecera">
        <div className="cabecera__marca">
          <h1>ACH · Modelos analíticos</h1>
          <p>Dashboard interactivo del Caso 04 · Comportamientos de consumo</p>
        </div>
        <div className="cabecera__acciones">
          {datos?.run_id && <span className="corrida mono">{datos.run_id}</span>}
          <button className="boton" onClick={() => cargar(corrida)} disabled={cargando}>
            {cargando ? 'Actualizando…' : 'Actualizar'}
          </button>
        </div>
      </header>

      <main className="contenido">
        {seleccionado ? (
          <ModelDetail modelId={seleccionado} onVolver={volver} />
        ) : (
          <Case04Dashboard
            datos={datos}
            error={error}
            cargando={cargando}
            recargando={cargando && !!datos}
            runId={corrida}
            onCambiarCorrida={setCorrida}
            onSeleccionarModelo={seleccionar}
          />
        )}
      </main>

      <footer className="pie">
        <span>API: <span className="mono">{api.base}</span></span>
        {datos?.generated_at && <span>Vista generada {formatearFecha(datos.generated_at)}</span>}
      </footer>
    </div>
  )
}
