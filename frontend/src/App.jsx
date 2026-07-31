import { useCallback, useEffect, useState } from 'react'
import { api } from './api.js'
import ModelCard from './components/ModelCard.jsx'
import ModelDetail from './components/ModelDetail.jsx'
import { formatearFecha } from './formato.js'

/** Modelo seleccionado en el hash de la URL, para que el detalle sea enlazable. */
function modeloDelHash() {
  const hash = window.location.hash.replace(/^#\/?/, '')
  return hash || null
}

export default function App() {
  const [datos, setDatos] = useState(null)
  const [error, setError] = useState(null)
  const [cargando, setCargando] = useState(true)
  const [seleccionado, setSeleccionado] = useState(modeloDelHash)

  const cargar = useCallback(() => {
    setCargando(true)
    api.modelos()
      .then((respuesta) => {
        setDatos(respuesta)
        setError(null)
      })
      .catch((e) => setError(e.message))
      .finally(() => setCargando(false))
  }, [])

  useEffect(cargar, [cargar])

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
          <p>Resultados de la última corrida de la pipeline</p>
        </div>
        <div className="cabecera__acciones">
          {datos?.run_id && <span className="corrida mono">{datos.run_id}</span>}
          <button className="boton" onClick={cargar} disabled={cargando}>
            {cargando ? 'Actualizando…' : 'Actualizar'}
          </button>
        </div>
      </header>

      <main className="contenido">
        {seleccionado ? (
          <ModelDetail modelId={seleccionado} onVolver={volver} />
        ) : (
          <Tablero datos={datos} error={error} cargando={cargando} onSeleccionar={seleccionar} />
        )}
      </main>

      <footer className="pie">
        <span>API: <span className="mono">{api.base}</span></span>
        {datos?.generated_at && <span>Índice generado {formatearFecha(datos.generated_at)}</span>}
      </footer>
    </div>
  )
}

function Tablero({ datos, error, cargando, onSeleccionar }) {
  if (cargando && !datos) return <div className="cargando">Cargando modelos…</div>

  if (error) {
    return (
      <EstadoVacio
        titulo="No se pudo hablar con la API"
        detalle={error}
        pistas={[
          'Comprueba que el backend esté arriba: make ps',
          `Prueba el endpoint de salud: ${api.base}/health`,
        ]}
      />
    )
  }

  const modelos = datos?.models || []
  if (!modelos.length) {
    return (
      <EstadoVacio
        titulo="Todavía no hay resultados"
        detalle="La pipeline no ha dejado ningún resultado en el bucket."
        pistas={[
          'Sube los datos de entrada: make seed',
          'Dispara la pipeline: make trigger (o desde la UI de Airflow)',
        ]}
      />
    )
  }

  const porCaso = modelos.reduce((acc, modelo) => {
    const caso = modelo.use_case ?? 0
    ;(acc[caso] ||= []).push(modelo)
    return acc
  }, {})

  return (
    <>
      <div className="resumen">
        <Indicador valor={datos.total_models} etiqueta="modelos" />
        <Indicador valor={datos.successful} etiqueta="correctos" tono="verde" />
        <Indicador valor={datos.failed} etiqueta="con error" tono={datos.failed ? 'rojo' : null} />
      </div>

      {Object.entries(porCaso)
        .sort(([a], [b]) => Number(a) - Number(b))
        .map(([caso, delCaso]) => (
          <section key={caso} className="grupo">
            <h2 className="grupo__titulo">Caso de uso {caso}</h2>
            <div className="rejilla">
              {delCaso.map((modelo) => (
                <ModelCard key={modelo.model_id} modelo={modelo} onSeleccionar={onSeleccionar} />
              ))}
            </div>
          </section>
        ))}
    </>
  )
}

function Indicador({ valor, etiqueta, tono }) {
  return (
    <div className={`indicador ${tono ? `indicador--${tono}` : ''}`}>
      <span className="indicador__valor">{valor ?? '—'}</span>
      <span className="indicador__etiqueta">{etiqueta}</span>
    </div>
  )
}

function EstadoVacio({ titulo, detalle, pistas = [] }) {
  return (
    <div className="vacio">
      <h2>{titulo}</h2>
      <p>{detalle}</p>
      {pistas.length > 0 && (
        <ul>
          {pistas.map((pista) => <li key={pista}><code>{pista}</code></li>)}
        </ul>
      )}
    </div>
  )
}
