import { useEffect, useState } from 'react'
import { api } from '../api.js'
import {
  ETIQUETA_TAREA, etiquetaMetrica, formatearDuracion, formatearFecha, formatearMetrica,
} from '../formato.js'
import { RENDERIZADORES, bloquesDesconocidos, bloquesRenderizables } from './charts/index.jsx'

/**
 * Detalle de un modelo.
 *
 * No sabe qué modelo está mostrando: recorre `charts` del JSON y dibuja los bloques
 * que el registro de renderizadores conoce. Si un modelo no reporta ROC, no hay
 * sección de ROC — sin estados vacíos inventados.
 */
export default function ModelDetail({ modelId, onVolver }) {
  const [resultado, setResultado] = useState(null)
  const [historial, setHistorial] = useState([])
  const [error, setError] = useState(null)
  const [cargando, setCargando] = useState(true)

  useEffect(() => {
    let vigente = true
    setCargando(true)
    setError(null)
    Promise.all([api.ultimo(modelId), api.historico(modelId).catch(() => ({ runs: [] }))])
      .then(([datos, hist]) => {
        if (!vigente) return
        setResultado(datos)
        setHistorial(hist.runs || [])
      })
      .catch((e) => vigente && setError(e.message))
      .finally(() => vigente && setCargando(false))
    return () => {
      vigente = false
    }
  }, [modelId])

  if (cargando) return <div className="cargando">Cargando {modelId}…</div>
  if (error) {
    return (
      <div className="detalle">
        <button className="volver" onClick={onVolver}>← Volver</button>
        <div className="aviso aviso--error"><strong>No se pudo cargar el modelo.</strong><p>{error}</p></div>
      </div>
    )
  }
  if (!resultado) return null

  const charts = resultado.charts || {}
  const bloques = bloquesRenderizables(charts)
  const desconocidos = bloquesDesconocidos(charts)
  const fallido = resultado.status !== 'success'

  return (
    <div className="detalle">
      <button className="volver" onClick={onVolver}>← Volver al tablero</button>

      <header className="detalle__encabezado">
        <div>
          <span className="etiqueta-catalogo">{resultado.catalog_ref}</span>
          <h2>{resultado.model_name}</h2>
          <p className="detalle__meta">
            Caso de uso {resultado.use_case} ·{' '}
            {ETIQUETA_TAREA[resultado.task_type] || resultado.task_type} ·{' '}
            {formatearFecha(resultado.finished_at)} · {formatearDuracion(resultado.duration_seconds)}
          </p>
        </div>
        <span className={`chip chip--${fallido ? 'fallo' : resultado.task_type}`}>
          {fallido ? 'Falló' : 'Correcto'}
        </span>
      </header>

      {fallido && (
        <div className="aviso aviso--error">
          <strong>Esta corrida terminó con error.</strong>
          <pre>{resultado.error}</pre>
        </div>
      )}

      <Seccion titulo="Métricas">
        <div className="metricas metricas--grande">
          {Object.entries(resultado.metrics || {}).map(([nombre, valor]) => (
            <div key={nombre} className="metrica">
              <span className="metrica__valor">{formatearMetrica(nombre, valor)}</span>
              <span className="metrica__nombre">{etiquetaMetrica(nombre)}</span>
            </div>
          ))}
        </div>
      </Seccion>

      {bloques.map((clave) => {
        const { titulo, descripcion, componente: Componente } = RENDERIZADORES[clave]
        return (
          <Seccion key={clave} titulo={titulo} descripcion={descripcion}>
            <Componente datos={charts[clave]} resultado={resultado} />
          </Seccion>
        )
      })}

      <Seccion titulo="Datos y parámetros">
        <div className="dos-columnas">
          <div>
            <h4>Dataset</h4>
            <dl className="lista-datos">
              <dt>Filas</dt><dd>{formatearMetrica('n_x', resultado.dataset?.rows)}</dd>
              <dt>Ventana</dt><dd>{resultado.dataset?.window?.join(' → ') || '—'}</dd>
              <dt>Linaje</dt><dd>{resultado.dataset?.lineage}</dd>
              <dt>Hash</dt><dd className="mono">{(resultado.dataset?.manifest_hash || '—').slice(0, 16)}</dd>
              <dt>Corrida</dt><dd className="mono">{resultado.run_id}</dd>
            </dl>
          </div>
          <div>
            <h4>Parámetros</h4>
            <dl className="lista-datos">
              {Object.entries(resultado.params || {}).map(([clave, valor]) => (
                <div key={clave} style={{ display: 'contents' }}>
                  <dt>{etiquetaMetrica(clave)}</dt>
                  <dd>{Array.isArray(valor) ? valor.join(', ') : String(valor)}</dd>
                </div>
              ))}
            </dl>
          </div>
        </div>
      </Seccion>

      {resultado.notes?.length > 0 && (
        <Seccion titulo="Limitaciones y supuestos">
          <ul className="notas">
            {resultado.notes.map((nota, i) => <li key={i}>{nota}</li>)}
          </ul>
        </Seccion>
      )}

      {historial.length > 1 && (
        <Seccion titulo="Histórico de corridas">
          <table className="tabla">
            <thead>
              <tr><th>Corrida</th><th>Estado</th><th>Fecha</th><th className="num">Duración</th></tr>
            </thead>
            <tbody>
              {historial.slice(0, 10).map((corrida) => (
                <tr key={corrida.run_id}>
                  <td className="mono">{corrida.run_id}</td>
                  <td>{corrida.status === 'success' ? 'Correcta' : 'Falló'}</td>
                  <td>{formatearFecha(corrida.finished_at)}</td>
                  <td className="num">{formatearDuracion(corrida.duration_seconds)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </Seccion>
      )}

      {Object.keys(resultado.artifacts || {}).length > 0 && (
        <Seccion titulo="Artefactos" descripcion="Se descargan a través del backend; el navegador nunca ve las credenciales del bucket.">
          <ul className="artefactos">
            {Object.entries(resultado.artifacts).map(([nombre, ruta]) => {
              const archivo = String(ruta).split('/').pop()
              return (
                <li key={nombre}>
                  <a href={api.urlArtefacto(resultado.model_id, resultado.run_id, archivo)}>
                    {archivo}
                  </a>
                  <span className="nota"> · {etiquetaMetrica(nombre)}</span>
                </li>
              )
            })}
          </ul>
        </Seccion>
      )}

      {desconocidos.length > 0 && (
        <p className="nota">
          Este modelo publica bloques que el tablero todavía no sabe dibujar:{' '}
          {desconocidos.join(', ')}.
        </p>
      )}
    </div>
  )
}

function Seccion({ titulo, descripcion, children }) {
  return (
    <section className="seccion">
      <h3>{titulo}</h3>
      {descripcion && <p className="seccion__descripcion">{descripcion}</p>}
      {children}
    </section>
  )
}
