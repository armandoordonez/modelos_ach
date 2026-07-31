import { ETIQUETA_TAREA, formatearDuracion, formatearFecha, formatearMetrica, etiquetaMetrica, metricasDestacadas } from '../formato.js'

/**
 * Tarjeta de un modelo. No conoce ningún modelo en particular: toma el tipo de tarea
 * y las métricas que vengan en el JSON.
 */
export default function ModelCard({ modelo, onSeleccionar }) {
  const fallido = modelo.status !== 'success'
  const destacadas = metricasDestacadas(modelo.task_type, modelo.metrics)

  return (
    <article
      className={`tarjeta ${fallido ? 'tarjeta--fallida' : ''}`}
      onClick={() => onSeleccionar(modelo.model_id)}
      onKeyDown={(e) => e.key === 'Enter' && onSeleccionar(modelo.model_id)}
      role="button"
      tabIndex={0}
    >
      <header className="tarjeta__encabezado">
        <div>
          <span className="etiqueta-catalogo">{modelo.catalog_ref || '—'}</span>
          <h3>{modelo.model_name}</h3>
        </div>
        <span className={`chip chip--${modelo.task_type}`}>
          {ETIQUETA_TAREA[modelo.task_type] || modelo.task_type}
        </span>
      </header>

      {fallido ? (
        <div className="aviso aviso--error">
          <strong>La última corrida falló.</strong>
          <p>{modelo.error?.split('\n')[0] || 'Sin detalle disponible.'}</p>
        </div>
      ) : (
        <div className="metricas">
          {destacadas.map(({ nombre, valor }) => (
            <div key={nombre} className="metrica">
              <span className="metrica__valor">{formatearMetrica(nombre, valor)}</span>
              <span className="metrica__nombre">{etiquetaMetrica(nombre)}</span>
            </div>
          ))}
        </div>
      )}

      <footer className="tarjeta__pie">
        <span>Caso {modelo.use_case}</span>
        <span>{formatearFecha(modelo.finished_at)}</span>
        <span>{formatearDuracion(modelo.duration_seconds)}</span>
      </footer>
    </article>
  )
}
