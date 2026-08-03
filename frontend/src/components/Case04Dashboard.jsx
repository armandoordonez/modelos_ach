import { useDeferredValue, useState } from 'react'
import { RENDERIZADORES, bloquesRenderizables } from './charts/index.jsx'
import {
  ETIQUETA_TAREA,
  etiquetaMetrica,
  formatearFecha,
  formatearMetrica,
  metricasDestacadas,
} from '../formato.js'

const ORDEN_TAREAS = ['clustering', 'regression', 'classification']

export default function Case04Dashboard({
  datos,
  error,
  cargando,
  recargando,
  runId,
  onCambiarCorrida,
  onSeleccionarModelo,
}) {
  const [filtros, setFiltros] = useState({
    period: '',
    model: '',
    segment: '',
    category: '',
  })
  const filtrosDiferidos = useDeferredValue(filtros)

  if (cargando && !datos) return <div className="cargando">Cargando dashboard del Caso 04…</div>

  if (error) {
    return (
      <EstadoPanel
        titulo="No se pudo cargar el dashboard"
        detalle={error}
      />
    )
  }

  if (!datos?.models?.length) {
    return (
      <EstadoPanel
        titulo="Todavía no hay datos para el Caso 04"
        detalle="La vista agregada no encontró resultados ni placeholders configurados."
      />
    )
  }

  const modelosFiltrados = filtrarModelos(datos.models, filtrosDiferidos)
  const disponibles = modelosFiltrados.filter((modelo) => modelo.availability === 'available')
  const pendientes = modelosFiltrados.filter((modelo) => modelo.availability !== 'available')
  const indicadores = construirIndicadores(modelosFiltrados)
  const resumenModelos = datos.models
  const grupos = ORDEN_TAREAS
    .map((taskType) => ({
      taskType,
      disponibles: disponibles.filter((modelo) => modelo.task_type === taskType),
      pendientes: pendientes.filter((modelo) => modelo.task_type === taskType),
    }))
    .filter((grupo) => grupo.disponibles.length || grupo.pendientes.length)

  return (
    <div className="caso04">
      <section className="hero-caso">
        <div>
          <span className="eyebrow">Caso 04</span>
          <h2>{datos.title}</h2>
          <p>{datos.description}</p>
        </div>
        <div className="hero-caso__meta">
          <DatoMeta etiqueta="Corrida activa" valor={runId || datos.latest_run_id || '—'} mono />
          <DatoMeta etiqueta="Última actualización" valor={formatearFecha(datos.generated_at)} />
        </div>
      </section>

      <section className="panel-filtros">
        <div className="panel-filtros__grid">
          <FiltroSelect
            etiqueta="Corrida"
            valor={runId || ''}
            opciones={datos.filters?.runs || []}
            placeholder="Última disponible"
            onChange={(valor) => onCambiarCorrida(valor || null)}
            deshabilitado={recargando}
          />
          <FiltroSelect
            etiqueta="Período"
            valor={filtros.period}
            opciones={datos.filters?.periods || []}
            placeholder="Todos"
            onChange={(valor) => actualizarFiltro(setFiltros, 'period', valor)}
          />
          <FiltroSelect
            etiqueta="Modelo"
            valor={filtros.model}
            opciones={(datos.filters?.models || []).map((modelo) => ({
              id: modelo.model_id,
              label: `${modelo.catalog_ref} · ${modelo.model_name}`,
            }))}
            placeholder="Todos"
            onChange={(valor) => actualizarFiltro(setFiltros, 'model', valor)}
          />
          <FiltroSelect
            etiqueta="Categoría"
            valor={filtros.category}
            opciones={datos.filters?.categories || []}
            placeholder="Todas"
            onChange={(valor) => actualizarFiltro(setFiltros, 'category', valor)}
          />
        </div>

        <div className="panel-filtros__segmentos">
          <span className="panel-filtros__etiqueta">Segmento</span>
          <div className="chips-filtro">
            <button
              className={`chip-filtro ${!filtros.segment ? 'chip-filtro--activo' : ''}`}
              onClick={() => actualizarFiltro(setFiltros, 'segment', '')}
            >
              Todos
            </button>
            {(datos.filters?.segments || []).map((segmento) => (
              <button
                key={segmento.id}
                className={`chip-filtro ${filtros.segment === segmento.id ? 'chip-filtro--activo' : ''}`}
                onClick={() => actualizarFiltro(setFiltros, 'segment', segmento.id)}
              >
                {segmento.label}
              </button>
            ))}
          </div>
        </div>
      </section>

      <section className="resumen-caso">
        {indicadores.map((indicador) => (
          <div key={indicador.etiqueta} className="indicador indicador--panel">
            <span className="indicador__valor">{indicador.valor}</span>
            <span className="indicador__etiqueta">{indicador.etiqueta}</span>
          </div>
        ))}
      </section>

      <section className="catalogo-caso">
        <div className="catalogo-caso__encabezado">
          <h3>Modelos del caso</h3>
          <p>Los placeholders permanecen visibles, pero sin métricas inventadas.</p>
        </div>
        <div className="catalogo-caso__rejilla">
          {resumenModelos.map((modelo) => (
            <ResumenModelo
              key={modelo.model_id}
              modelo={modelo}
              activo={filtros.model === modelo.model_id}
              onSeleccionar={() => actualizarFiltro(setFiltros, 'model', modelo.model_id)}
              onAbrirDetalle={() => onSeleccionarModelo(modelo.model_id)}
            />
          ))}
        </div>
      </section>

      {!modelosFiltrados.length ? (
        <EstadoPanel
          titulo="No hay resultados para esa combinación de filtros"
          detalle="Prueba con otra corrida, período, categoría o segmento."
        />
      ) : (
        grupos.map((grupo) => (
          <TaskTypeSection
            key={grupo.taskType}
            taskType={grupo.taskType}
            disponibles={grupo.disponibles}
            pendientes={grupo.pendientes}
            segmentoActivo={filtros.segment}
            onSeleccionarModelo={onSeleccionarModelo}
          />
        ))
      )}
    </div>
  )
}

function TaskTypeSection({ taskType, disponibles, pendientes, segmentoActivo, onSeleccionarModelo }) {
  return (
    <section className="seccion-caso">
      <div className="seccion-caso__encabezado">
        <div>
          <span className={`chip chip--${taskType}`}>{ETIQUETA_TAREA[taskType] || taskType}</span>
          <h3>{tituloDeSeccion(taskType)}</h3>
        </div>
        <p>{descripcionDeSeccion(taskType)}</p>
      </div>

      {disponibles.length > 0 ? (
        <div className="paneles-modelo">
          {disponibles.map((modelo) => (
            <ModeloDisponiblePanel
              key={modelo.model_id}
              modelo={aplicarFiltroSegmento(modelo, segmentoActivo)}
              segmentoActivo={segmentoActivo}
              onSeleccionarModelo={onSeleccionarModelo}
            />
          ))}
        </div>
      ) : (
        <EstadoEmbebido mensaje="No hay resultados disponibles para este tipo con los filtros actuales." />
      )}

      {pendientes.length > 0 && (
        <div className="paneles-placeholder">
          {pendientes.map((modelo) => (
            <ModeloPendientePanel key={modelo.model_id} modelo={modelo} />
          ))}
        </div>
      )}
    </section>
  )
}

function ModeloDisponiblePanel({ modelo, segmentoActivo, onSeleccionarModelo }) {
  const bloques = bloquesRenderizables(modelo.charts || {})
  const metricas = metricasDestacadas(modelo.task_type, modelo.metrics || {})

  return (
    <article className="panel-modelo">
      <header className="panel-modelo__encabezado">
        <div>
          <div className="panel-modelo__marca">
            <span className="etiqueta-catalogo">{modelo.catalog_ref}</span>
            <span className="estado-modelo estado-modelo--disponible">Disponible</span>
          </div>
          <h4>{modelo.model_name}</h4>
          <p>
            {modelo.category ? `${modelo.category} · ` : ''}
            {modelo.period?.label || 'Sin período reportado'} · {formatearFecha(modelo.finished_at)}
          </p>
        </div>
        <button className="boton" onClick={() => onSeleccionarModelo(modelo.model_id)}>
          Ver detalle técnico
        </button>
      </header>

      {modelo.status === 'failed' && (
        <div className="aviso aviso--error">
          <strong>La última corrida de este modelo falló.</strong>
          <p>{modelo.error || 'Sin detalle adicional.'}</p>
        </div>
      )}

      <div className="metricas metricas--grande">
        {metricas.map(({ nombre, valor }) => (
          <div key={nombre} className="metrica">
            <span className="metrica__valor">{formatearMetrica(nombre, valor)}</span>
            <span className="metrica__nombre">{etiquetaMetrica(nombre)}</span>
          </div>
        ))}
      </div>

      {segmentoActivo && modelo.task_type === 'clustering' && (
        <p className="nota">
          Viendo el segmento <strong>{segmentoActivo}</strong> dentro de este modelo.
        </p>
      )}

      {bloques.length > 0 ? (
        <div className="bloques-dashboard">
          {bloques.map((clave) => {
            const { titulo, componente: Componente } = RENDERIZADORES[clave]
            return (
              <section key={clave} className="bloque-dashboard">
                <div className="bloque-dashboard__encabezado">
                  <h5>{titulo}</h5>
                </div>
                <Componente datos={modelo.charts[clave]} resultado={modelo} />
              </section>
            )
          })}
        </div>
      ) : (
        <EstadoEmbebido mensaje="Este resultado no publicó gráficos reutilizables." />
      )}
    </article>
  )
}

function ModeloPendientePanel({ modelo }) {
  const tono = modelo.availability === 'coming_soon' ? 'pendiente' : 'sin-datos'
  const etiqueta = modelo.availability === 'coming_soon' ? 'Próximamente' : 'Sin resultados'

  return (
    <article className={`panel-placeholder panel-placeholder--${tono}`}>
      <div className="panel-placeholder__cabecera">
        <span className="etiqueta-catalogo">{modelo.catalog_ref}</span>
        <span className={`estado-modelo estado-modelo--${tono}`}>{etiqueta}</span>
      </div>
      <h4>{modelo.model_name}</h4>
      <p>{modelo.category || 'Sin categoría reportada'}</p>
      <p className="nota">{modelo.notes?.[0] || 'Todavía no hay resultados publicados para este modelo.'}</p>
    </article>
  )
}

function ResumenModelo({ modelo, activo, onSeleccionar, onAbrirDetalle }) {
  const disponible = modelo.availability === 'available'

  return (
    <article className={`resumen-modelo ${activo ? 'resumen-modelo--activo' : ''}`}>
      <button className="resumen-modelo__boton" onClick={onSeleccionar}>
        <span className="etiqueta-catalogo">{modelo.catalog_ref}</span>
        <strong>{modelo.model_name}</strong>
        <span className={`estado-modelo estado-modelo--${disponible ? 'disponible' : 'pendiente'}`}>
          {disponible ? 'Disponible' : modelo.availability === 'coming_soon' ? 'Próximamente' : 'Sin resultados'}
        </span>
      </button>
      {disponible && (
        <button className="enlace-accion" onClick={onAbrirDetalle}>
          Abrir detalle
        </button>
      )}
    </article>
  )
}

function FiltroSelect({ etiqueta, valor, opciones, placeholder, onChange, deshabilitado = false }) {
  return (
    <label className="campo-filtro">
      <span>{etiqueta}</span>
      <select value={valor} onChange={(e) => onChange(e.target.value)} disabled={deshabilitado}>
        <option value="">{placeholder}</option>
        {opciones.map((opcion) => (
          <option key={opcion.id} value={opcion.id}>
            {opcion.label}
          </option>
        ))}
      </select>
    </label>
  )
}

function EstadoPanel({ titulo, detalle }) {
  return (
    <div className="vacio">
      <h2>{titulo}</h2>
      <p>{detalle}</p>
    </div>
  )
}

function EstadoEmbebido({ mensaje }) {
  return <div className="estado-embebido">{mensaje}</div>
}

function DatoMeta({ etiqueta, valor, mono = false }) {
  return (
    <div className="dato-meta">
      <span>{etiqueta}</span>
      <strong className={mono ? 'mono' : ''}>{valor}</strong>
    </div>
  )
}

function actualizarFiltro(setFiltros, clave, valor) {
  setFiltros((actual) => ({ ...actual, [clave]: valor }))
}

function filtrarModelos(modelos, filtros) {
  return modelos.filter((modelo) => {
    if (filtros.model && modelo.model_id !== filtros.model) return false
    if (filtros.category && modelo.category !== filtros.category) return false

    if (filtros.period) {
      if (modelo.period?.id && modelo.period.id !== filtros.period) return false
      if (!modelo.period?.id && modelo.availability === 'available') return false
    }

    if (filtros.segment && modelo.task_type === 'clustering') {
      return (modelo.available_segments || []).includes(filtros.segment)
    }

    return true
  })
}

function construirIndicadores(modelos) {
  const disponibles = modelos.filter((modelo) => modelo.availability === 'available')
  const totalEntidades = disponibles.reduce(
    (acc, modelo) => acc + Number(modelo.metrics?.n_entities || 0),
    0,
  )
  return [
    { etiqueta: 'modelos visibles', valor: modelos.length },
    { etiqueta: 'disponibles', valor: disponibles.length },
    { etiqueta: 'próximamente', valor: modelos.filter((modelo) => modelo.availability === 'coming_soon').length },
    { etiqueta: 'sin resultados', valor: modelos.filter((modelo) => modelo.availability === 'no_results').length },
    { etiqueta: 'personas analizadas', valor: formatearMetrica('n_entities', totalEntidades) },
  ]
}

function aplicarFiltroSegmento(modelo, segmento) {
  if (!segmento || modelo.task_type !== 'clustering') return modelo

  const charts = { ...(modelo.charts || {}) }
  if (Array.isArray(charts.segment_distribution)) {
    charts.segment_distribution = charts.segment_distribution.filter((item) => item.label === segmento)
  }
  if (charts.scatter_2d?.points) {
    charts.scatter_2d = {
      ...charts.scatter_2d,
      points: charts.scatter_2d.points.filter((item) => item.label === segmento),
    }
  }
  return {
    ...modelo,
    segments: (modelo.segments || []).filter((item) => item.label === segmento),
    charts,
  }
}

function tituloDeSeccion(taskType) {
  if (taskType === 'clustering') return 'Segmentación y comportamiento por grupos'
  if (taskType === 'regression') return 'Modelos de regresión'
  if (taskType === 'classification') return 'Propensiones y clasificación'
  return taskType
}

function descripcionDeSeccion(taskType) {
  if (taskType === 'clustering') return 'Explora segmentos, tamaños relativos y separación visual.'
  if (taskType === 'regression') return 'Resume modelos que estiman montos o shares continuos.'
  if (taskType === 'classification') return 'Mide propensión, discriminación y señales explicativas.'
  return ''
}
