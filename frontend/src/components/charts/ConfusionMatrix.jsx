import { formatearMetrica } from '../../formato.js'

/** Matriz de confusión como mapa de calor sobrio, sin dependencias extra. */
export default function ConfusionMatrix({ datos }) {
  const { labels = [], matrix = [] } = datos
  if (!matrix.length) return null

  const total = matrix.flat().reduce((a, b) => a + b, 0)
  const maximo = Math.max(...matrix.flat(), 1)

  return (
    <div className="matriz-envoltorio">
      <table className="matriz">
        <thead>
          <tr>
            <th className="esquina">Real ╲ Predicho</th>
            {labels.map((etiqueta) => (
              <th key={etiqueta}>{etiqueta}</th>
            ))}
          </tr>
        </thead>
        <tbody>
          {matrix.map((fila, i) => (
            <tr key={labels[i] ?? i}>
              <th>{labels[i] ?? `Clase ${i}`}</th>
              {fila.map((valor, j) => {
                const intensidad = valor / maximo
                const acierto = i === j
                return (
                  <td
                    key={j}
                    className={acierto ? 'celda acierto' : 'celda'}
                    style={{
                      background: `color-mix(in srgb, ${acierto ? 'var(--verde)' : 'var(--rojo)'} ${Math.round(intensidad * 70)}%, var(--fondo-tarjeta))`,
                      color: intensidad > 0.55 ? '#fff' : 'var(--texto)',
                    }}
                    title={`${((valor / total) * 100).toFixed(1)}% del total`}
                  >
                    {formatearMetrica('n_x', valor)}
                  </td>
                )
              })}
            </tr>
          ))}
        </tbody>
      </table>
      <p className="nota">
        La diagonal son los aciertos. Con clases muy desbalanceadas, mirar el recall de la clase
        positiva importa más que la exactitud global.
      </p>
    </div>
  )
}
