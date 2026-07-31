import { Bar, BarChart, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { colorPorIndice, formatearMetrica } from '../../formato.js'

export default function SegmentDistribution({ datos }) {
  const filas = [...datos].sort((a, b) => b.n - a.n)
  const total = filas.reduce((acc, s) => acc + s.n, 0)

  return (
    <div className="grafico-con-tabla">
      <ResponsiveContainer width="100%" height={Math.max(200, filas.length * 46)}>
        <BarChart data={filas} layout="vertical" margin={{ left: 8, right: 48, top: 8, bottom: 8 }}>
          <XAxis type="number" tickFormatter={(v) => formatearMetrica('n_x', v)} fontSize={12} />
          <YAxis type="category" dataKey="label" width={210} fontSize={12} tickLine={false} />
          <Tooltip
            formatter={(valor, _n, item) => [
              `${formatearMetrica('n_x', valor)} personas (${(item.payload.share * 100).toFixed(1)}%)`,
              'Tamaño',
            ]}
            cursor={{ fill: 'var(--fondo-sutil)' }}
          />
          <Bar dataKey="n" radius={[0, 3, 3, 0]}>
            {filas.map((fila, i) => (
              <Cell key={fila.id} fill={colorPorIndice(i)} />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <table className="tabla">
        <thead>
          <tr>
            <th>Segmento</th>
            <th className="num">Personas</th>
            <th className="num">% de la base</th>
          </tr>
        </thead>
        <tbody>
          {filas.map((fila, i) => (
            <tr key={fila.id}>
              <td>
                <span className="punto" style={{ background: colorPorIndice(i) }} />
                {fila.label}
              </td>
              <td className="num">{formatearMetrica('n_x', fila.n)}</td>
              <td className="num">{(fila.share * 100).toFixed(1)}%</td>
            </tr>
          ))}
        </tbody>
        <tfoot>
          <tr>
            <td>Total</td>
            <td className="num">{formatearMetrica('n_x', total)}</td>
            <td className="num">100,0%</td>
          </tr>
        </tfoot>
      </table>
    </div>
  )
}
