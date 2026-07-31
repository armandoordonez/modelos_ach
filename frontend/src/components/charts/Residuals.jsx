import {
  CartesianGrid, Line, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from 'recharts'
import { PALETA } from '../../formato.js'

export default function Residuals({ datos }) {
  const reales = datos.actual || []
  const predichos = datos.predicted || []
  if (!reales.length || reales.length !== predichos.length) return null

  const puntos = reales.map((real, i) => ({ real, predicho: predichos[i] }))
  const minimo = Math.min(...reales, ...predichos)
  const maximo = Math.max(...reales, ...predichos)

  return (
    <>
      <ResponsiveContainer width="100%" height={340}>
        <ScatterChart margin={{ left: 8, right: 16, top: 8, bottom: 16 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--borde-sutil)" />
          <XAxis
            type="number" dataKey="real" domain={[minimo, maximo]} fontSize={12}
            tickFormatter={(v) => v.toFixed(1)}
            label={{ value: 'Valor real', position: 'insideBottom', offset: -8, fontSize: 12 }}
          />
          <YAxis
            type="number" dataKey="predicho" domain={[minimo, maximo]} fontSize={12}
            tickFormatter={(v) => v.toFixed(1)}
            label={{ value: 'Predicho', angle: -90, position: 'insideLeft', fontSize: 12 }}
          />
          <ZAxis range={[14, 14]} />
          <Tooltip formatter={(v) => (typeof v === 'number' ? v.toFixed(3) : v)} cursor={{ strokeDasharray: '3 3' }} />
          <Scatter data={puntos} fill={PALETA[0]} fillOpacity={0.45} />
          <Line
            type="linear" dataKey="real" data={[{ real: minimo }, { real: maximo }]}
            stroke="var(--texto-tenue)" strokeDasharray="4 4" dot={false} legendType="none"
          />
        </ScatterChart>
      </ResponsiveContainer>
      <p className="nota">
        Cuanto más cerca de la diagonal, mejor la predicción. Escala logarítmica del objetivo.
        Muestra de {puntos.length.toLocaleString('es-CO')} casos de prueba.
      </p>
    </>
  )
}
