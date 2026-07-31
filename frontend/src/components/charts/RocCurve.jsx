import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { PALETA } from '../../formato.js'

export default function RocCurve({ datos }) {
  const puntos = (datos.fpr || []).map((fpr, i) => ({
    fpr,
    tpr: datos.tpr?.[i] ?? 0,
    azar: fpr,
  }))
  if (!puntos.length) return null

  return (
    <>
      <ResponsiveContainer width="100%" height={320}>
        <LineChart data={puntos} margin={{ left: 8, right: 16, top: 8, bottom: 16 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--borde-sutil)" />
          <XAxis
            type="number" dataKey="fpr" domain={[0, 1]} fontSize={12}
            tickFormatter={(v) => v.toFixed(1)}
            label={{ value: 'Falsos positivos', position: 'insideBottom', offset: -8, fontSize: 12 }}
          />
          <YAxis
            type="number" domain={[0, 1]} fontSize={12} tickFormatter={(v) => v.toFixed(1)}
            label={{ value: 'Verdaderos positivos', angle: -90, position: 'insideLeft', fontSize: 12 }}
          />
          <Tooltip formatter={(v, n) => [v.toFixed(3), n === 'tpr' ? 'Verdaderos positivos' : 'Azar']} />
          <Line type="monotone" dataKey="tpr" stroke={PALETA[0]} strokeWidth={2.5} dot={false} name="tpr" />
          <Line
            type="linear" dataKey="azar" stroke="var(--texto-tenue)" strokeWidth={1}
            strokeDasharray="4 4" dot={false} name="azar"
          />
        </LineChart>
      </ResponsiveContainer>
      <p className="nota">
        Área bajo la curva: <strong>{datos.auc?.toFixed(4)}</strong>. La diagonal punteada es el azar (0,5).
      </p>
    </>
  )
}
