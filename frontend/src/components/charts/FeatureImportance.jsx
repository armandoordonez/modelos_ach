import { Bar, BarChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { PALETA } from '../../formato.js'

export default function FeatureImportance({ datos }) {
  const filas = [...datos].sort((a, b) => b.importance - a.importance).slice(0, 15)

  return (
    <ResponsiveContainer width="100%" height={Math.max(220, filas.length * 26)}>
      <BarChart data={filas} layout="vertical" margin={{ left: 8, right: 24, top: 8, bottom: 8 }}>
        <XAxis type="number" fontSize={12} tickFormatter={(v) => v.toFixed(3)} />
        <YAxis type="category" dataKey="feature" width={230} fontSize={11} tickLine={false} />
        <Tooltip formatter={(v) => [v.toFixed(5), 'Importancia']} cursor={{ fill: 'var(--fondo-sutil)' }} />
        <Bar dataKey="importance" fill={PALETA[0]} radius={[0, 3, 3, 0]} />
      </BarChart>
    </ResponsiveContainer>
  )
}
