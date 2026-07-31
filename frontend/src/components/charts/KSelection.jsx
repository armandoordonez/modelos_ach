import { CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { PALETA } from '../../formato.js'

const SERIES = [
  { clave: 'silhouette', nombre: 'Silueta (mayor es mejor)', color: PALETA[1] },
  { clave: 'davies_bouldin', nombre: 'Davies-Bouldin (menor es mejor)', color: PALETA[5] },
]

export default function KSelection({ datos, resultado }) {
  const kElegido = resultado?.metrics?.k
  const presentes = SERIES.filter((s) => datos.some((d) => d[s.clave] != null))
  if (!presentes.length) return null

  return (
    <ResponsiveContainer width="100%" height={260}>
      <LineChart data={datos} margin={{ left: 8, right: 16, top: 8, bottom: 8 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="var(--borde-sutil)" />
        <XAxis dataKey="k" fontSize={12} label={{ value: 'k', position: 'insideBottom', offset: -2, fontSize: 12 }} />
        <YAxis fontSize={12} />
        <Tooltip
          formatter={(v, n) => [typeof v === 'number' ? v.toFixed(4) : v, n]}
          labelFormatter={(k) => `k = ${k}${k === kElegido ? ' · elegido' : ''}`}
        />
        {presentes.map((serie) => (
          <Line
            key={serie.clave}
            type="monotone"
            dataKey={serie.clave}
            name={serie.nombre}
            stroke={serie.color}
            strokeWidth={2}
            dot={{ r: 3 }}
            activeDot={{ r: 5 }}
          />
        ))}
      </LineChart>
    </ResponsiveContainer>
  )
}
