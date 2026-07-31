import {
  CartesianGrid, Legend, ResponsiveContainer, Scatter, ScatterChart, Tooltip, XAxis, YAxis, ZAxis,
} from 'recharts'
import { colorPorIndice } from '../../formato.js'

export default function Scatter2D({ datos }) {
  const puntos = datos.points || []
  if (!puntos.length) return null

  const grupos = new Map()
  puntos.forEach((p) => {
    const etiqueta = p.label ?? 'Sin etiqueta'
    if (!grupos.has(etiqueta)) grupos.set(etiqueta, [])
    grupos.get(etiqueta).push(p)
  })

  return (
    <>
      <ResponsiveContainer width="100%" height={380}>
        <ScatterChart margin={{ left: 8, right: 16, top: 8, bottom: 8 }}>
          <CartesianGrid strokeDasharray="3 3" stroke="var(--borde-sutil)" />
          <XAxis type="number" dataKey="x" name={datos.x_label || 'PC1'} fontSize={12} />
          <YAxis type="number" dataKey="y" name={datos.y_label || 'PC2'} fontSize={12} />
          <ZAxis range={[18, 18]} />
          <Tooltip cursor={{ strokeDasharray: '3 3' }} formatter={(v) => (typeof v === 'number' ? v.toFixed(2) : v)} />
          <Legend wrapperStyle={{ fontSize: 12, paddingTop: 8 }} />
          {[...grupos.entries()].map(([etiqueta, valores], i) => (
            <Scatter key={etiqueta} name={etiqueta} data={valores} fill={colorPorIndice(i)} fillOpacity={0.6} />
          ))}
        </ScatterChart>
      </ResponsiveContainer>
      {datos.explained_variance != null && (
        <p className="nota">
          Las dos componentes recogen el {(datos.explained_variance * 100).toFixed(0)}% de la varianza.
          Muestra de {puntos.length.toLocaleString('es-CO')} personas.
        </p>
      )}
    </>
  )
}
