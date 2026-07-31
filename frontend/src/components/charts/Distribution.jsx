import { Bar, BarChart, CartesianGrid, ResponsiveContainer, Tooltip, XAxis, YAxis } from 'recharts'
import { PALETA, formatearMetrica } from '../../formato.js'

/** Bloque genérico: dibuja cualquier {serie: [numeros]} como histograma simple. */
export default function Distribution({ datos }) {
  const series = Object.entries(datos).filter(([, valores]) => Array.isArray(valores) && valores.length)
  if (!series.length) return null

  return (
    <>
      {series.map(([nombre, valores], indice) => {
        const filas = valores.map((valor, i) => ({ i, valor }))
        return (
          <div key={nombre} className="bloque-serie">
            <h4>{nombre}</h4>
            <ResponsiveContainer width="100%" height={200}>
              <BarChart data={filas} margin={{ left: 8, right: 16, top: 8, bottom: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="var(--borde-sutil)" />
                <XAxis dataKey="i" fontSize={11} />
                <YAxis fontSize={11} tickFormatter={(v) => formatearMetrica('x', v)} />
                <Tooltip formatter={(v) => formatearMetrica('x', v)} />
                <Bar dataKey="valor" fill={PALETA[indice % PALETA.length]} />
              </BarChart>
            </ResponsiveContainer>
          </div>
        )
      })}
    </>
  )
}
