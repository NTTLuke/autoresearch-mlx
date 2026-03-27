import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer, Cell } from 'recharts'

const COLORS = ['#6366f1', '#8b5cf6', '#a78bfa', '#c4b5fd', '#818cf8', '#7c3aed', '#4f46e5']

function medal(index) {
  if (index === 0) return '🥇'
  if (index === 1) return '🥈'
  if (index === 2) return '🥉'
  return `#${index + 1}`
}

function efficiency(m) {
  if (!m.total_runs || m.total_runs === 0) return 0
  const successRate = (m.kept || 0) / m.total_runs
  const bpbScore = m.best_val_bpb > 0 ? 1 / m.best_val_bpb : 0
  return (successRate * 0.4 + bpbScore * 0.6).toFixed(4)
}

export default function Leaderboard({ models }) {
  if (!models || models.length === 0) {
    return (
      <section className="rounded-xl bg-gray-900 border border-gray-800 p-6">
        <h2 className="text-lg font-semibold mb-2">Leaderboard</h2>
        <p className="text-gray-500 text-sm">No experiment data yet. Run a benchmark to populate the leaderboard.</p>
      </section>
    )
  }

  const chartData = models.map((m) => ({
    name: m.model_id.length > 24 ? m.model_id.slice(0, 22) + '…' : m.model_id,
    val_bpb: Number(m.best_val_bpb?.toFixed(4)),
  }))

  return (
    <section className="rounded-xl bg-gray-900 border border-gray-800 p-6">
      <h2 className="text-lg font-semibold mb-4">Leaderboard</h2>

      <div className="grid grid-cols-1 lg:grid-cols-2 gap-6">
        {/* Chart */}
        <div className="h-64">
          <ResponsiveContainer width="100%" height="100%">
            <BarChart data={chartData} layout="vertical" margin={{ left: 8, right: 24 }}>
              <XAxis type="number" tick={{ fill: '#9ca3af', fontSize: 12 }} />
              <YAxis
                type="category"
                dataKey="name"
                width={160}
                tick={{ fill: '#d1d5db', fontSize: 12 }}
              />
              <Tooltip
                contentStyle={{ background: '#1f2937', border: '1px solid #374151', borderRadius: 8 }}
                labelStyle={{ color: '#e5e7eb' }}
                itemStyle={{ color: '#a5b4fc' }}
                formatter={(v) => [`${v}`, 'val_bpb']}
              />
              <Bar dataKey="val_bpb" radius={[0, 6, 6, 0]}>
                {chartData.map((_, i) => (
                  <Cell key={i} fill={COLORS[i % COLORS.length]} />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        {/* Table */}
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-400 border-b border-gray-800">
                <th className="text-left py-2 px-2">Rank</th>
                <th className="text-left py-2 px-2">Model</th>
                <th className="text-right py-2 px-2">Best val_bpb</th>
                <th className="text-right py-2 px-2">Runs</th>
                <th className="text-right py-2 px-2">Kept</th>
                <th className="text-right py-2 px-2">Crashed</th>
                <th className="text-right py-2 px-2">Efficiency</th>
              </tr>
            </thead>
            <tbody>
              {models.map((m, i) => (
                <tr key={m.model_id} className="border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors">
                  <td className="py-2 px-2 text-center">{medal(i)}</td>
                  <td className="py-2 px-2 font-mono text-xs text-indigo-300 max-w-[200px] truncate" title={m.model_id}>
                    {m.model_id}
                  </td>
                  <td className="py-2 px-2 text-right font-mono font-semibold text-green-400">
                    {m.best_val_bpb?.toFixed(4)}
                  </td>
                  <td className="py-2 px-2 text-right">{m.total_runs}</td>
                  <td className="py-2 px-2 text-right text-green-400">{m.kept}</td>
                  <td className="py-2 px-2 text-right text-red-400">{m.crashed}</td>
                  <td className="py-2 px-2 text-right font-mono text-yellow-300">{efficiency(m)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      </div>
    </section>
  )
}
