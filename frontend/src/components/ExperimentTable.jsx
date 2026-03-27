import { useState, useMemo } from 'react'

const SORTABLE = ['id', 'timestamp', 'model_id', 'val_bpb', 'memory_gb', 'status']

function statusBadge(status) {
  const base = 'inline-block px-2 py-0.5 rounded-full text-[10px] font-semibold uppercase tracking-wider'
  switch (status) {
    case 'keep':
      return <span className={`${base} bg-green-900/60 text-green-300`}>keep</span>
    case 'discard':
      return <span className={`${base} bg-yellow-900/60 text-yellow-300`}>discard</span>
    case 'crash':
      return <span className={`${base} bg-red-900/60 text-red-300`}>crash</span>
    default:
      return <span className={`${base} bg-gray-800 text-gray-400`}>{status}</span>
  }
}

function fmtTime(ts) {
  if (!ts) return '—'
  const d = new Date(ts * 1000)
  return d.toLocaleString(undefined, {
    month: 'short',
    day: 'numeric',
    hour: '2-digit',
    minute: '2-digit',
  })
}

function parseChange(description) {
  const m = description?.match(/:\s*(\w+):\s*(.+?)\s*→\s*(.+?)(?:\s*→|$)/)
  if (m) return { param: m[1], from: m[2], to: m[3] }
  return null
}

export default function ExperimentTable({ experiments, models, modelFilter, onFilterChange, onTryExperiment }) {
  const [sortKey, setSortKey] = useState('id')
  const [sortAsc, setSortAsc] = useState(false)
  const [expanded, setExpanded] = useState(null)

  const sorted = useMemo(() => {
    const copy = [...experiments]
    copy.sort((a, b) => {
      const va = a[sortKey]
      const vb = b[sortKey]
      if (va == null) return 1
      if (vb == null) return -1
      if (typeof va === 'number') return sortAsc ? va - vb : vb - va
      return sortAsc ? String(va).localeCompare(String(vb)) : String(vb).localeCompare(String(va))
    })
    return copy
  }, [experiments, sortKey, sortAsc])

  const handleSort = (key) => {
    if (sortKey === key) {
      setSortAsc(!sortAsc)
    } else {
      setSortKey(key)
      setSortAsc(key === 'id')
    }
  }

  const uniqueModels = useMemo(() => {
    return [...new Set(experiments.map((e) => e.model_id))].sort()
  }, [experiments])

  const arrow = (key) => {
    if (sortKey !== key) return ''
    return sortAsc ? ' ↑' : ' ↓'
  }

  return (
    <section className="rounded-xl bg-gray-900 border border-gray-800 p-6">
      <div className="flex items-center justify-between mb-4">
        <h2 className="text-lg font-semibold">Experiment History</h2>
        <div className="flex items-center gap-2">
          <label className="text-xs text-gray-400">Filter by model:</label>
          <select
            value={modelFilter}
            onChange={(e) => onFilterChange(e.target.value)}
            className="rounded-lg bg-gray-800 border border-gray-700 px-2 py-1 text-sm
                       focus:outline-none focus:ring-2 focus:ring-indigo-500"
          >
            <option value="">All models</option>
            {uniqueModels.map((m) => (
              <option key={m} value={m}>{m}</option>
            ))}
          </select>
        </div>
      </div>

      {sorted.length === 0 ? (
        <p className="text-gray-500 text-sm">No experiments recorded yet.</p>
      ) : (
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-gray-400 border-b border-gray-800">
                {[
                  ['id', '#'],
                  ['timestamp', 'Time'],
                  ['model_id', 'Model'],
                  ['change', 'Change'],
                  ['val_bpb', 'val_bpb'],
                  ['memory_gb', 'Mem (GB)'],
                  ['status', 'Status'],
                  ['actions', ''],
                ].map(([key, label]) => (
                  <th
                    key={key}
                    onClick={() => SORTABLE.includes(key) && handleSort(key)}
                    className={`text-left py-2 px-2 ${
                      SORTABLE.includes(key)
                        ? 'cursor-pointer hover:text-gray-200 select-none'
                        : ''
                    } ${key === 'val_bpb' || key === 'memory_gb' ? 'text-right' : ''}`}
                  >
                    {label}{arrow(key)}
                  </th>
                ))}
              </tr>
            </thead>
            <tbody>
              {sorted.map((e) => {
                const change = parseChange(e.description)
                const isExpanded = expanded === e.id
                const hasWeights = !!e.weights_path
                let config = null
                try { config = e.config_json ? JSON.parse(e.config_json) : null } catch {}

                return (
                  <>
                    <tr
                      key={e.id}
                      onClick={() => setExpanded(isExpanded ? null : e.id)}
                      className="border-b border-gray-800/50 hover:bg-gray-800/30 transition-colors cursor-pointer"
                    >
                      <td className="py-2 px-2 text-gray-500">{e.id}</td>
                      <td className="py-2 px-2 text-gray-400 text-xs">{fmtTime(e.timestamp)}</td>
                      <td className="py-2 px-2 font-mono text-xs text-indigo-300 max-w-[140px] truncate" title={e.model_id}>
                        {e.model_id}
                      </td>
                      <td className="py-2 px-2 text-xs font-mono">
                        {change ? (
                          <span>
                            <span className="text-yellow-300">{change.param}</span>
                            {' '}<span className="text-gray-500">{change.from}</span>
                            {' → '}<span className="text-white">{change.to}</span>
                          </span>
                        ) : (
                          <span className="text-gray-500 truncate max-w-[180px] inline-block" title={e.description}>
                            {e.description?.slice(0, 40)}
                          </span>
                        )}
                      </td>
                      <td className="py-2 px-2 text-right font-mono font-semibold text-green-400">
                        {e.val_bpb < 99 ? e.val_bpb?.toFixed(4) : <span className="text-red-400">--</span>}
                      </td>
                      <td className="py-2 px-2 text-right font-mono text-gray-300">
                        {e.memory_gb > 0 ? e.memory_gb?.toFixed(1) : '--'}
                      </td>
                      <td className="py-2 px-2">{statusBadge(e.status)}</td>
                      <td className="py-2 px-2">
                        {hasWeights && (
                          <button
                            onClick={(ev) => {
                              ev.stopPropagation()
                              onTryExperiment?.(e.id)
                            }}
                            className="px-2 py-0.5 rounded text-[10px] font-semibold uppercase
                                       bg-indigo-600/40 text-indigo-300 hover:bg-indigo-500/60
                                       transition-colors"
                            title="Generate music with this experiment's weights"
                          >
                            Try
                          </button>
                        )}
                      </td>
                    </tr>
                    {isExpanded && config && Object.keys(config).length > 0 && (
                      <tr key={`${e.id}-config`} className="bg-gray-800/20">
                        <td colSpan={8} className="px-4 py-3">
                          <div className="text-xs font-mono text-gray-400 flex flex-wrap gap-x-6 gap-y-1">
                            {Object.entries(config).map(([k, v]) => (
                              <span key={k}>
                                <span className={change?.param === k ? 'text-yellow-300' : 'text-gray-500'}>{k}</span>
                                {'='}
                                <span className={change?.param === k ? 'text-white' : 'text-gray-300'}>{v}</span>
                              </span>
                            ))}
                            {hasWeights && (
                              <span className="text-green-600 ml-4">
                                weights: {e.weights_path}
                              </span>
                            )}
                          </div>
                        </td>
                      </tr>
                    )}
                  </>
                )
              })}
            </tbody>
          </table>
        </div>
      )}
    </section>
  )
}
