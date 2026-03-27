import { useState, useRef, useEffect, useCallback } from 'react'

export default function BenchmarkPanel({ onComplete }) {
  const [modelsInput, setModelsInput] = useState('')
  const [expCount, setExpCount] = useState(5)
  const [includeRandom, setIncludeRandom] = useState(false)
  const [status, setStatus] = useState(null)
  const [logs, setLogs] = useState([])
  const [running, setRunning] = useState(false)
  const logEndRef = useRef(null)
  const eventSourceRef = useRef(null)

  useEffect(() => {
    logEndRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [logs])

  useEffect(() => {
    return () => eventSourceRef.current?.close()
  }, [])

  const pollStatus = useCallback(async () => {
    try {
      const res = await fetch('/api/benchmark/status')
      if (res.ok) setStatus(await res.json())
    } catch {}
  }, [])

  useEffect(() => {
    if (!running) return
    const id = setInterval(pollStatus, 2000)
    return () => clearInterval(id)
  }, [running, pollStatus])

  const startBenchmark = async () => {
    const models = modelsInput
      .split(',')
      .map((s) => s.trim())
      .filter(Boolean)
    if (models.length === 0) return
    if (includeRandom && !models.includes('random-search')) {
      models.push('random-search')
    }

    setLogs([])
    setRunning(true)

    try {
      const res = await fetch('/api/benchmark/start', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ models, experiments_per_model: expCount }),
      })
      const data = await res.json()
      if (data.error) {
        setLogs([{ t: 'error', msg: data.error }])
        setRunning(false)
        return
      }
    } catch (err) {
      setLogs([{ t: 'error', msg: String(err) }])
      setRunning(false)
      return
    }

    const es = new EventSource('/api/benchmark/stream')
    eventSourceRef.current = es

    es.onmessage = (e) => {
      try {
        const payload = JSON.parse(e.data)
        const { event, data } = payload

        if (event === 'stream_end') {
          es.close()
          setRunning(false)
          onComplete?.()
          return
        }
        if (event === 'heartbeat') return

        if (event === 'training_progress') {
          setLogs((prev) => {
            if (prev.length > 0 && prev[prev.length - 1].t === 'training_progress') {
              const updated = [...prev]
              updated[updated.length - 1] = { t: event, msg: data }
              return updated
            }
            return [...prev, { t: event, msg: data }]
          })
          return
        }

        setLogs((prev) => [...prev, { t: event, msg: formatEvent(event, data) }])
        if (event === 'benchmark_done' || event === 'fatal_error') {
          es.close()
          setRunning(false)
          onComplete?.()
        }
      } catch {}
    }

    es.onerror = () => {
      es.close()
      setRunning(false)
    }
  }

  const stopBenchmark = async () => {
    try {
      await fetch('/api/benchmark/stop', { method: 'POST' })
      setLogs((prev) => [...prev, { t: 'info', msg: 'Stop requested — finishing current experiment…' }])
    } catch {}
  }

  const progress = status?.progress || []
  const totalModels = progress.length
  const currentModel = status?.current_model || ''
  const currentExp = status?.current_experiment || 0
  const perModel = status?.experiments_per_model || expCount

  return (
    <section className="rounded-xl bg-gray-900 border border-gray-800 p-6">
      <h2 className="text-lg font-semibold mb-4">Benchmark</h2>

      {/* Controls */}
      <div className="flex flex-wrap items-end gap-4 mb-6">
        <div className="flex-1 min-w-[240px]">
          <label className="block text-xs text-gray-400 mb-1">Model IDs (comma-separated)</label>
          <input
            type="text"
            value={modelsInput}
            onChange={(e) => setModelsInput(e.target.value)}
            disabled={running}
            placeholder="qwen2.5-coder-32b, llama-3.3-70b, deepseek-r1-14b"
            className="w-full rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm
                       placeholder:text-gray-600 focus:outline-none focus:ring-2 focus:ring-indigo-500
                       disabled:opacity-50"
          />
        </div>
        <div className="w-48">
          <label className="block text-xs text-gray-400 mb-1">
            Experiments per model: <span className="text-white font-semibold">{expCount}</span>
          </label>
          <input
            type="range"
            min={1}
            max={20}
            value={expCount}
            onChange={(e) => setExpCount(Number(e.target.value))}
            disabled={running}
            className="w-full accent-indigo-500"
          />
        </div>
        <label className="flex items-center gap-2 self-end cursor-pointer select-none">
          <input
            type="checkbox"
            checked={includeRandom}
            onChange={(e) => setIncludeRandom(e.target.checked)}
            disabled={running}
            className="rounded border-gray-600 bg-gray-800 accent-indigo-500"
          />
          <span className="text-xs text-gray-400">Random baseline</span>
        </label>
        <div className="flex gap-2">
          <button
            onClick={startBenchmark}
            disabled={running || !modelsInput.trim()}
            className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 font-medium text-sm
                       disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            Start
          </button>
          <button
            onClick={stopBenchmark}
            disabled={!running}
            className="px-4 py-2 rounded-lg bg-red-600/80 hover:bg-red-500 font-medium text-sm
                       disabled:opacity-40 disabled:cursor-not-allowed transition-colors"
          >
            Stop
          </button>
        </div>
      </div>

      {/* Progress bars */}
      {running && progress.length > 0 && (
        <div className="space-y-3 mb-4">
          {progress.map((p) => {
            const pct = p.total > 0 ? Math.round((p.completed / p.total) * 100) : 0
            const isCurrent = p.model === currentModel
            return (
              <div key={p.model}>
                <div className="flex items-center justify-between text-xs mb-1">
                  <span className={`font-mono ${isCurrent ? 'text-indigo-300' : 'text-gray-400'}`}>
                    {p.model}
                  </span>
                  <span className="text-gray-500">
                    {p.completed}/{p.total}
                    {isCurrent && ` — running #${currentExp}`}
                  </span>
                </div>
                <div className="h-2 rounded-full bg-gray-800 overflow-hidden">
                  <div
                    className={`h-full rounded-full transition-all duration-500 ${
                      isCurrent ? 'bg-indigo-500' : 'bg-indigo-700'
                    }`}
                    style={{ width: `${pct}%` }}
                  />
                </div>
              </div>
            )
          })}
        </div>
      )}

      {/* Live log */}
      {logs.length > 0 && (
        <div className="rounded-lg bg-gray-950 border border-gray-800 p-3 max-h-64 overflow-y-auto font-mono text-xs space-y-0.5">
          {logs.map((l, i) => (
            <div key={i} className={logColor(l.t)}>
              <span className="text-gray-600 mr-2">{String(i + 1).padStart(3, ' ')}</span>
              {l.msg}
            </div>
          ))}
          <div ref={logEndRef} />
        </div>
      )}
    </section>
  )
}

function formatEvent(event, data) {
  switch (event) {
    case 'model_start':
      return `▶ Starting model: ${data.model}`
    case 'model_done':
      return `✓ Finished model: ${data.model}`
    case 'experiment_start':
      return `  Experiment ${data.experiment}/${data.total} (${data.model})`
    case 'experiment_done':
      return `  ✓ Done — ${(data.summary || '').slice(0, 120)}`
    case 'experiment_error':
      return `  ✗ Error — ${data.error}`
    case 'benchmark_done':
      return data.stopped ? '■ Benchmark stopped by user' : '■ Benchmark complete!'
    case 'fatal_error':
      return `✗ FATAL: ${data.error}`
    default:
      return JSON.stringify(data)
  }
}

function logColor(type) {
  switch (type) {
    case 'model_start':
    case 'model_done':
      return 'text-indigo-400'
    case 'experiment_start':
    case 'training_progress':
      return 'text-gray-400'
    case 'experiment_done':
      return 'text-green-400'
    case 'experiment_error':
    case 'fatal_error':
    case 'error':
      return 'text-red-400'
    case 'benchmark_done':
      return 'text-yellow-300'
    case 'info':
      return 'text-blue-400'
    default:
      return 'text-gray-300'
  }
}
