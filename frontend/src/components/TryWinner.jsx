import { useState, useEffect, useCallback } from 'react'

const PRESETS = [
  { label: 'Reel in D (4/4)', prompt: 'L:1/8\nM:4/4\nK:D\n' },
  { label: 'Jig in G (6/8)', prompt: 'L:1/8\nM:6/8\nK:G\n' },
  { label: 'Waltz in A (3/4)', prompt: 'L:1/4\nM:3/4\nK:A\n' },
  { label: 'Polka in C (2/4)', prompt: 'L:1/8\nM:2/4\nK:C\n' },
]

export default function TryWinner({ selectedExpId }) {
  const [experiments, setExperiments] = useState([])
  const [expId, setExpId] = useState(null)
  const [prompt, setPrompt] = useState(PRESETS[0].prompt)
  const [maxTokens, setMaxTokens] = useState(200)
  const [temperature, setTemperature] = useState(0.8)
  const [output, setOutput] = useState('')
  const [loading, setLoading] = useState(false)

  const fetchExperiments = useCallback(async () => {
    try {
      const res = await fetch('/api/experiments/with-weights')
      if (res.ok) {
        const data = await res.json()
        setExperiments(data)
        if (data.length > 0 && expId === null) {
          setExpId(data[0].id)
        }
      }
    } catch {}
  }, [expId])

  useEffect(() => {
    fetchExperiments()
    const id = setInterval(fetchExperiments, 10000)
    return () => clearInterval(id)
  }, [fetchExperiments])

  useEffect(() => {
    if (selectedExpId != null) {
      setExpId(selectedExpId)
    }
  }, [selectedExpId])

  const generate = async () => {
    if (expId == null) return
    setLoading(true)
    setOutput('')
    try {
      const res = await fetch('/api/sample', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          prompt,
          max_tokens: maxTokens,
          temperature,
          experiment_id: expId,
        }),
      })
      const data = await res.json()
      setOutput(data.output || 'No output')
    } catch (err) {
      setOutput(`Error: ${err}`)
    } finally {
      setLoading(false)
    }
  }

  const selected = experiments.find((e) => e.id === expId)

  return (
    <section id="try-winner" className="rounded-xl bg-gray-900 border border-gray-800 p-6">
      <h2 className="text-lg font-semibold mb-4">Try the Winner</h2>

      {experiments.length === 0 ? (
        <p className="text-gray-500 text-sm">
          No trained weights available yet. Run a benchmark first — weights are saved
          for each successful experiment.
        </p>
      ) : (
        <>
          <div className="flex flex-wrap items-end gap-4 mb-4">
            <div className="flex-1 min-w-[260px]">
              <label className="block text-xs text-gray-400 mb-1">Select Experiment Weights</label>
              <select
                value={expId ?? ''}
                onChange={(e) => setExpId(Number(e.target.value))}
                className="w-full rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm
                           font-mono focus:outline-none focus:ring-2 focus:ring-indigo-500"
              >
                {experiments.map((e) => (
                  <option key={e.id} value={e.id}>
                    Exp #{e.id} — val_bpb={e.val_bpb?.toFixed(4)} — {e.model_id}
                    {e.id === experiments[0]?.id ? ' (best)' : ''}
                  </option>
                ))}
              </select>
            </div>
          </div>

          {selected && (
            <p className="text-xs text-gray-500 mb-4 font-mono">
              {selected.description?.slice(0, 100)}
              {' | '}weights: {selected.weights_path}
            </p>
          )}

          <div className="flex flex-wrap items-end gap-4 mb-4">
            <div className="flex-1 min-w-[200px]">
              <label className="block text-xs text-gray-400 mb-1">ABC Prompt</label>
              <textarea
                value={prompt}
                onChange={(e) => setPrompt(e.target.value)}
                rows={3}
                className="w-full rounded-lg bg-gray-800 border border-gray-700 px-3 py-2 text-sm
                           font-mono placeholder:text-gray-600 focus:outline-none focus:ring-2
                           focus:ring-indigo-500 resize-none"
              />
            </div>
            <div className="flex flex-col gap-2">
              <div>
                <label className="block text-xs text-gray-400 mb-1">
                  Tokens: <span className="text-white">{maxTokens}</span>
                </label>
                <input type="range" min={50} max={500} value={maxTokens}
                  onChange={(e) => setMaxTokens(Number(e.target.value))}
                  className="w-32 accent-indigo-500" />
              </div>
              <div>
                <label className="block text-xs text-gray-400 mb-1">
                  Temp: <span className="text-white">{temperature.toFixed(1)}</span>
                </label>
                <input type="range" min={0} max={15} value={Math.round(temperature * 10)}
                  onChange={(e) => setTemperature(Number(e.target.value) / 10)}
                  className="w-32 accent-indigo-500" />
              </div>
            </div>
            <button
              onClick={generate}
              disabled={loading || expId == null}
              className="px-4 py-2 rounded-lg bg-indigo-600 hover:bg-indigo-500 font-medium text-sm
                         disabled:opacity-40 disabled:cursor-not-allowed transition-colors self-end"
            >
              {loading ? 'Generating...' : 'Generate'}
            </button>
          </div>

          <div className="flex flex-wrap gap-2 mb-4">
            {PRESETS.map((p) => (
              <button
                key={p.label}
                onClick={() => setPrompt(p.prompt)}
                className="px-3 py-1 rounded-full text-xs bg-gray-800 border border-gray-700
                           hover:border-indigo-500 transition-colors"
              >
                {p.label}
              </button>
            ))}
          </div>

          {output && (
            <pre className="rounded-lg bg-gray-950 border border-gray-800 p-4 text-sm font-mono
                            text-green-300 overflow-x-auto whitespace-pre-wrap max-h-64 overflow-y-auto">
              {output}
            </pre>
          )}
        </>
      )}
    </section>
  )
}
