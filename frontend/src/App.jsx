import { useState, useEffect, useCallback } from 'react'
import Leaderboard from './components/Leaderboard'
import BenchmarkPanel from './components/BenchmarkPanel'
import ExperimentTable from './components/ExperimentTable'
import TryWinner from './components/TryWinner'

const API = '/api'

export default function App() {
  const [models, setModels] = useState([])
  const [experiments, setExperiments] = useState([])
  const [modelFilter, setModelFilter] = useState('')
  const [selectedExpId, setSelectedExpId] = useState(null)

  const refresh = useCallback(async () => {
    try {
      const [mRes, eRes] = await Promise.all([
        fetch(`${API}/models`),
        fetch(`${API}/experiments${modelFilter ? `?model_id=${encodeURIComponent(modelFilter)}` : ''}`),
      ])
      if (mRes.ok) setModels(await mRes.json())
      if (eRes.ok) setExperiments(await eRes.json())
    } catch (err) {
      console.error('Failed to fetch data:', err)
    }
  }, [modelFilter])

  useEffect(() => {
    refresh()
    const id = setInterval(refresh, 5000)
    return () => clearInterval(id)
  }, [refresh])

  const handleTryExperiment = (expId) => {
    setSelectedExpId(expId)
    document.getElementById('try-winner')?.scrollIntoView({ behavior: 'smooth' })
  }

  return (
    <div className="min-h-screen bg-gray-950 text-gray-100">
      <header className="border-b border-gray-800 bg-gray-900/80 backdrop-blur sticky top-0 z-10">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <span className="text-2xl">🎵</span>
            <h1 className="text-xl font-semibold tracking-tight">
              Autoresearch <span className="text-indigo-400">Dashboard</span>
            </h1>
          </div>
          <span className="text-xs text-gray-500 font-mono">MLX · Apple Silicon</span>
        </div>
      </header>

      <main className="max-w-7xl mx-auto px-6 py-8 space-y-8">
        <Leaderboard models={models} />
        <BenchmarkPanel onComplete={refresh} />
        <ExperimentTable
          experiments={experiments}
          models={models}
          modelFilter={modelFilter}
          onFilterChange={setModelFilter}
          onTryExperiment={handleTryExperiment}
        />
        <TryWinner selectedExpId={selectedExpId} />
      </main>
    </div>
  )
}
