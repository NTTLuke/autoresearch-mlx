# Autoresearch Music — Dashboard + Model Benchmark

This document describes the **music adaptation** of autoresearch-mlx: an autonomous ML experimentation pipeline where local LLM agents (via [LM Studio](https://lmstudio.ai)) iteratively modify and train a GPT model on folk melodies in ABC notation, running entirely on Apple Silicon.

A custom **React dashboard** with a FastAPI backend replaces the original AgentOS UI, adding a **benchmark mode** that automatically evaluates multiple LLM models as autonomous researchers and compares their efficiency.

For the original autoresearch-mlx readme, see [README.md](README.md).

## Objective

Use the autoresearch loop to **autonomously discover the best hyperparameters and architecture tweaks** for a small GPT model that generates folk melodies in ABC notation — while tracking which *local LLM model* makes the best research decisions.

The key questions this experiment answers:

1. What model/optimizer configuration minimizes `val_bpb` on a real music dataset within a fixed training budget?
2. Does the choice of local LLM (the "researcher brain") affect the quality of experiments it designs?
3. Given N candidate models, **which one is the most efficient autonomous researcher?**

## The GPT Model

The model being trained is a **decoder-only GPT** implemented in MLX (`train.py`), based on the architecture from Karpathy's autoresearch. It's a small, single-file transformer designed to run and iterate quickly on Apple Silicon.
The agent autonomously modifies hyperparameters like `DEPTH`, `ASPECT_RATIO`, `HEAD_DIM`, learning rates, batch sizes, warmup/warmdown ratios, and weight decay across experiments, searching for the configuration that minimizes `val_bpb` (validation bits per byte).

## Architecture

```
Browser ──> React Dashboard (localhost:8000)
                │
                ├── GET  /api/experiments ── experiment history (with ?model_id= filter)
                ├── GET  /api/models ────── leaderboard (best val_bpb per model)
                ├── POST /api/benchmark/start ── kick off multi-model benchmark
                ├── GET  /api/benchmark/stream ── SSE live log
                ├── GET  /api/benchmark/status ── progress JSON
                ├── POST /api/benchmark/stop ──── graceful stop
                └── POST /api/sample ──────────── generate music
                │
           FastAPI (api.py)
                │
                ├── Benchmark Engine (background thread)
                │       └── For each model: create Researcher agent → run N experiments
                │               ├── read_file / write_file (train.py, results.tsv)
                │               ├── run_shell (git, uv run train.py)
                │               └── log_experiment → SQLite
                │
                ├── SQLite (autoresearch.db) — experiment results
                └── LM Studio (localhost:1234) — local LLM inference
```

### Dashboard Sections

| Section              | What it shows                                                                     |
| -------------------- | --------------------------------------------------------------------------------- |
| **Leaderboard**      | Bar chart + table: best val_bpb per model, runs, kept/crashed, efficiency         |
| **Benchmark Panel**  | Start/stop controls, model input, experiment slider, random baseline toggle, live log with training progress (0-100%) |
| **Experiment Table** | Sortable/filterable history — click a row to expand full config snapshot           |
| **Try the Winner**   | Generate music using the best-performing model's weights with preset prompts       |

### Benchmark Mode

The benchmark engine runs in a background thread. For each model you specify:

1. Creates a fresh Researcher agent configured to call LM Studio with that model ID
2. Runs N experiments sequentially — each builds on the previous result
3. Streams progress events (model start/done, experiment start/done/error, **training %**) via SSE
4. All results logged to the `experiments` table with the model_id and a **`config_json`** snapshot

**Random baseline**: check the "Random baseline" toggle in the benchmark panel to add a
`random-search` participant that picks random hyperparameter values — useful as a
control group to see if LLM-guided search is actually better than chance.

After the benchmark completes, the leaderboard shows which model achieved the lowest `val_bpb` and which was the most efficient researcher overall.

### Experiment Tracking

Every experiment is logged to an `experiments` table in `autoresearch.db`:

| Column        | Description                                                    |
| ------------- | -------------------------------------------------------------- |
| `model_id`    | LM Studio model that drove the experiment (or `random-search`) |
| `val_bpb`     | Validation bits per byte (lower is better)                     |
| `memory_gb`   | Peak VRAM usage                                                |
| `status`      | `keep`, `discard`, or `crash`                                  |
| `description` | What was changed                                               |
| `config_json` | Full hyperparameter snapshot (JSON) at end of experiment       |

## Dataset

The music dataset comes from [MelodyHub](https://huggingface.co/datasets/sander-wood/melodyhub)
(MIT license), a curated collection of 261,900 public-domain folk melodies in
ABC notation hosted on Hugging Face. The `make_music_dataset.py` script:

1. Downloads the MelodyHub parquet files from Hugging Face.
2. Filters for the "generation" task subset — one complete ABC score per row.
3. Strips MelodyHub-specific control codes (S:/B:/E:/X:), keeping standard ABC.
4. Deduplicates and splits into 4 training shards + 1 validation shard.

Each document looks like:

```
L:1/8
M:4/4
K:D
"D" f2 fedB | A2 AF A2 |"G" B2 Bd B2 |"D" A2 AF A2 |...
```

## Setup

### Prerequisites

- Apple Silicon Mac (M1/M2/M3/M4)
- Python 3.10+, Node.js 18+
- [uv](https://docs.astral.sh/uv/)
- [LM Studio](https://lmstudio.ai/) with a model loaded

### Install

```bash
# Python dependencies
uv sync
uv pip install agno openai sqlalchemy "fastapi[standard]" sse-starlette

# Frontend dependencies
cd frontend && npm install && cd ..
```

### Prepare the Dataset

```bash
uv run python make_music_dataset.py
uv run python prepare.py --num-shards 4
```

### Run a Baseline Training

```bash
uv run python train.py
```

### Launch the Dashboard

Start LM Studio and load a model, then:

```bash
# Build the React frontend (one-time or after changes)
cd frontend && npm run build && cd ..

# Start the dashboard
uv run python api.py

# Or with custom options
uv run python api.py --port 8000 --base-url http://localhost:1234/v1
```

Open **[http://localhost:8000](http://localhost:8000)** in your browser.

### Development Mode

For frontend hot-reload during development:

```bash
# Terminal 1 — backend
uv run python api.py

# Terminal 2 — frontend dev server (proxies API to :8000)
cd frontend && npm run dev
```

Then open **http://localhost:5173** (Vite dev server).

### Running a Benchmark

1. Open the dashboard
2. In the **Benchmark** panel, enter model IDs (comma-separated), e.g.: `qwen2.5-coder-32b, llama-3.3-70b, deepseek-r1-14b`
3. Adjust the **experiments per model** slider (1-20)
4. Click **Start**
5. Watch the live log and progress bars as each model runs experiments
6. When complete, the **Leaderboard** updates with final rankings

### Generate Music (CLI)

```bash
uv run python sample_music.py
uv run python sample_music.py --prompt "L:1/8
M:6/8
K:G
"
uv run python sample_music.py --max-new-tokens 300 --temperature 0.9
```

## File Overview

| File                    | Description                                            |
| ----------------------- | ------------------------------------------------------ |
| `api.py`                | FastAPI backend — dashboard API + benchmark engine      |
| `run_autoresearch.py`   | Agent tools, prompts, and experiment runner (library)   |
| `experiments_db.py`     | SQLite experiment tracker (model comparison)            |
| `frontend/`             | React dashboard (Vite + Tailwind + Recharts)           |
| `make_music_dataset.py` | Downloads and shards MelodyHub music data              |
| `prepare.py`            | Tokenizer training and data loading (DO NOT EDIT)      |
| `train.py`              | Model + training loop (agent edits this)               |
| `sample_music.py`       | Generate ABC music from trained weights                |
| `program.md`            | Autoresearch protocol spec (system prompt)             |
| `results.tsv`           | TSV log of experiment history                          |
| `autoresearch.db`       | SQLite database (experiment results)                   |

## Comparing Local Models

After running a benchmark, the leaderboard shows which model achieved the best
results. You can also query programmatically:

```python
from experiments_db import get_best_by_model
for row in get_best_by_model():
    print(f"{row['model_id']}: best val_bpb={row['best_val_bpb']:.6f} "
          f"({row['total_runs']} runs, {row['kept']} kept)")
```

## Acknowledgments

- [Andrej Karpathy](https://github.com/karpathy) — autoresearch and the
autonomous research loop concept
- [MelodyHub / MelodyT5](https://huggingface.co/datasets/sander-wood/melodyhub) — curated
ABC notation dataset (Shangda Wu et al.)
- [Agno](https://github.com/agno-agi/agno) — agent framework
- [LM Studio](https://lmstudio.ai/) — local LLM inference
- [Apple MLX](https://github.com/ml-explore/mlx) — Apple Silicon ML framework

## License

MIT. See [LICENSE](LICENSE).
