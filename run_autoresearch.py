"""
Autoresearch engine — tools, prompts, and benchmark runner.

This module is imported by api.py. It does NOT start any server on its own.
All agent creation, experiment execution, and tool definitions live here.
"""

from __future__ import annotations

import json
import random
import re
import shutil
import subprocess
import textwrap
import time
from pathlib import Path

from agno.agent import Agent
from agno.models.lmstudio import LMStudio
from agno.tools import tool

from experiments_db import (
    init_db,
    log_experiment as db_log_experiment,
    get_experiments as db_get_experiments,
    get_best_by_model as db_get_best_by_model,
    update_weights_path as db_update_weights_path,
)

# ---------------------------------------------------------------------------
# Paths & constants
# ---------------------------------------------------------------------------

PROJECT_DIR = Path(__file__).resolve().parent
DB_FILE = str(PROJECT_DIR / "autoresearch.db")
ALLOWED_WRITE = {"train.py", "results.tsv"}
SHELL_TIMEOUT = 900

init_db(DB_FILE)

# ---------------------------------------------------------------------------
# Tools  (model_id is injected at call-time via closure)
# ---------------------------------------------------------------------------

_active_model_id: str = "unknown"
_experiment_logged: bool = False


def set_active_model(model_id: str) -> None:
    global _active_model_id, _experiment_logged
    _active_model_id = model_id
    _experiment_logged = False


@tool
def read_file(path: str) -> str:
    """Read a file from the project directory.
    Use relative paths: 'train.py', 'results.tsv', 'prepare.py', 'run.log'.
    """
    target = (PROJECT_DIR / path).resolve()
    if not str(target).startswith(str(PROJECT_DIR)):
        return f"ERROR: path escapes project directory: {path}"
    if not target.exists():
        return f"ERROR: file not found: {path}"
    try:
        text = target.read_text(encoding="utf-8", errors="replace")
        if len(text) > 30_000:
            return text[:30_000] + f"\n\n... [truncated, {len(text)} chars total]"
        return text
    except Exception as e:
        return f"ERROR reading {path}: {e}"


@tool
def write_file(path: str, content: str) -> str:
    """Write content to a project file. Only train.py and results.tsv are writable."""
    if path not in ALLOWED_WRITE:
        return f"ERROR: writing to '{path}' is not allowed. Only {sorted(ALLOWED_WRITE)} can be modified."
    try:
        (PROJECT_DIR / path).write_text(content, encoding="utf-8")
        return f"OK: wrote {len(content)} chars to {path}"
    except Exception as e:
        return f"ERROR writing {path}: {e}"


@tool
def run_shell(command: str) -> str:
    """Run a shell command in the project directory.
    Use for: git commands, 'uv run train.py > run.log 2>&1', grep, etc.
    """
    try:
        result = subprocess.run(
            command, shell=True, capture_output=True, text=True,
            timeout=SHELL_TIMEOUT, cwd=PROJECT_DIR,
        )
        output = result.stdout + result.stderr
        if len(output) > 15_000:
            output = output[:15_000] + "\n\n... [truncated]"
        return f"exit_code={result.returncode}\n{output}"
    except subprocess.TimeoutExpired:
        return f"ERROR: command timed out after {SHELL_TIMEOUT}s"
    except Exception as e:
        return f"ERROR: {e}"


@tool
def log_experiment(val_bpb: float, memory_gb: float, status: str, description: str) -> str:
    """Log a completed experiment to the persistent database.
    status must be one of: keep, discard, crash.
    The LM Studio model ID is recorded automatically.
    """
    global _experiment_logged
    if status not in ("keep", "discard", "crash"):
        return f"ERROR: status must be keep/discard/crash, got '{status}'"
    row_id = db_log_experiment(
        model_id=_active_model_id,
        val_bpb=val_bpb,
        memory_gb=memory_gb,
        status=status,
        description=description,
        db_path=DB_FILE,
    )
    _experiment_logged = True
    print(f"[log_experiment] #{row_id} model={_active_model_id} val_bpb={val_bpb} status={status}")
    return f"OK: logged experiment #{row_id} (model={_active_model_id}, val_bpb={val_bpb}, status={status})"


@tool
def compare_models() -> str:
    """Show the best val_bpb achieved by each LM Studio model."""
    rows = db_get_best_by_model(DB_FILE)
    if not rows:
        return "No experiments logged yet."
    lines = ["model_id | best_val_bpb | total_runs | kept | crashed", "-" * 60]
    for r in rows:
        lines.append(
            f"{r['model_id']} | {r['best_val_bpb']:.6f} | "
            f"{r['total_runs']} | {r['kept']} | {r['crashed']}"
        )
    return "\n".join(lines)


@tool
def get_experiment_history(model_id: str = "") -> str:
    """Get the full experiment history, optionally filtered by model_id."""
    rows = db_get_experiments(model_id=model_id or None, db_path=DB_FILE)
    if not rows:
        return "No experiments found."
    lines = []
    for r in rows:
        lines.append(
            f"#{r['id']} [{r['status']}] val_bpb={r['val_bpb']:.6f} "
            f"mem={r['memory_gb']:.1f}GB model={r['model_id']} — {r['description']}"
        )
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# Agent factory
# ---------------------------------------------------------------------------

RESEARCHER_TOOLS = [read_file, write_file, run_shell, log_experiment, compare_models, get_experiment_history]


def build_researcher_prompt(model_id: str) -> str:
    program = (PROJECT_DIR / "program.md").read_text(encoding="utf-8")
    return textwrap.dedent(f"""\
    You are an autonomous ML researcher running experiments on Apple Silicon
    using the autoresearch-mlx framework.

    You are currently using LM Studio model: **{model_id}**

    You have these tools:
    - read_file / write_file — read and edit project files (only train.py and results.tsv are writable)
    - run_shell — execute shell commands (git, uv run train.py, grep, etc.)
    - log_experiment — record experiment results to the persistent database
    - compare_models — see which model achieved the best results
    - get_experiment_history — view all past experiments

    WORKFLOW FOR EACH EXPERIMENT:
    1. Read results.tsv and train.py to understand the current state.
    2. Decide what to try next (depth, width, batch size, learning rates, architecture).
    3. Write the modified train.py.
    4. Commit: run_shell("git add train.py && git commit -m 'experiment: <description>'")
    5. Train: run_shell("uv run train.py > run.log 2>&1")
    6. Read results: run_shell("tail -5 run.log")
    7. Update results.tsv and call log_experiment to persist the result.
    8. If val_bpb improved: git add results.tsv && git commit --amend --no-edit
    9. If val_bpb did NOT improve: log as discard, then git reset --hard HEAD~1
    10. Report what you did and the result.

    RULES:
    - NEVER modify prepare.py. NEVER install new packages.
    - Try ONE change at a time. Lower val_bpb is better.
    - Always call log_experiment after each run.
    - You are running ONE experiment per invocation. Focus on doing it well.

    REFERENCE — full autoresearch specification:

    """) + program


def create_researcher(model_id: str, base_url: str = "http://localhost:1234/v1") -> Agent:
    return Agent(
        name="Researcher",
        model=LMStudio(id=model_id, base_url=base_url),
        tools=RESEARCHER_TOOLS,
        instructions=[build_researcher_prompt(model_id)],
        markdown=True,
        tool_call_limit=50,
    )


# ---------------------------------------------------------------------------
# Parse run.log for fallback DB logging
# ---------------------------------------------------------------------------

def _parse_run_log() -> tuple[float | None, float | None]:
    """Extract val_bpb and peak_vram_mb from the last run.log."""
    log_path = PROJECT_DIR / "run.log"
    if not log_path.exists():
        return None, None
    try:
        text = log_path.read_text(errors="replace")
    except Exception:
        return None, None
    val_bpb = None
    peak_mb = None
    for m in re.finditer(r"val_bpb[:\s=]+([0-9]+\.?[0-9]*)", text):
        val_bpb = float(m.group(1))
    for m in re.finditer(r"peak_vram_mb[:\s=]+([0-9]+\.?[0-9]*)", text):
        peak_mb = float(m.group(1))
    return val_bpb, peak_mb


# ---------------------------------------------------------------------------
# Single-experiment runner  (deterministic — no tool-calling needed)
# ---------------------------------------------------------------------------

def _strip_thinking(text: str) -> str:
    """Remove <think>...</think> blocks that reasoning models emit."""
    return re.sub(r"<think>[\s\S]*?</think>", "", text).strip()


# Hyperparameters the LLM is allowed to tweak
TUNABLE_PARAMS = {
    "DEPTH": (2, 12, int),
    "ASPECT_RATIO": (32, 128, int),
    "HEAD_DIM": (64, 256, int),
    "TOTAL_BATCH_SIZE": (2**14, 2**18, int),
    "DEVICE_BATCH_SIZE": (4, 64, int),
    "EMBEDDING_LR": (0.01, 2.0, float),
    "UNEMBEDDING_LR": (0.001, 0.1, float),
    "MATRIX_LR": (0.001, 0.2, float),
    "SCALAR_LR": (0.01, 2.0, float),
    "WEIGHT_DECAY": (0.0, 0.5, float),
    "WARMUP_RATIO": (0.0, 0.3, float),
    "WARMDOWN_RATIO": (0.0, 1.0, float),
}


def _extract_current_params(train_py: str) -> dict[str, str]:
    """Read the current value of each tunable param from train.py."""
    params = {}
    for name in TUNABLE_PARAMS:
        m = re.search(rf"^{name}\s*=\s*(.+)$", train_py, re.MULTILINE)
        if m:
            params[name] = m.group(1).strip()
    return params


_LLM_MAX_RETRIES = 3
_LLM_RETRY_DELAYS = [5, 10, 20]


def _parse_llm_suggestion(text: str) -> tuple[str, str] | None:
    """Try to extract 'PARAM = value' from LLM text. Returns None if unparseable."""
    cleaned = _strip_thinking(text).strip()
    for line in cleaned.splitlines():
        line = line.strip()
        m = re.match(r"(\w+)\s*=\s*(.+)", line)
        if m and m.group(1) in TUNABLE_PARAMS:
            return m.group(1), m.group(2).strip()
    return None


def _ask_llm_for_change(model_id: str, base_url: str,
                        train_py: str, results_tsv: str,
                        iteration: int, prev_result: str,
                        goal: str) -> tuple[str, str]:
    """Ask the LLM which hyperparameter to change. Returns (param_name, new_value).

    Retries with exponential backoff on connection errors (e.g. LM Studio
    'Channel Error' caused by memory pressure after training).
    Also reads reasoning_content for models like Qwen that put their
    answer there instead of in content.
    """
    from openai import OpenAI

    client = OpenAI(base_url=base_url, api_key="lm-studio")

    current = _extract_current_params(train_py)
    param_lines = "\n".join(f"  {k} = {v}  (range: {TUNABLE_PARAMS[k][0]}..{TUNABLE_PARAMS[k][1]})"
                            for k, v in current.items())

    prev_block = ""
    if prev_result:
        prev_block = f"\nPrevious result: {prev_result}\nTry something DIFFERENT.\n"

    prompt = textwrap.dedent(f"""\
        You are an ML researcher optimizing a GPT model. Goal: {goal}.

        Current hyperparameters:
        {param_lines}

        Recent results:
        {results_tsv[:1500]}
        {prev_block}
        EXPERIMENT #{iteration}: Pick exactly ONE parameter to change.
        Reply with exactly one line in this format:
        PARAM_NAME = new_value

        For example:
        DEPTH = 6

        Reply with ONLY that one line, nothing else.
    """)

    last_error = None
    for attempt in range(_LLM_MAX_RETRIES):
        if attempt > 0:
            delay = _LLM_RETRY_DELAYS[min(attempt - 1, len(_LLM_RETRY_DELAYS) - 1)]
            print(f"[llm] Retry {attempt}/{_LLM_MAX_RETRIES} in {delay}s...")
            time.sleep(delay)

        try:
            print(f"[llm] Asking {model_id} for hyperparameter change (attempt {attempt + 1})...")
            response = client.chat.completions.create(
                model=model_id,
                messages=[{"role": "user", "content": prompt}],
                temperature=0.8,
                max_tokens=200,
            )
        except Exception as e:
            last_error = e
            print(f"[llm] Connection error: {e}")
            continue

        msg = response.choices[0].message

        # 1) Try content field
        raw_content = msg.content or ""
        result = _parse_llm_suggestion(raw_content)
        if result:
            print(f"[llm] Parsed from content: {result[0]} = {result[1]}")
            return result

        # 2) Try reasoning_content (Qwen, DeepSeek put answers here)
        reasoning = getattr(msg, "reasoning_content", None) or ""
        if reasoning:
            result = _parse_llm_suggestion(reasoning)
            if result:
                print(f"[llm] Parsed from reasoning_content: {result[0]} = {result[1]}")
                return result

        print(f"[llm] Empty/unparseable response — content={raw_content!r} reasoning={reasoning[:200]!r}")

    # All retries exhausted — random fallback
    if last_error:
        print(f"[llm] All {_LLM_MAX_RETRIES} attempts failed, last error: {last_error}")
    param = random.choice(list(current.keys()))
    lo, hi, typ = TUNABLE_PARAMS[param]
    if typ == int:
        new_val = str(random.choice([lo, hi, (lo + hi) // 2]))
    else:
        new_val = str(round(random.uniform(lo, hi), 4))
    print(f"[llm] Random fallback: {param} = {new_val}")
    return param, new_val


def run_one_experiment(
    agent: Agent | None,
    model_id: str,
    iteration: int,
    prev_result: str = "",
    goal: str = "minimize val_bpb",
    base_url: str = "http://localhost:1234/v1",
    on_progress: "callable | None" = None,
    stop_check: "callable | None" = None,
) -> str:
    """Run a single experiment: ask LLM for a change, apply it, train, log results.

    This is fully deterministic — the LLM only suggests the code change,
    everything else (git, training, logging) is done programmatically.
    stop_check: if provided, called periodically — returns True to abort.
    """
    set_active_model(model_id)

    print(f"\n{'='*60}")
    print(f"[runner] Experiment #{iteration} — model={model_id}")
    print(f"{'='*60}")

    # 1. Read current state
    train_path = PROJECT_DIR / "train.py"
    results_path = PROJECT_DIR / "results.tsv"
    train_py = train_path.read_text(encoding="utf-8") if train_path.exists() else ""
    results_tsv = results_path.read_text(encoding="utf-8") if results_path.exists() else "No results yet"

    # 2. Decide which hyperparameter to change
    if model_id == "random-search":
        current = _extract_current_params(train_py)
        param_name = random.choice(list(current.keys()))
        lo, hi, typ = TUNABLE_PARAMS[param_name]
        if typ == int:
            new_value = str(random.randint(lo, hi))
        else:
            new_value = str(round(random.uniform(lo, hi), 4))
        print(f"[runner] Random search: {param_name} = {new_value}")
    else:
        try:
            param_name, new_value = _ask_llm_for_change(
                model_id, base_url, train_py, results_tsv,
                iteration, prev_result, goal,
            )
        except Exception as e:
            desc = f"experiment {iteration}: LLM call failed — {e}"
            print(f"[runner] LLM error: {e}")
            db_log_experiment(model_id=model_id, val_bpb=99.0, memory_gb=0.0,
                              status="crash", description=desc, db_path=DB_FILE)
            return desc

    # 3. Validate and clamp the value
    lo, hi, typ = TUNABLE_PARAMS.get(param_name, (None, None, str))
    try:
        parsed = typ(eval(new_value))  # handles expressions like 2**16
        if lo is not None and hi is not None:
            parsed = max(lo, min(hi, parsed))
        new_value = str(parsed)
    except Exception:
        pass

    # 4. Apply the change via regex replacement in train.py
    old_match = re.search(rf"^({param_name}\s*=\s*)(.+)$", train_py, re.MULTILINE)
    if not old_match:
        desc = f"experiment {iteration}: param {param_name} not found in train.py"
        print(f"[runner] {desc}")
        db_log_experiment(model_id=model_id, val_bpb=99.0, memory_gb=0.0,
                          status="crash", description=desc, db_path=DB_FILE)
        return desc

    old_value = old_match.group(2).strip()
    change_desc = f"{param_name}: {old_value} → {new_value}"
    print(f"[runner] Change: {change_desc}")

    new_train_py = train_py[:old_match.start(2)] + new_value + train_py[old_match.end(2):]
    train_path.write_text(new_train_py, encoding="utf-8")

    # 5. Git commit
    subprocess.run(
        f"git add train.py && git commit -m 'experiment {iteration}: {change_desc[:60]}'",
        shell=True, capture_output=True, cwd=PROJECT_DIR,
    )

    # 6. Train — stream output to run.log AND emit progress via callback
    #    train.py uses \r (carriage return) for progress lines, so we read
    #    in small chunks and split on both \r and \n.
    print(f"[runner] Training...")
    log_path = PROJECT_DIR / "run.log"
    stopped = False
    with open(log_path, "w", encoding="utf-8") as log_f:
        proc = subprocess.Popen(
            ["uv", "run", "train.py"],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            cwd=PROJECT_DIR,
        )
        buf = b""
        last_pct = ""
        check_counter = 0
        while True:
            chunk = proc.stdout.read(256)
            if not chunk:
                break
            buf += chunk
            while b"\n" in buf or b"\r" in buf:
                idx_n = buf.find(b"\n")
                idx_r = buf.find(b"\r")
                if idx_n == -1:
                    idx = idx_r
                elif idx_r == -1:
                    idx = idx_n
                else:
                    idx = min(idx_n, idx_r)
                segment = buf[:idx]
                buf = buf[idx + 1:]
                line = segment.decode("utf-8", errors="replace").strip()
                if not line:
                    continue
                log_f.write(line + "\n")
                log_f.flush()
                pct_m = re.search(r"\((\d+\.\d)%\)", line)
                if pct_m:
                    pct = pct_m.group(1)
                    if pct != last_pct:
                        last_pct = pct
                        if on_progress:
                            on_progress(f"  [{model_id}] training {pct}%")
                elif on_progress and ("val_bpb" in line or "FAIL" in line or "compiled" in line):
                    on_progress(f"  [{model_id}] {line[:120]}")
            check_counter += 1
            if stop_check and check_counter % 10 == 0 and stop_check():
                print(f"[runner] Stop requested — killing training subprocess")
                proc.terminate()
                proc.wait(timeout=10)
                stopped = True
                break
        if buf.strip():
            remaining = buf.decode("utf-8", errors="replace").strip()
            log_f.write(remaining + "\n")
            if on_progress and ("val_bpb" in remaining or "FAIL" in remaining):
                on_progress(f"  [{model_id}] {remaining[:120]}")
        if not stopped:
            proc.wait(timeout=SHELL_TIMEOUT)
    print(f"[runner] Training exit code: {proc.returncode}")

    # Let LM Studio reclaim memory after training finishes
    if not stopped:
        print("[runner] Waiting 5s for LM Studio to recover memory...")
        time.sleep(5)

    # 7. Parse results from run.log
    if stopped:
        val_bpb = 99.0
        peak_mb = None
        mem_gb = 0.0
        status = "crash"
        desc = f"experiment {iteration}: {change_desc[:80]} → stopped by user"
        print(f"[runner] Experiment stopped by user")
        subprocess.run("git reset --hard HEAD~1", shell=True,
                        capture_output=True, cwd=PROJECT_DIR)
    else:
        val_bpb, peak_mb = _parse_run_log()
        mem_gb = round(peak_mb / 1024, 2) if peak_mb else 0.0

        if val_bpb is not None:
            status = "keep"
            desc = f"experiment {iteration}: {change_desc[:80]} → val_bpb={val_bpb:.4f}"
            print(f"[runner] Result: val_bpb={val_bpb:.4f} mem={mem_gb}GB")
        else:
            status = "crash"
            val_bpb = 99.0
            desc = f"experiment {iteration}: {change_desc[:80]} → training failed"
            print(f"[runner] Training failed — no val_bpb found in run.log")
            subprocess.run("git reset --hard HEAD~1", shell=True,
                            capture_output=True, cwd=PROJECT_DIR)

    # 8. Log to DB with full config snapshot
    updated_params = _extract_current_params(
        train_path.read_text(encoding="utf-8") if train_path.exists() else ""
    )
    config_snapshot = json.dumps(updated_params)
    row_id = db_log_experiment(
        model_id=model_id, val_bpb=val_bpb, memory_gb=mem_gb,
        status=status, description=desc, db_path=DB_FILE,
        config_json=config_snapshot,
    )
    print(f"[runner] Logged #{row_id}: {status} val_bpb={val_bpb}")

    # 9. Save weights snapshot for successful experiments
    if status == "keep":
        src_weights = PROJECT_DIR / "music_model.safetensors"
        if src_weights.exists():
            weights_dir = PROJECT_DIR / "weights"
            weights_dir.mkdir(exist_ok=True)
            dest = weights_dir / f"exp_{row_id}.safetensors"
            shutil.copy2(str(src_weights), str(dest))
            rel_path = f"weights/exp_{row_id}.safetensors"
            db_update_weights_path(row_id, rel_path, db_path=DB_FILE)
            print(f"[runner] Saved weights: {rel_path}")

    # 10. Update results.tsv
    tsv_line = f"{iteration}\t{val_bpb}\t{peak_mb or 0}\t{status}\t{change_desc[:80]}\n"
    with open(results_path, "a", encoding="utf-8") as f:
        f.write(tsv_line)

    if status == "keep":
        subprocess.run("git add results.tsv && git commit --amend --no-edit",
                        shell=True, capture_output=True, cwd=PROJECT_DIR)

    return desc


# ---------------------------------------------------------------------------
# Music sampler (subprocess)
# ---------------------------------------------------------------------------

def sample_music_cli(
    prompt: str = "",
    max_tokens: int = 200,
    temperature: float = 0.8,
    weights_path: str = "",
) -> str:
    cmd = ["uv", "run", "python", "sample_music.py",
           "--max-new-tokens", str(max_tokens),
           "--temperature", str(temperature)]
    if weights_path:
        cmd.extend(["--weights", weights_path])
    if prompt:
        cmd.extend(["--prompt", prompt])
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60, cwd=PROJECT_DIR)
        if result.returncode != 0:
            return f"ERROR: {result.stderr}"
        return result.stdout
    except Exception as e:
        return f"ERROR: {e}"
