"""
Download ~261K ABC notation melodies from the MelodyHub dataset (HuggingFace)
and write them as parquet shards for the autoresearch pipeline.

MelodyHub (MIT license) is a curated collection of public-domain folk melodies
in ABC notation. We use the "generation" task subset which contains one
complete ABC score per row.

Source: https://huggingface.co/datasets/sander-wood/melodyhub

Usage:
    python make_music_dataset.py            # download + shard
    python make_music_dataset.py --shards 8 # use more training shards

Data is cached in ~/.cache/autoresearch/.
"""

import argparse
import random
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import requests

CACHE_DIR = Path.home() / ".cache" / "autoresearch"
DATA_DIR = CACHE_DIR / "data"
RAW_DIR = CACHE_DIR / "melodyhub_raw"

PARQUET_URLS = {
    "train_0": "https://huggingface.co/api/datasets/sander-wood/melodyhub/parquet/default/train/0.parquet",
    "train_1": "https://huggingface.co/api/datasets/sander-wood/melodyhub/parquet/default/train/1.parquet",
    "val_0": "https://huggingface.co/api/datasets/sander-wood/melodyhub/parquet/default/validation/0.parquet",
}

VAL_SHARD_ID = 6542
VAL_RATIO = 0.05
SEED = 42


def download_parquets() -> list[Path]:
    """Download MelodyHub parquet files and cache locally."""
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    paths = []
    for name, url in PARQUET_URLS.items():
        dest = RAW_DIR / f"{name}.parquet"
        if dest.exists():
            print(f"  Cached: {dest.name}")
        else:
            print(f"  Downloading {name}...")
            resp = requests.get(url, timeout=300)
            resp.raise_for_status()
            dest.write_bytes(resp.content)
            print(f"  Saved {len(resp.content) / 1e6:.1f} MB -> {dest.name}")
        paths.append(dest)
    return paths


CONTROL_CODE_PREFIXES = ("S:", "B:", "E:", "X:")


def _clean_abc(raw: str) -> str:
    """Strip MelodyHub control codes (S:/B:/E:/X:), keep standard ABC."""
    lines = []
    for line in raw.strip().splitlines():
        stripped = line.strip()
        if stripped and not any(stripped.startswith(p) for p in CONTROL_CODE_PREFIXES):
            lines.append(line)
    return "\n".join(lines).strip()


def extract_melodies(parquet_paths: list[Path]) -> list[str]:
    """Read parquet files and extract ABC scores from 'generation' task rows."""
    docs = []
    seen = set()
    for path in parquet_paths:
        table = pq.read_table(path, columns=["task", "output"])
        task_col = table.column("task").to_pylist()
        output_col = table.column("output").to_pylist()
        for task, output in zip(task_col, output_col):
            if task == "generation" and output:
                cleaned = _clean_abc(output)
                if cleaned and cleaned not in seen:
                    seen.add(cleaned)
                    docs.append(cleaned)
    return docs


def write_shards(docs: list[str], num_train_shards: int) -> None:
    """Shuffle, split, and write parquet shards."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    rng = random.Random(SEED)
    rng.shuffle(docs)

    split_idx = max(1, int(len(docs) * (1 - VAL_RATIO)))
    train_docs = docs[:split_idx]
    val_docs = docs[split_idx:]

    print(f"Total melodies:   {len(docs):,}")
    print(f"Training docs:    {len(train_docs):,}")
    print(f"Validation docs:  {len(val_docs):,}")

    shard_size = (len(train_docs) + num_train_shards - 1) // num_train_shards
    for shard_idx in range(num_train_shards):
        start = shard_idx * shard_size
        end = min(start + shard_size, len(train_docs))
        shard_docs = train_docs[start:end]
        path = DATA_DIR / f"shard_{shard_idx:05d}.parquet"
        table = pa.table({"text": shard_docs})
        pq.write_table(table, path)
        print(f"  Wrote {path.name}  ({len(shard_docs):,} docs)")

    val_path = DATA_DIR / f"shard_{VAL_SHARD_ID:05d}.parquet"
    table = pa.table({"text": val_docs})
    pq.write_table(table, val_path)
    print(f"  Wrote {val_path.name}  ({len(val_docs):,} docs)")


def main():
    parser = argparse.ArgumentParser(
        description="Build music dataset from MelodyHub for autoresearch"
    )
    parser.add_argument(
        "--shards", type=int, default=4, help="Number of training shards"
    )
    args = parser.parse_args()

    print("Downloading MelodyHub parquets...")
    parquet_paths = download_parquets()

    print("Extracting generation melodies...")
    docs = extract_melodies(parquet_paths)
    if not docs:
        raise RuntimeError("No melodies extracted from MelodyHub")

    print(f"\nFound {len(docs):,} melodies. Writing shards...")
    write_shards(docs, num_train_shards=args.shards)
    print("\nDone! Run `python prepare.py --num-shards 4` next to train the tokenizer.")


if __name__ == "__main__":
    main()
