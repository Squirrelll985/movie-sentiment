# -*- coding: utf-8 -*-
"""Data acquisition, tokenisation, vocabulary construction and splits.

The official IMDb archive is mirrored on Hugging Face as two parquet files.
They contain exactly the same 25,000 training and 25,000 test reviews as the
original ``aclImdb`` release, which keeps the pipeline reproducible on machines
where the Stanford host is unreachable.
"""
from __future__ import annotations

import html
import json
import re
import urllib.request
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np

from .config import Config

PAD_TOKEN = "<pad>"
UNK_TOKEN = "<unk>"
PAD_ID = 0
UNK_ID = 1

_TOKEN_RE = re.compile(r"[a-z]+(?:n't|'[a-z]+)?|[0-9]+|[!?]")


def clean_text(text: str) -> str:
    """Normalise an IMDb review without destroying sentiment cues."""
    text = html.unescape(text)
    text = re.sub(r"<br\s*/?>", " ", text, flags=re.IGNORECASE)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("’", "'").replace("`", "'")
    text = re.sub(r"\s+", " ", text)
    return text.strip().lower()


def tokenize(text: str) -> list[str]:
    """Word-level tokeniser that preserves negation and exclamation marks."""
    return _TOKEN_RE.findall(clean_text(text))


def _download(url: str, destination: Path) -> Path:
    if destination.exists() and destination.stat().st_size > 1_000_000:
        print(f"[data] file already exists: {destination.name}")
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    print(f"[data] downloading {destination.name}")
    with urllib.request.urlopen(request, timeout=120) as response, tmp.open("wb") as fh:
        total = int(response.headers.get("Content-Length", 0))
        downloaded = 0
        while True:
            chunk = response.read(1024 * 1024)
            if not chunk:
                break
            fh.write(chunk)
            downloaded += len(chunk)
            if total:
                pct = downloaded * 100 / total
                print(f"\r[data] {pct:5.1f}% ({downloaded/1024/1024:.1f} MB)", end="")
    print()
    tmp.replace(destination)
    return destination


def ensure_dataset(config: Config) -> tuple[Path, Path]:
    """Return local paths to the train/test parquet files."""
    raw = Path(config.paths.raw_dir) / "imdb_hf"
    train_path = raw / "train.parquet"
    test_path = raw / "test.parquet"
    _download(config.data.train_url, train_path)
    _download(config.data.test_url, test_path)
    return train_path, test_path


def _read_parquet(path: Path) -> tuple[list[str], list[int]]:
    import pyarrow.parquet as pq

    table = pq.read_table(path, columns=["text", "label"])
    texts = [clean_text(t) for t in table.column("text").to_pylist()]
    labels = [int(v) for v in table.column("label").to_pylist()]
    return texts, labels


def _stratified_split(
    texts: list[str], labels: list[int], val_ratio: float, seed: int
) -> tuple[list[str], list[int], list[str], list[int]]:
    rng = np.random.default_rng(seed)
    train_idx: list[int] = []
    val_idx: list[int] = []
    for cls in (0, 1):
        indices = np.flatnonzero(np.asarray(labels) == cls)
        rng.shuffle(indices)
        n_val = int(round(len(indices) * val_ratio))
        val_idx.extend(indices[:n_val].tolist())
        train_idx.extend(indices[n_val:].tolist())
    rng.shuffle(train_idx)
    rng.shuffle(val_idx)
    return (
        [texts[i] for i in train_idx],
        [labels[i] for i in train_idx],
        [texts[i] for i in val_idx],
        [labels[i] for i in val_idx],
    )


def build_vocabulary(
    tokenized_texts: Iterable[list[str]], max_vocab: int, min_freq: int
) -> dict[str, int]:
    counter: Counter[str] = Counter()
    for tokens in tokenized_texts:
        counter.update(tokens)
    vocab = {PAD_TOKEN: PAD_ID, UNK_TOKEN: UNK_ID}
    for token, freq in sorted(counter.items(), key=lambda kv: (-kv[1], kv[0])):
        if freq < min_freq or len(vocab) >= max_vocab:
            break
        vocab[token] = len(vocab)
    return vocab


def encode_text(tokens: list[str], vocab: dict[str, int], max_len: int) -> list[int]:
    return [vocab.get(token, UNK_ID) for token in tokens[:max_len]]


def pad_sequences(sequences: list[list[int]], max_len: int) -> np.ndarray:
    out = np.zeros((len(sequences), max_len), dtype=np.int32)
    for i, seq in enumerate(sequences):
        seq = seq[:max_len]
        out[i, : len(seq)] = seq
    return out


@dataclass
class DataBundle:
    train_x: np.ndarray
    train_y: np.ndarray
    val_x: np.ndarray
    val_y: np.ndarray
    test_x: np.ndarray
    test_y: np.ndarray
    vocab: dict[str, int]
    stats: dict

    @property
    def vocab_size(self) -> int:
        return len(self.vocab)


def prepare_data(config: Config, force: bool = False) -> DataBundle:
    paths = config.paths
    processed = Path(paths.processed_dir)
    arrays_path = processed / "imdb_processed.npz"
    vocab_path = processed / "vocab.json"
    stats_path = processed / "stats.json"
    if not force and arrays_path.exists() and vocab_path.exists() and stats_path.exists():
        data = np.load(arrays_path)
        vocab = json.loads(vocab_path.read_text(encoding="utf-8"))
        stats = json.loads(stats_path.read_text(encoding="utf-8"))
        print(f"[data] loaded cached arrays: {arrays_path}")
        return DataBundle(
            train_x=data["train_x"],
            train_y=data["train_y"],
            val_x=data["val_x"],
            val_y=data["val_y"],
            test_x=data["test_x"],
            test_y=data["test_y"],
            vocab=vocab,
            stats=stats,
        )

    train_path, test_path = ensure_dataset(config)
    print("[data] reading labelled train/test reviews")
    train_texts, train_labels = _read_parquet(train_path)
    test_texts, test_labels = _read_parquet(test_path)
    train_texts, train_labels, val_texts, val_labels = _stratified_split(
        train_texts, train_labels, config.data.val_ratio, config.train.seed
    )

    print("[data] tokenising and building vocabulary")
    train_tokens = [tokenize(t) for t in train_texts]
    val_tokens = [tokenize(t) for t in val_texts]
    test_tokens = [tokenize(t) for t in test_texts]
    vocab = build_vocabulary(train_tokens, config.data.max_vocab, config.data.min_freq)

    train_x = pad_sequences(
        [encode_text(t, vocab, config.data.max_len) for t in train_tokens], config.data.max_len
    )
    val_x = pad_sequences(
        [encode_text(t, vocab, config.data.max_len) for t in val_tokens], config.data.max_len
    )
    test_x = pad_sequences(
        [encode_text(t, vocab, config.data.max_len) for t in test_tokens], config.data.max_len
    )
    train_y = np.asarray(train_labels, dtype=np.int64)
    val_y = np.asarray(val_labels, dtype=np.int64)
    test_y = np.asarray(test_labels, dtype=np.int64)

    lengths = np.asarray([len(t) for t in train_tokens], dtype=np.int32)
    stats = {
        "dataset": "IMDb Large Movie Review Dataset",
        "train_size": int(len(train_y)),
        "val_size": int(len(val_y)),
        "test_size": int(len(test_y)),
        "vocab_size": len(vocab),
        "max_len": config.data.max_len,
        "min_freq": config.data.min_freq,
        "val_ratio": config.data.val_ratio,
        "positive_ratio_train": float(train_y.mean()),
        "class_counts_train": {
            str(k): int(v) for k, v in zip(*np.unique(train_y, return_counts=True))
        },
        "review_length": {
            "mean": float(lengths.mean()),
            "median": float(np.median(lengths)),
            "p90": float(np.percentile(lengths, 90)),
            "p95": float(np.percentile(lengths, 95)),
            "p99": float(np.percentile(lengths, 99)),
            "max": int(lengths.max()),
        },
        "truncated_train_ratio": float((lengths > config.data.max_len).mean()),
    }

    processed.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        arrays_path,
        train_x=train_x,
        train_y=train_y,
        val_x=val_x,
        val_y=val_y,
        test_x=test_x,
        test_y=test_y,
    )
    vocab_path.write_text(json.dumps(vocab, ensure_ascii=False), encoding="utf-8")
    stats_path.write_text(json.dumps(stats, ensure_ascii=False, indent=2), encoding="utf-8")
    print(
        f"[data] saved {arrays_path} | vocab={len(vocab)} | "
        f"train={len(train_y)} val={len(val_y)} test={len(test_y)}"
    )
    return DataBundle(train_x, train_y, val_x, val_y, test_x, test_y, vocab, stats)
