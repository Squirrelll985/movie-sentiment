# -*- coding: utf-8 -*-
"""Evaluate a saved checkpoint on the held-out IMDb test split."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentiment.config import Config, save_json  # noqa: E402
from sentiment.data import prepare_data  # noqa: E402
from sentiment.metrics import compute_metrics  # noqa: E402
from sentiment.model import load_model  # noqa: E402
from sentiment.utils import format_metrics  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", default="outputs/main/best_model.pt")
    parser.add_argument("--split", default="test", choices=["val", "test"])
    parser.add_argument("--output", default="outputs/main/eval_report.json")
    args = parser.parse_args()

    config = Config.load().resolve()
    data = prepare_data(config)
    inputs = data.test_x if args.split == "test" else data.val_x
    labels = data.test_y if args.split == "test" else data.val_y
    checkpoint_path = Path(args.checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = ROOT / checkpoint_path

    model, checkpoint = load_model(checkpoint_path)
    max_len = int(checkpoint.get("data_config", {}).get("max_len", inputs.shape[1]))
    inputs = inputs[:, :max_len]

    probs = []
    batch_size = 256
    with torch.no_grad():
        for start in range(0, len(inputs), batch_size):
            batch = torch.from_numpy(inputs[start : start + batch_size].astype(np.int64))
            lengths = (batch != 0).sum(dim=1).clamp(min=1)
            logits = model(batch, lengths)["logits"]
            probs.append(torch.softmax(logits, dim=-1)[:, 1].numpy())
    y_prob = np.concatenate(probs)
    metrics = compute_metrics(labels, y_prob)
    print(format_metrics(metrics))
    report = {
        "checkpoint": str(checkpoint_path),
        "split": args.split,
        "model": checkpoint["model_name"],
        "metrics": metrics,
    }
    output = Path(args.output)
    if not output.is_absolute():
        output = ROOT / output
    save_json(report, output)
    np.savez_compressed(output.with_suffix(".npz"), y_true=labels, y_prob=y_prob)
    print(f"saved {output}")


if __name__ == "__main__":
    main()
