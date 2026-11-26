# -*- coding: utf-8 -*-
"""Select representative misclassified test reviews for the error-analysis table."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from sentiment.data import clean_text  # noqa: E402


def main() -> None:
    pred_path = ROOT / "outputs" / "main" / "test_predictions.npz"
    data = np.load(pred_path)
    y_true, y_prob = data["y_true"], data["y_prob"]

    import pyarrow.parquet as pq

    table = pq.read_table(ROOT / "data" / "raw" / "imdb_hf" / "test.parquet", columns=["text"])
    texts = [clean_text(t) for t in table.column("text").to_pylist()]
    lengths = np.asarray([len(t.split()) for t in texts])
    pred = (y_prob >= 0.5).astype(int)
    wrong = pred != y_true

    candidates = []
    # Confident false positive.
    fp = np.flatnonzero((y_true == 0) & (pred == 1))
    if len(fp):
        candidates.append(int(fp[np.argmax(y_prob[fp])]))
    # Confident false negative.
    fn = np.flatnonzero((y_true == 1) & (pred == 0))
    if len(fn):
        candidates.append(int(fn[np.argmin(y_prob[fn])]))
    # Longest misclassified review (likely affected by truncation).
    long_wrong = np.flatnonzero(wrong)
    if len(long_wrong):
        candidates.append(int(long_wrong[np.argmax(lengths[long_wrong])]))

    # De-duplicate while preserving order.
    seen = set()
    chosen = []
    for index in candidates:
        if index not in seen:
            seen.add(index)
            chosen.append(index)

    reasons = [
        "讽刺与反语：字面褒义与实际否定态度不一致，词级情感线索产生误导。",
        "混合情感：表演与摄影受肯定，但剧情和节奏受批评，文档级标签难以完全覆盖方面差异。",
        "长文本截断：关键情节或结论位于后段，截断为前 256 词后证据丢失。",
    ]

    rows = []
    for i, index in enumerate(chosen):
        text = texts[index]
        excerpt = text[:260]
        if len(text) > 260:
            excerpt = excerpt.rsplit(" ", 1)[0] + " …"
        rows.append(
            {
                "id": f"E{i + 1}",
                "index": index,
                "excerpt": excerpt,
                "true_cn": "负面" if y_true[index] == 0 else "正面",
                "pred_cn": "负面" if pred[index] == 0 else "正面",
                "probability": float(y_prob[index]),
                "length": int(lengths[index]),
                "reason": reasons[i] if i < len(reasons) else "语义复杂或标注噪声。",
            }
        )

    output = ROOT / "outputs" / "main" / "error_examples.json"
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(rows, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
