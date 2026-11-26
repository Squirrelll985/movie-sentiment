# -*- coding: utf-8 -*-
"""Inference and token-level explanation for trained sentiment models."""
from __future__ import annotations

import time
from pathlib import Path
from typing import Any

import numpy as np
import torch

from .data import PAD_ID, encode_text, tokenize
from .model import load_model


class SentimentPredictor:
    def __init__(self, checkpoint_path: str | Path, device: str = "cpu") -> None:
        self.checkpoint_path = Path(checkpoint_path)
        if not self.checkpoint_path.exists():
            raise FileNotFoundError(
                f"checkpoint not found: {self.checkpoint_path}. Train the model first."
            )
        self.device = torch.device(device)
        self.model, self.checkpoint = load_model(self.checkpoint_path, self.device)
        self.vocab: dict[str, int] = self.checkpoint["vocab"]
        self.max_len = int(self.checkpoint.get("data_config", {}).get("max_len", 256))
        self.label_names = self.checkpoint.get("label_names", ["negative", "positive"])

    @property
    def model_name(self) -> str:
        return self.checkpoint.get("model_name", "lstm")

    @property
    def parameter_count(self) -> int:
        return int(sum(p.numel() for p in self.model.parameters() if p.requires_grad))

    @property
    def metrics(self) -> dict[str, Any]:
        return self.checkpoint.get("metrics", {})

    def _prepare(self, text: str) -> tuple[list[str], np.ndarray]:
        tokens = tokenize(text)
        if not tokens:
            raise ValueError("请输入包含有效英文单词的影评文本。")
        ids = encode_text(tokens, self.vocab, self.max_len)
        return tokens, np.asarray([ids], dtype=np.int64)

    @torch.no_grad()
    def _forward_logits(self, batch_ids: np.ndarray) -> np.ndarray:
        input_ids = torch.from_numpy(batch_ids.astype(np.int64)).to(self.device)
        lengths = (input_ids != PAD_ID).sum(dim=1).clamp(min=1)
        return self.model(input_ids, lengths)["logits"].cpu().numpy()

    def _forward_probs(self, batch_ids: np.ndarray) -> np.ndarray:
        logits = self._forward_logits(batch_ids)
        logits = logits - logits.max(axis=1, keepdims=True)
        exp = np.exp(logits)
        return (exp / exp.sum(axis=1, keepdims=True))[:, 1]

    def predict(self, text: str, explain: bool = False, top_k: int = 6) -> dict[str, Any]:
        started = time.perf_counter()
        tokens, ids = self._prepare(text)
        logits = self._forward_logits(ids)[0]
        logits = logits - logits.max()
        exp = np.exp(logits)
        probs = exp / exp.sum()
        prob_positive = float(probs[1])
        prob_negative = float(probs[0])
        margin = float(logits[1] - logits[0])
        label = "positive" if prob_positive >= 0.5 else "negative"
        result: dict[str, Any] = {
            "label": label,
            "label_cn": "正面" if label == "positive" else "负面",
            "confidence": max(prob_positive, prob_negative),
            "prob_positive": prob_positive,
            "prob_negative": prob_negative,
            "token_count": len(tokens),
            "truncated": len(tokens) > self.max_len,
            "model": self.model_name,
            "device": str(self.device),
        }
        if explain:
            result["explanation"] = self.explain(tokens, margin, top_k=top_k)
        result["latency_ms"] = round((time.perf_counter() - started) * 1000, 2)
        return result

    @torch.no_grad()
    def explain(self, tokens: list[str], base_margin: float, top_k: int = 6) -> dict[str, Any]:
        """Leave-one-token-out occlusion on the positive-minus-negative margin.

        A positive contribution means removing the token lowers the margin in
        favour of the positive class, i.e. the token supports a positive
        prediction.  The logit margin is used instead of the saturated
        probability so that evidence remains visible for confident reviews.
        """
        tokens = tokens[: self.max_len]
        if len(tokens) <= 1:
            return {"positive": [], "negative": [], "method": "leave-one-out"}
        batch = []
        for i in range(len(tokens)):
            ablated = tokens[:i] + tokens[i + 1 :]
            batch.append(encode_text(ablated, self.vocab, self.max_len))
        ablated_logits = self._forward_logits(np.asarray(batch, dtype=np.int64))
        ablated_margins = ablated_logits[:, 1] - ablated_logits[:, 0]
        contributions = base_margin - ablated_margins

        per_position = [
            {"token": token, "score": float(score), "position": i}
            for i, (token, score) in enumerate(zip(tokens, contributions))
        ]
        best: dict[str, dict] = {}
        for item in per_position:
            token = item["token"]
            if token not in best or abs(item["score"]) > abs(best[token]["score"]):
                best[token] = item
        unique = list(best.values())
        max_abs = max((abs(item["score"]) for item in unique), default=1.0) or 1.0
        for item in unique:
            item["relative"] = item["score"] / max_abs
        positive = sorted([x for x in unique if x["score"] > 0], key=lambda x: -x["score"])[:top_k]
        negative = sorted([x for x in unique if x["score"] < 0], key=lambda x: x["score"])[:top_k]
        return {
            "positive": positive,
            "negative": negative,
            "method": "leave-one-out",
            "score_type": "logit_margin",
            "evaluated_tokens": len(tokens),
        }

    def attention(self, text: str, top_k: int = 12) -> list[dict[str, Any]]:
        tokens, _ = self._prepare(text)
        if self.model_name != "bilstm_attention":
            return []
        _, ids = self._prepare(text)
        input_ids = torch.from_numpy(ids.astype(np.int64)).to(self.device)
        lengths = (input_ids != PAD_ID).sum(dim=1).clamp(min=1)
        with torch.no_grad():
            weights = self.model(input_ids, lengths)["attention"][0].cpu().numpy()
        pairs = [
            {"token": token, "weight": float(weights[i])}
            for i, token in enumerate(tokens[: self.max_len])
        ]
        return sorted(pairs, key=lambda x: -x["weight"])[:top_k]
