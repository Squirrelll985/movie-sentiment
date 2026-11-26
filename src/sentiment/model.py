# -*- coding: utf-8 -*-
"""LSTM and TextCNN classifiers implemented with PyTorch."""
from __future__ import annotations

from dataclasses import asdict
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.nn import functional as F

from .config import ModelConfig


class AttentionPooling(nn.Module):
    """Additive attention over the non-padding time steps."""

    def __init__(self, input_dim: int) -> None:
        super().__init__()
        self.projection = nn.Linear(input_dim, 1, bias=True)

    def forward(self, sequence: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        # sequence: [batch, time, dim], mask: [batch, time] with 1 for tokens
        scores = self.projection(sequence).squeeze(-1)
        scores = scores.masked_fill(mask == 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        return torch.bmm(weights.unsqueeze(1), sequence).squeeze(1), weights


class SentimentLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        hidden_dim: int = 128,
        num_layers: int = 1,
        bidirectional: bool = True,
        pooling: str = "attention",
        dropout: float = 0.5,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        if pooling not in {"attention", "mean", "max"}:
            raise ValueError(f"unsupported pooling mode: {pooling}")
        self.bidirectional = bidirectional
        self.pooling = pooling
        self.hidden_dim = hidden_dim
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.embedding_dropout = nn.Dropout(dropout)
        self.lstm = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        out_dim = hidden_dim * (2 if bidirectional else 1)
        self.attention = AttentionPooling(out_dim) if pooling == "attention" else None
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Sequential(
            nn.Linear(out_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 2),
        )
        self._init_weights()

    def _init_weights(self) -> None:
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        with torch.no_grad():
            self.embedding.weight[0].fill_(0)
        for name, param in self.lstm.named_parameters():
            if "weight_ih" in name:
                nn.init.xavier_uniform_(param)
            elif "weight_hh" in name:
                nn.init.orthogonal_(param)
            elif "bias" in name:
                nn.init.zeros_(param)

    def forward(
        self, input_ids: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        mask = (input_ids != 0).float()
        embedded = self.embedding_dropout(self.embedding(input_ids))
        if lengths is not None:
            lengths_cpu = lengths.detach().to("cpu").clamp(min=1)
            packed = nn.utils.rnn.pack_padded_sequence(
                embedded, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            sequence, _ = nn.utils.rnn.pad_packed_sequence(
                packed_out, batch_first=True, total_length=input_ids.size(1)
            )
        else:
            sequence, _ = self.lstm(embedded)

        if self.pooling == "attention":
            assert self.attention is not None
            pooled, attention = self.attention(sequence, mask)
        elif self.pooling == "mean":
            denom = mask.sum(dim=1, keepdim=True).clamp(min=1.0)
            pooled = (sequence * mask.unsqueeze(-1)).sum(dim=1) / denom
            attention = mask / denom
        else:
            masked = sequence.masked_fill(mask.unsqueeze(-1) == 0, -1e4)
            pooled = masked.max(dim=1).values
            attention = torch.zeros_like(mask)

        max_pooled = sequence.masked_fill(mask.unsqueeze(-1) == 0, -1e4).max(dim=1).values
        features = torch.cat([pooled, max_pooled], dim=-1)
        logits = self.classifier(self.dropout(features))
        return {"logits": logits, "attention": attention, "features": features}


class TextCNN(nn.Module):
    """A compact Kim-style CNN used as a reproducible baseline."""

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int = 128,
        num_filters: int = 100,
        kernel_sizes: list[int] | None = None,
        dropout: float = 0.5,
        padding_idx: int = 0,
    ) -> None:
        super().__init__()
        kernel_sizes = kernel_sizes or [3, 4, 5]
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.convs = nn.ModuleList(
            nn.Conv1d(embedding_dim, num_filters, kernel_size=k) for k in kernel_sizes
        )
        self.dropout = nn.Dropout(dropout)
        self.classifier = nn.Linear(num_filters * len(kernel_sizes), 2)
        nn.init.uniform_(self.embedding.weight, -0.1, 0.1)
        with torch.no_grad():
            self.embedding.weight[0].fill_(0)

    def forward(
        self, input_ids: torch.Tensor, lengths: torch.Tensor | None = None
    ) -> dict[str, torch.Tensor]:
        embedded = self.embedding(input_ids).transpose(1, 2)
        pooled = [F.relu(conv(embedded)).max(dim=2).values for conv in self.convs]
        features = self.dropout(torch.cat(pooled, dim=1))
        logits = self.classifier(features)
        attention = torch.zeros(
            (input_ids.size(0), input_ids.size(1)), device=input_ids.device
        )
        return {"logits": logits, "attention": attention, "features": features}


def build_model(name: str, vocab_size: int, config: ModelConfig) -> nn.Module:
    if name in {"lstm", "bilstm", "bilstm_attention"}:
        bidirectional = name in {"bilstm", "bilstm_attention"}
        pooling = "attention" if name == "bilstm_attention" else "mean"
        return SentimentLSTM(
            vocab_size=vocab_size,
            embedding_dim=config.embedding_dim,
            hidden_dim=config.hidden_dim,
            num_layers=config.num_layers,
            bidirectional=bidirectional,
            pooling=pooling,
            dropout=config.dropout,
        )
    if name == "textcnn":
        return TextCNN(
            vocab_size=vocab_size,
            embedding_dim=config.embedding_dim,
            num_filters=config.num_filters,
            kernel_sizes=config.kernel_sizes,
            dropout=config.dropout,
        )
    raise ValueError(f"unknown model: {name}")


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def save_model_checkpoint(
    model: nn.Module,
    path: str | Path,
    model_name: str,
    model_config: ModelConfig,
    vocab: dict[str, int],
    metrics: dict[str, Any],
    extra: dict[str, Any] | None = None,
) -> None:
    state = {
        "model_name": model_name,
        "model_config": asdict(model_config),
        "model_state": model.state_dict(),
        "vocab": vocab,
        "metrics": metrics,
        "label_names": ["negative", "positive"],
    }
    if extra:
        state.update(extra)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    torch.save(state, path)


def load_model(checkpoint_path: str | Path, device: str | torch.device = "cpu"):
    checkpoint = torch.load(Path(checkpoint_path), map_location=device, weights_only=False)
    model_config = ModelConfig(**checkpoint["model_config"])
    vocab = checkpoint["vocab"]
    model = build_model(checkpoint["model_name"], len(vocab), model_config)
    model.load_state_dict(checkpoint["model_state"])
    model.to(device)
    model.eval()
    return model, checkpoint
