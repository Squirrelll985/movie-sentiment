# -*- coding: utf-8 -*-
"""Project-wide configuration for the LSTM sentiment analysis system."""
from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = PROJECT_ROOT / "configs" / "default.json"


@dataclass
class PathConfig:
    data_dir: str = "data"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    output_dir: str = "outputs"
    checkpoint_dir: str = "outputs/checkpoints"
    figure_dir: str = "outputs/figures"
    log_dir: str = "outputs/logs"


@dataclass
class DataConfig:
    dataset_url: str = "https://huggingface.co/datasets/stanfordnlp/imdb"
    train_url: str = (
        "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text/"
        "train-00000-of-00001.parquet"
    )
    test_url: str = (
        "https://huggingface.co/datasets/stanfordnlp/imdb/resolve/main/plain_text/"
        "test-00000-of-00001.parquet"
    )
    max_vocab: int = 20000
    min_freq: int = 3
    max_len: int = 256
    val_ratio: float = 0.1
    num_classes: int = 2
    num_workers: int = 0


@dataclass
class ModelConfig:
    name: str = "bilstm_attention"
    embedding_dim: int = 128
    hidden_dim: int = 128
    num_layers: int = 1
    bidirectional: bool = True
    pooling: str = "attention"
    dropout: float = 0.5
    kernel_sizes: list[int] = field(default_factory=lambda: [3, 4, 5])
    num_filters: int = 100


@dataclass
class TrainConfig:
    batch_size: int = 64
    epochs: int = 8
    learning_rate: float = 1e-3
    weight_decay: float = 1e-4
    grad_clip: float = 1.0
    patience: int = 3
    seed: int = 42
    device: str = "auto"


@dataclass
class Config:
    paths: PathConfig = field(default_factory=PathConfig)
    data: DataConfig = field(default_factory=DataConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)

    def resolve(self) -> "Config":
        """Resolve all relative paths against the project root."""
        for name, value in vars(self.paths).items():
            path = Path(value)
            if not path.is_absolute():
                setattr(self.paths, name, str(PROJECT_ROOT / path))
        return self

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def load(cls, path: str | Path | None = None) -> "Config":
        path = Path(path) if path else DEFAULT_CONFIG_PATH
        if not path.exists():
            return cls().resolve()
        raw = json.loads(path.read_text(encoding="utf-8"))
        cfg = cls(
            paths=PathConfig(**raw.get("paths", {})),
            data=DataConfig(**raw.get("data", {})),
            model=ModelConfig(**raw.get("model", {})),
            train=TrainConfig(**raw.get("train", {})),
        )
        return cfg.resolve()


def save_json(obj: Any, path: str | Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))
