# -*- coding: utf-8 -*-
"""Train and evaluate a sentiment classifier on the IMDb dataset.

Example
-------
python -m sentiment.train --model bilstm_attention --output-dir outputs/main
"""
from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import Config, save_json
from .data import DataBundle, prepare_data
from .metrics import compute_metrics
from .model import build_model, count_parameters, save_model_checkpoint
from .utils import AverageMeter, Timer, format_metrics, get_device, human_time, set_seed


class ReviewDataset(Dataset):
    def __init__(self, inputs: np.ndarray, labels: np.ndarray) -> None:
        self.inputs = inputs
        self.labels = labels

    def __len__(self) -> int:
        return len(self.labels)

    def __getitem__(self, index: int):
        return (
            torch.from_numpy(self.inputs[index].astype(np.int64)),
            torch.tensor(int(self.labels[index]), dtype=torch.long),
        )


def make_loader(
    inputs: np.ndarray, labels: np.ndarray, batch_size: int, shuffle: bool, num_workers: int = 0
) -> DataLoader:
    return DataLoader(
        ReviewDataset(inputs, labels),
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
    )


@torch.no_grad()
def infer(
    model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device
) -> tuple[float, np.ndarray, np.ndarray]:
    model.eval()
    loss_meter = AverageMeter()
    probs: list[np.ndarray] = []
    labels: list[np.ndarray] = []
    for input_ids, targets in loader:
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        lengths = (input_ids != 0).sum(dim=1).clamp(min=1)
        outputs = model(input_ids, lengths)
        loss = criterion(outputs["logits"], targets)
        loss_meter.update(loss.item(), input_ids.size(0))
        probs.append(torch.softmax(outputs["logits"], dim=-1)[:, 1].cpu().numpy())
        labels.append(targets.cpu().numpy())
    return loss_meter.avg, np.concatenate(labels), np.concatenate(probs)


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    grad_clip: float,
    epoch: int,
    total_epochs: int,
    log,
) -> float:
    model.train()
    loss_meter = AverageMeter()
    correct = 0
    seen = 0
    started = time.perf_counter()
    for step, (input_ids, targets) in enumerate(loader, start=1):
        input_ids = input_ids.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        lengths = (input_ids != 0).sum(dim=1).clamp(min=1)

        optimizer.zero_grad(set_to_none=True)
        outputs = model(input_ids, lengths)
        loss = criterion(outputs["logits"], targets)
        loss.backward()
        if grad_clip > 0:
            nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        loss_meter.update(loss.item(), input_ids.size(0))
        correct += (outputs["logits"].argmax(dim=1) == targets).sum().item()
        seen += input_ids.size(0)
        if step % 50 == 0 or step == len(loader):
            elapsed = time.perf_counter() - started
            speed = seen / max(elapsed, 1e-9)
            log(
                f"  epoch {epoch:02d}/{total_epochs:02d} | step {step:04d}/{len(loader):04d} | "
                f"loss {loss_meter.avg:.4f} | acc {correct/seen:.4f} | {speed:5.1f} samples/s"
            )
    return loss_meter.avg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train an LSTM sentiment classifier")
    parser.add_argument("--config", type=str, default=None)
    parser.add_argument("--model", type=str, default=None, choices=["lstm", "bilstm", "bilstm_attention", "textcnn"])
    parser.add_argument("--output-dir", type=str, default="outputs/main")
    parser.add_argument("--epochs", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=None)
    parser.add_argument("--learning-rate", type=float, default=None)
    parser.add_argument("--max-len", type=int, default=None)
    parser.add_argument("--max-vocab", type=int, default=None)
    parser.add_argument("--embedding-dim", type=int, default=None)
    parser.add_argument("--hidden-dim", type=int, default=None)
    parser.add_argument("--num-layers", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--device", type=str, default=None)
    parser.add_argument("--threads", type=int, default=0, help="CPU threads (0 keeps PyTorch default)")
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--limit", type=int, default=None, help="debug: cap training/validation samples")
    parser.add_argument("--no-test", action="store_true", help="skip test-set evaluation")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = Config.load(args.config).resolve()
    if args.model:
        config.model.name = args.model
    if args.epochs:
        config.train.epochs = args.epochs
    if args.batch_size:
        config.train.batch_size = args.batch_size
    if args.learning_rate:
        config.train.learning_rate = args.learning_rate
    if args.max_len:
        config.data.max_len = args.max_len
    if args.max_vocab:
        config.data.max_vocab = args.max_vocab
    if args.embedding_dim:
        config.model.embedding_dim = args.embedding_dim
    if args.hidden_dim:
        config.model.hidden_dim = args.hidden_dim
    if args.num_layers:
        config.model.num_layers = args.num_layers
    if args.dropout is not None:
        config.model.dropout = args.dropout
    if args.device:
        config.train.device = args.device
    if args.seed is not None:
        config.train.seed = args.seed

    output_dir = Path(args.output_dir)
    if not output_dir.is_absolute():
        from .config import PROJECT_ROOT

        output_dir = PROJECT_ROOT / output_dir
    output_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"
    log_file = log_path.open("w", encoding="utf-8")

    def log(message: str) -> None:
        print(message, flush=True)
        log_file.write(message + "\n")
        log_file.flush()

    set_seed(config.train.seed)
    if args.threads and args.threads > 0:
        torch.set_num_threads(args.threads)
        torch.set_num_interop_threads(max(1, min(4, args.threads // 2)))
    device = get_device(config.train.device)
    log("=" * 78)
    log(f"model={config.model.name} | device={device} | seed={config.train.seed}")
    log(f"torch_threads={torch.get_num_threads()} | output_dir={output_dir}")
    log("=" * 78)

    data: DataBundle = prepare_data(config)
    available_len = data.train_x.shape[1]
    if config.data.max_len != available_len:
        seq_len = min(config.data.max_len, available_len)
        data.train_x = data.train_x[:, :seq_len]
        data.val_x = data.val_x[:, :seq_len]
        data.test_x = data.test_x[:, :seq_len]
        config.data.max_len = seq_len
        log(f"[data] sliced cached sequences to max_len={seq_len}")
    log(f"data: {json.dumps(data.stats, ensure_ascii=False)}")

    train_x, train_y = data.train_x, data.train_y
    val_x, val_y = data.val_x, data.val_y
    test_x, test_y = data.test_x, data.test_y
    if args.limit:
        rng = np.random.default_rng(config.train.seed)
        train_idx = rng.choice(len(train_y), size=min(args.limit, len(train_y)), replace=False)
        val_idx = rng.choice(len(val_y), size=min(max(args.limit // 5, 32), len(val_y)), replace=False)
        train_x, train_y = train_x[train_idx], train_y[train_idx]
        val_x, val_y = val_x[val_idx], val_y[val_idx]
        log(f"[debug] limited train={len(train_y)} val={len(val_y)}")

    train_loader = make_loader(
        train_x, train_y, config.train.batch_size, True, config.data.num_workers
    )
    val_loader = make_loader(val_x, val_y, config.train.batch_size * 2, False, config.data.num_workers)
    test_loader = make_loader(test_x, test_y, config.train.batch_size * 2, False, config.data.num_workers)

    model = build_model(config.model.name, data.vocab_size, config.model).to(device)
    n_params = count_parameters(model)
    log(f"parameters={n_params:,} ({n_params/1e6:.2f}M) | vocab={data.vocab_size}")

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config.train.learning_rate, weight_decay=config.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="max", factor=0.5, patience=1, min_lr=1e-6
    )

    history: list[dict] = []
    best_score = (-1.0, -1.0)
    best_metrics: dict = {}
    best_epoch = 0
    epochs_without_improvement = 0
    checkpoint_path = output_dir / "best_model.pt"
    timer = Timer()

    for epoch in range(1, config.train.epochs + 1):
        started = time.perf_counter()
        train_loss = train_one_epoch(
            model,
            train_loader,
            criterion,
            optimizer,
            device,
            config.train.grad_clip,
            epoch,
            config.train.epochs,
            log,
        )
        val_loss, y_true, y_prob = infer(model, val_loader, criterion, device)
        val_metrics = compute_metrics(y_true, y_prob)
        lr = optimizer.param_groups[0]["lr"]
        scheduler.step(val_metrics["f1"])
        record = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_accuracy": val_metrics["accuracy"],
            "val_precision": val_metrics["precision"],
            "val_recall": val_metrics["recall"],
            "val_f1": val_metrics["f1"],
            "val_roc_auc": val_metrics["roc_auc"],
            "learning_rate": lr,
            "elapsed_seconds": time.perf_counter() - started,
        }
        history.append(record)
        log(
            f"epoch {epoch:02d} | train_loss {train_loss:.4f} | val_loss {val_loss:.4f} | "
            f"{format_metrics(val_metrics)} | {human_time(record['elapsed_seconds'])}"
        )

        score = (val_metrics["f1"], val_metrics["accuracy"])
        if score > best_score:
            best_score = score
            best_metrics = val_metrics
            best_epoch = epoch
            epochs_without_improvement = 0
            save_model_checkpoint(
                model,
                checkpoint_path,
                config.model.name,
                config.model,
                data.vocab,
                val_metrics,
                extra={
                    "epoch": epoch,
                    "train_config": asdict(config.train),
                    "data_config": asdict(config.data),
                    "data_stats": data.stats,
                    "parameter_count": n_params,
                },
            )
            log(f"  saved new best checkpoint -> {checkpoint_path}")
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= config.train.patience:
                log(f"early stopping at epoch {epoch} (patience={config.train.patience})")
                break

    total_time = timer.elapsed()
    log(f"training finished in {human_time(total_time)} | best epoch={best_epoch}")

    test_metrics: dict = {}
    if not args.no_test:
        from .model import load_model

        model, checkpoint = load_model(checkpoint_path, device)
        test_loss, test_true, test_prob = infer(model, test_loader, criterion, device)
        test_metrics = compute_metrics(test_true, test_prob)
        test_metrics["loss"] = float(test_loss)
        test_metrics["parameter_count"] = int(n_params)
        log(f"TEST | {format_metrics(test_metrics)} | loss={test_loss:.4f}")
        np.savez_compressed(
            output_dir / "test_predictions.npz", y_true=test_true, y_prob=test_prob
        )

    results = {
        "model": config.model.name,
        "model_config": asdict(config.model),
        "train_config": asdict(config.train),
        "data_config": asdict(config.data),
        "data_stats": data.stats,
        "parameter_count": int(n_params),
        "best_epoch": int(best_epoch),
        "training_seconds": float(total_time),
        "history": history,
        "best_val_metrics": best_metrics,
        "test_metrics": test_metrics,
    }
    save_json(results, output_dir / "metrics.json")
    save_json(history, output_dir / "history.json")
    save_json(config.to_dict(), output_dir / "config.json")
    log(f"results written to {output_dir / 'metrics.json'}")
    log_file.close()


if __name__ == "__main__":
    main()
