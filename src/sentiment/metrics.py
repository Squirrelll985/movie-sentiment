# -*- coding: utf-8 -*-
"""Evaluation metrics implemented with NumPy (no scikit-learn dependency)."""
from __future__ import annotations

from typing import Any

import numpy as np


def confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray) -> np.ndarray:
    matrix = np.zeros((2, 2), dtype=np.int64)
    for t, p in zip(np.asarray(y_true).astype(int), np.asarray(y_pred).astype(int)):
        matrix[t, p] += 1
    return matrix


def roc_auc_score(y_true: np.ndarray, scores: np.ndarray) -> float:
    """Rank-based ROC-AUC with average ranks for ties."""
    y_true = np.asarray(y_true).astype(int)
    scores = np.asarray(scores, dtype=np.float64)
    positives = int((y_true == 1).sum())
    negatives = int((y_true == 0).sum())
    if positives == 0 or negatives == 0:
        return float("nan")
    order = np.argsort(scores, kind="mergesort")
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(scores) + 1)
    sorted_scores = scores[order]
    i = 0
    while i < len(sorted_scores):
        j = i + 1
        while j < len(sorted_scores) and sorted_scores[j] == sorted_scores[i]:
            j += 1
        if j - i > 1:
            ranks[order[i:j]] = (i + 1 + j) / 2.0
        i = j
    positive_rank_sum = ranks[y_true == 1].sum()
    return float((positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives))


def compute_metrics(
    y_true: np.ndarray, y_prob: np.ndarray, threshold: float = 0.5
) -> dict[str, Any]:
    y_true = np.asarray(y_true).astype(int)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    y_pred = (y_prob >= threshold).astype(int)
    matrix = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = matrix.ravel().tolist()
    accuracy = (tp + tn) / max(tp + tn + fp + fn, 1)
    precision = tp / max(tp + fp, 1)
    recall = tp / max(tp + fn, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-12)
    specificity = tn / max(tn + fp, 1)
    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "specificity": float(specificity),
        "roc_auc": roc_auc_score(y_true, y_prob),
        "threshold": float(threshold),
        "confusion_matrix": matrix.tolist(),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
        "n_samples": int(len(y_true)),
    }


def roc_curve_points(y_true: np.ndarray, y_prob: np.ndarray, n_points: int = 400):
    y_true = np.asarray(y_true).astype(int)
    thresholds = np.linspace(1.01, -0.01, n_points)
    tpr, fpr = [], []
    positives = max(int((y_true == 1).sum()), 1)
    negatives = max(int((y_true == 0).sum()), 1)
    for threshold in thresholds:
        pred = y_prob >= threshold
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        tpr.append(tp / positives)
        fpr.append(fp / negatives)
    return np.asarray(fpr), np.asarray(tpr)


def precision_recall_curve_points(y_true: np.ndarray, y_prob: np.ndarray, n_points: int = 400):
    y_true = np.asarray(y_true).astype(int)
    thresholds = np.linspace(1.01, -0.01, n_points)
    precisions, recalls = [], []
    positives = max(int((y_true == 1).sum()), 1)
    for threshold in thresholds:
        pred = y_prob >= threshold
        tp = int(((pred == 1) & (y_true == 1)).sum())
        fp = int(((pred == 1) & (y_true == 0)).sum())
        precisions.append(tp / max(tp + fp, 1))
        recalls.append(tp / positives)
    return np.asarray(recalls), np.asarray(precisions)
