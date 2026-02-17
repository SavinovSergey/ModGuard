"""Shared training utilities for model scripts."""
from __future__ import annotations

from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import precision_recall_curve


class BinaryFocalLoss(nn.Module):
    """Focal loss for binary classification on logits."""

    def __init__(self, alpha: float | None = None, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        targets = targets.float()
        bce_loss = F.binary_cross_entropy_with_logits(logits, targets, reduction="none")
        probs = torch.sigmoid(logits)
        p_t = probs * targets + (1.0 - probs) * (1.0 - targets)
        focal_factor = (1.0 - p_t).pow(self.gamma)
        loss = focal_factor * bce_loss

        if self.alpha is not None:
            alpha_t = self.alpha * targets + (1.0 - self.alpha) * (1.0 - targets)
            loss = alpha_t * loss

        if self.reduction == "mean":
            return loss.mean()
        if self.reduction == "sum":
            return loss.sum()
        return loss


def find_optimal_threshold(
    y_true: np.ndarray,
    y_proba: np.ndarray,
    min_precision: float = 0.9,
) -> tuple[float, float, float]:
    """
    Find threshold maximizing recall at precision >= min_precision.

    Returns:
        (optimal_threshold, precision_at_threshold, recall_at_threshold)
    """
    precision, recall, thresholds = precision_recall_curve(y_true, y_proba)

    # precision_recall_curve returns precision/recall with one extra point
    # that does not correspond to a threshold.
    precision = precision[:-1]
    recall = recall[:-1]

    valid_indices = np.where(precision >= min_precision)[0]

    if len(valid_indices) > 0:
        best_idx = valid_indices[np.argmax(recall[valid_indices])]
    else:
        best_idx = int(np.argmax(recall))
        print(
            f"  Warning: could not reach precision >= {min_precision:.2f}, using max recall threshold"
        )

    if len(thresholds) == 0:
        optimal_threshold = 0.5
    elif best_idx < len(thresholds):
        optimal_threshold = float(thresholds[best_idx])
    else:
        optimal_threshold = float(thresholds[-1])

    return optimal_threshold, float(precision[best_idx]), float(recall[best_idx])


def compute_auto_alpha(y_train: np.ndarray) -> float:
    """
    Estimate alpha for focal loss.

    alpha = share of negative class to upweight rare positive class.
    """
    positives = float(np.sum(y_train > 0.5))
    negatives = float(np.sum(y_train <= 0.5))
    total = positives + negatives
    if total == 0 or positives == 0 or negatives == 0:
        return 0.5
    alpha = negatives / total
    return float(np.clip(alpha, 0.05, 0.95))


def convert_to_json_serializable(obj: Any) -> Any:
    """Convert numpy/scalar objects to plain JSON-serializable values."""
    if isinstance(obj, (np.integer, np.floating)):
        return obj.item()
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    if isinstance(obj, dict):
        return {key: convert_to_json_serializable(value) for key, value in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [convert_to_json_serializable(item) for item in obj]
    if isinstance(obj, (bool, int, float, str, type(None))):
        return obj
    try:
        return str(obj)
    except Exception:
        return repr(obj)
