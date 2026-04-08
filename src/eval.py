from __future__ import annotations

import pandas as pd
import torch
from sklearn.metrics import f1_score


def compute_accuracy(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1)
    return float((predictions == labels).float().mean().item())


def compute_macro_f1(logits: torch.Tensor, labels: torch.Tensor) -> float:
    predictions = logits.argmax(dim=1).detach().cpu().numpy()
    targets = labels.detach().cpu().numpy()
    return float(f1_score(targets, predictions, average="macro"))


def evaluate_split(logits: torch.Tensor, labels: torch.Tensor, indices: torch.Tensor) -> dict[str, float]:
    split_logits = logits[indices]
    split_labels = labels[indices]
    return {
        "accuracy": compute_accuracy(split_logits, split_labels),
        "macro_f1": compute_macro_f1(split_logits, split_labels),
    }


def summarize_results(results: pd.DataFrame, metric_columns: list[str]) -> pd.DataFrame:
    aggregated = results.groupby("model").agg({metric: ["mean", "std"] for metric in metric_columns})
    aggregated.columns = [f"{metric}_{statistic}" for metric, statistic in aggregated.columns.to_flat_index()]
    return aggregated.reset_index()


def format_mean_std(mean: float, std: float, decimals: int = 4) -> str:
    return f"{mean:.{decimals}f} ± {std:.{decimals}f}"
