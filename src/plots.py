from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.manifold import TSNE

from src.utils import ensure_dir


def _finalize_figure(fig: plt.Figure, save_path: str | Path | None = None) -> plt.Figure:
    fig.tight_layout()
    if save_path is not None:
        output_path = Path(save_path)
        ensure_dir(output_path.parent)
        fig.savefig(output_path, dpi=300, bbox_inches="tight")
    return fig


def plot_training_curves(history: pd.DataFrame, title: str, save_path: str | Path | None = None) -> plt.Figure:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history["epoch"], history["train_loss"], label="Train loss")
    axes[0].plot(history["epoch"], history["val_loss"], label="Validation loss")
    axes[0].set_title(f"{title}: Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Cross-entropy")
    axes[0].legend()
    axes[0].grid(alpha=0.3)

    axes[1].plot(history["epoch"], history["train_accuracy"], label="Train accuracy")
    axes[1].plot(history["epoch"], history["val_accuracy"], label="Validation accuracy")
    axes[1].set_title(f"{title}: Accuracy")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].legend()
    axes[1].grid(alpha=0.3)

    return _finalize_figure(fig, save_path)


def plot_model_comparison(
    summary: pd.DataFrame,
    metric: str,
    title: str,
    save_path: str | Path | None = None,
) -> plt.Figure:
    mean_column = f"{metric}_mean"
    std_column = f"{metric}_std"

    fig, ax = plt.subplots(figsize=(6, 4))
    positions = np.arange(len(summary))
    ax.bar(
        positions,
        summary[mean_column],
        yerr=summary[std_column],
        capsize=6,
        color=["#4C78A8", "#F58518"][: len(summary)],
    )
    ax.set_xticks(positions, summary["model"])
    ax.set_ylabel(metric.replace("_", " ").title())
    ax.set_title(title)
    ax.set_ylim(0.0, max(1.0, float((summary[mean_column] + summary[std_column]).max()) + 0.02))
    ax.grid(axis="y", alpha=0.3)

    return _finalize_figure(fig, save_path)


def plot_label_sparsity(
    results: pd.DataFrame,
    metric: str = "test_accuracy",
    save_path: str | Path | None = None,
) -> plt.Figure:
    summary = (
        results.groupby(["labels_per_class", "model"])[metric]
        .agg(mean="mean", std="std")
        .reset_index()
    )

    fig, ax = plt.subplots(figsize=(7, 4.5))
    ordered_models = [model for model in ["MLP", "GCN"] if model in summary["model"].unique()]
    for model_name in ordered_models:
        model_summary = summary[summary["model"] == model_name]
        ax.errorbar(
            model_summary["labels_per_class"],
            model_summary["mean"],
            yerr=model_summary["std"],
            marker="o",
            linewidth=2,
            capsize=4,
            label=model_name,
        )

    ax.set_xlabel("Labeled nodes per class")
    ax.set_xticks(sorted(results["labels_per_class"].unique()))
    ax.set_ylabel("Test accuracy" if metric == "test_accuracy" else metric.replace("_", " ").title())
    ax.set_title("Label Sparsity Study: Test Accuracy vs. Labels per Class")
    ax.legend(title="Model")
    ax.grid(alpha=0.3)

    return _finalize_figure(fig, save_path)


def plot_tsne(
    embeddings: np.ndarray,
    labels: np.ndarray,
    title: str,
    seed: int = 0,
    save_path: str | Path | None = None,
) -> plt.Figure:
    perplexity = min(30, embeddings.shape[0] - 1)
    projection = TSNE(
        n_components=2,
        init="pca",
        learning_rate="auto",
        perplexity=perplexity,
        random_state=seed,
    ).fit_transform(embeddings)

    fig, ax = plt.subplots(figsize=(6.5, 5.5))
    scatter = ax.scatter(
        projection[:, 0],
        projection[:, 1],
        c=labels,
        cmap="tab10",
        s=18,
        alpha=0.85,
    )
    ax.set_title(title)
    ax.set_xticks([])
    ax.set_yticks([])
    fig.colorbar(scatter, ax=ax, fraction=0.046, pad=0.04, label="Class")

    return _finalize_figure(fig, save_path)
