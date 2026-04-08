from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd
import torch
from torch import nn

from src.data import GraphData
from src.eval import evaluate_split
from src.models import FeatureMLP, TwoLayerGCN
from src.utils import set_seed


@dataclass
class TrainConfig:
    hidden_dim: int = 16
    dropout: float = 0.5
    learning_rate: float = 0.01
    weight_decay: float = 5e-4
    max_epochs: int = 200
    patience: int = 10


@dataclass
class TrainingResult:
    model: nn.Module
    history: pd.DataFrame
    stop_epoch: int
    best_epoch: int
    best_val_loss: float


def build_model(
    model_name: str,
    input_dim: int,
    hidden_dim: int,
    output_dim: int,
    dropout: float,
) -> nn.Module:
    model_key = model_name.lower()
    if model_key == "mlp":
        return FeatureMLP(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout)
    if model_key == "gcn":
        return TwoLayerGCN(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, dropout=dropout)
    raise ValueError(f"Unsupported model: {model_name}")


def _forward(model: nn.Module, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    if isinstance(model, FeatureMLP):
        return model(features)
    if isinstance(model, TwoLayerGCN):
        return model(features, adjacency)
    raise TypeError(f"Unexpected model type: {type(model)!r}")


def extract_embeddings(model: nn.Module, features: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
    model.eval()
    with torch.no_grad():
        if isinstance(model, FeatureMLP):
            return model.encode(features)
        if isinstance(model, TwoLayerGCN):
            return model.encode(features, adjacency)
    raise TypeError(f"Unexpected model type: {type(model)!r}")


def _regularization_loss(model: nn.Module) -> torch.Tensor:
    if isinstance(model, FeatureMLP):
        return 0.5 * model.input_layer.weight.pow(2).sum()
    if isinstance(model, TwoLayerGCN):
        return 0.5 * model.gcn1.weight.pow(2).sum()
    raise TypeError(f"Unexpected model type: {type(model)!r}")


def fit_model(
    model: nn.Module,
    data: GraphData,
    config: TrainConfig,
    device: torch.device,
) -> TrainingResult:
    features = data.features.to(device)
    adjacency = data.adjacency.to(device)
    labels = data.labels.to(device)
    idx_train = data.idx_train.to(device)
    idx_val = data.idx_val.to(device)

    model = model.to(device)
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=0.0,
    )

    best_val_loss = float("inf")
    best_epoch = 0
    stop_epoch = config.max_epochs
    val_losses: list[float] = []
    history_rows: list[dict[str, float]] = []

    for epoch in range(1, config.max_epochs + 1):
        model.train()
        optimizer.zero_grad()
        logits = _forward(model, features, adjacency)
        data_loss = nn.functional.cross_entropy(logits[idx_train], labels[idx_train])
        train_loss = data_loss + config.weight_decay * _regularization_loss(model)
        train_loss.backward()
        optimizer.step()

        model.eval()
        with torch.no_grad():
            logits = _forward(model, features, adjacency)
            val_loss = nn.functional.cross_entropy(logits[idx_val], labels[idx_val]).item()
            train_metrics = evaluate_split(logits, labels, idx_train)
            val_metrics = evaluate_split(logits, labels, idx_val)

        history_rows.append(
            {
                "epoch": epoch,
                "train_loss": float(train_loss.item()),
                "train_data_loss": float(data_loss.item()),
                "val_loss": float(val_loss),
                "train_accuracy": train_metrics["accuracy"],
                "train_macro_f1": train_metrics["macro_f1"],
                "val_accuracy": val_metrics["accuracy"],
                "val_macro_f1": val_metrics["macro_f1"],
            }
        )

        val_losses.append(float(val_loss))
        if val_loss < best_val_loss:
            best_val_loss = float(val_loss)
            best_epoch = epoch

        if epoch > config.patience and val_losses[-1] > float(pd.Series(val_losses[-(config.patience + 1) : -1]).mean()):
            stop_epoch = epoch
            break

    history = pd.DataFrame(history_rows)
    if history.empty:
        raise RuntimeError("Training produced no history rows.")
    if stop_epoch == config.max_epochs:
        stop_epoch = int(history["epoch"].iloc[-1])
    return TrainingResult(
        model=model,
        history=history,
        stop_epoch=stop_epoch,
        best_epoch=best_epoch,
        best_val_loss=best_val_loss,
    )


def run_experiment(
    model_name: str,
    data: GraphData,
    config: TrainConfig,
    seed: int,
    device: torch.device,
) -> dict[str, Any]:
    set_seed(seed)
    model = build_model(
        model_name=model_name,
        input_dim=data.num_features,
        hidden_dim=config.hidden_dim,
        output_dim=data.num_classes,
        dropout=config.dropout,
    )

    training_result = fit_model(model=model, data=data, config=config, device=device)
    features = data.features.to(device)
    adjacency = data.adjacency.to(device)
    labels = data.labels.to(device)
    idx_test = data.idx_test.to(device)

    training_result.model.eval()
    with torch.no_grad():
        logits = _forward(training_result.model, features, adjacency)
        test_metrics = evaluate_split(logits, labels, idx_test)
        embeddings = extract_embeddings(training_result.model, features, adjacency).detach().cpu()

    return {
        "model": model_name.upper(),
        "seed": seed,
        "test_accuracy": test_metrics["accuracy"],
        "test_macro_f1": test_metrics["macro_f1"],
        "stop_epoch": training_result.stop_epoch,
        "best_epoch": training_result.best_epoch,
        "best_val_loss": training_result.best_val_loss,
        "history": training_result.history,
        "embeddings": embeddings,
        "trained_model": training_result.model,
    }


def run_repeated_experiments(
    model_name: str,
    data: GraphData,
    config: TrainConfig,
    seeds: list[int],
    device: torch.device,
) -> tuple[pd.DataFrame, list[dict[str, Any]]]:
    artifacts: list[dict[str, Any]] = []
    summary_rows: list[dict[str, float | int | str]] = []

    for seed in seeds:
        result = run_experiment(model_name=model_name, data=data, config=config, seed=seed, device=device)
        artifacts.append(result)
        summary_rows.append(
            {
                "model": result["model"],
                "seed": seed,
                "test_accuracy": result["test_accuracy"],
                "test_macro_f1": result["test_macro_f1"],
                "stop_epoch": result["stop_epoch"],
                "best_epoch": result["best_epoch"],
                "best_val_loss": result["best_val_loss"],
            }
        )

    return pd.DataFrame(summary_rows), artifacts
