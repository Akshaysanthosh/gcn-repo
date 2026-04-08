from __future__ import annotations

import pickle as pkl
import urllib.request
import warnings
from dataclasses import dataclass
from pathlib import Path
from urllib.error import URLError

import numpy as np
import scipy.sparse as sp
import torch

from src.utils import ensure_dir

PLANETOID_BASE_URL = "https://raw.githubusercontent.com/tkipf/gcn/master/gcn/data"
PLANETOID_FILES = (
    "ind.cora.x",
    "ind.cora.y",
    "ind.cora.tx",
    "ind.cora.ty",
    "ind.cora.allx",
    "ind.cora.ally",
    "ind.cora.graph",
    "ind.cora.test.index",
)


@dataclass
class GraphData:
    features: torch.Tensor
    adjacency: torch.Tensor
    labels: torch.Tensor
    idx_train: torch.Tensor
    idx_val: torch.Tensor
    idx_test: torch.Tensor

    @property
    def num_nodes(self) -> int:
        return int(self.features.shape[0])

    @property
    def num_features(self) -> int:
        return int(self.features.shape[1])

    @property
    def num_classes(self) -> int:
        return int(torch.unique(self.labels).numel())


def with_splits(
    data: GraphData,
    idx_train: torch.Tensor,
    idx_val: torch.Tensor,
    idx_test: torch.Tensor,
) -> GraphData:
    """Return a new GraphData object with the same graph and updated splits."""
    return GraphData(
        features=data.features,
        adjacency=data.adjacency,
        labels=data.labels,
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
    )


def download_cora_raw(data_dir: str | Path = "data") -> Path:
    """Download the original Planetoid-format Cora files if missing."""
    data_path = ensure_dir(data_dir)

    for file_name in PLANETOID_FILES:
        target_path = data_path / file_name
        if not target_path.exists():
            url = f"{PLANETOID_BASE_URL}/{file_name}"
            try:
                urllib.request.urlretrieve(url, target_path)
            except URLError as error:
                raise RuntimeError(
                    "Failed to download the original Cora files. "
                    "Ensure network access is available or place the raw Planetoid files in the data directory manually."
                ) from error

    return data_path


def _parse_index_file(path: Path) -> list[int]:
    with path.open("r", encoding="utf-8") as handle:
        return [int(line.strip()) for line in handle]


def _load_pickle(path: Path):
    with path.open("rb") as handle:
        with warnings.catch_warnings():
            warnings.filterwarnings("ignore", category=np.exceptions.VisibleDeprecationWarning)
            return pkl.load(handle, encoding="latin1")


def _row_normalize(features: sp.spmatrix) -> sp.csr_matrix:
    rowsum = np.asarray(features.sum(1)).flatten()
    inverse = np.zeros_like(rowsum, dtype=np.float32)
    nonzero = rowsum != 0
    inverse[nonzero] = 1.0 / rowsum[nonzero]
    return sp.diags(inverse) @ features


def _normalize_adjacency(adjacency: sp.spmatrix) -> sp.csr_matrix:
    rowsum = np.asarray(adjacency.sum(1)).flatten()
    inverse_sqrt = np.zeros_like(rowsum, dtype=np.float32)
    nonzero = rowsum != 0
    inverse_sqrt[nonzero] = np.power(rowsum[nonzero], -0.5)
    degree = sp.diags(inverse_sqrt)
    return degree @ adjacency @ degree


def _build_adjacency(graph: dict[int, list[int]], num_nodes: int) -> sp.csr_matrix:
    rows: list[int] = []
    cols: list[int] = []
    for source, neighbors in graph.items():
        rows.extend([source] * len(neighbors))
        cols.extend(neighbors)

    data = np.ones(len(rows), dtype=np.float32)
    adjacency = sp.coo_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float32)
    adjacency = adjacency + adjacency.T.multiply(adjacency.T > adjacency) - adjacency.multiply(adjacency.T > adjacency)
    return adjacency.tocsr()


def load_cora_dataset(data_dir: str | Path = "data") -> GraphData:
    """
    Load Cora using the original public Planetoid split from the GCN codebase.

    The split matches the commonly used semi-supervised setup:
    140 training nodes, 500 validation nodes, and 1000 test nodes.
    Features are row-normalized, and adjacency is symmetrized before applying
    the GCN renormalization A_hat = D^(-1/2) (A + I) D^(-1/2).
    """
    data_path = download_cora_raw(data_dir)

    _ = _load_pickle(data_path / "ind.cora.x")
    y = _load_pickle(data_path / "ind.cora.y")
    tx = _load_pickle(data_path / "ind.cora.tx")
    ty = _load_pickle(data_path / "ind.cora.ty")
    allx = _load_pickle(data_path / "ind.cora.allx")
    ally = _load_pickle(data_path / "ind.cora.ally")
    graph = _load_pickle(data_path / "ind.cora.graph")

    test_idx_reorder = _parse_index_file(data_path / "ind.cora.test.index")
    test_idx_range = np.sort(test_idx_reorder)

    features = sp.vstack((allx, tx)).tolil()
    features[test_idx_reorder, :] = features[test_idx_range, :]
    features = _row_normalize(features).tocsr()

    labels = np.vstack((ally, ty))
    labels[test_idx_reorder, :] = labels[test_idx_range, :]
    label_ids = labels.argmax(axis=1)

    adjacency = _build_adjacency(graph, labels.shape[0])
    adjacency = _normalize_adjacency(adjacency + sp.eye(adjacency.shape[0], dtype=np.float32)).astype(np.float32)

    idx_train = torch.arange(len(y), dtype=torch.long)
    idx_val = torch.arange(len(y), len(y) + 500, dtype=torch.long)
    idx_test = torch.tensor(test_idx_range, dtype=torch.long)

    return GraphData(
        features=torch.from_numpy(features.toarray()).float(),
        adjacency=torch.from_numpy(adjacency.toarray()).float(),
        labels=torch.from_numpy(label_ids).long(),
        idx_train=idx_train,
        idx_val=idx_val,
        idx_test=idx_test,
    )


def sample_label_sparsity_split(
    labels: torch.Tensor,
    idx_test: torch.Tensor,
    labels_per_class: int,
    seed: int,
    val_size: int = 500,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Create a reduced-label split while keeping the original test nodes fixed.

    Training nodes are sampled class-balanced from the non-test nodes.
    Validation nodes are sampled uniformly at random from the remaining non-test nodes.
    """
    labels_np = labels.cpu().numpy()
    idx_test_np = idx_test.cpu().numpy()
    candidate_idx = np.setdiff1d(np.arange(labels_np.shape[0]), idx_test_np)
    rng = np.random.default_rng(seed)

    train_indices: list[int] = []
    num_classes = int(labels.max().item()) + 1

    for class_id in range(num_classes):
        class_candidates = candidate_idx[labels_np[candidate_idx] == class_id]
        if class_candidates.shape[0] < labels_per_class:
            raise ValueError(
                f"Class {class_id} only has {class_candidates.shape[0]} candidate nodes; "
                f"cannot sample {labels_per_class} labels per class."
            )
        rng.shuffle(class_candidates)
        train_indices.extend(class_candidates[:labels_per_class].tolist())

    train_idx = np.array(sorted(train_indices), dtype=np.int64)
    remaining_idx = np.setdiff1d(candidate_idx, train_idx)
    rng.shuffle(remaining_idx)
    val_idx = np.sort(remaining_idx[:val_size])

    return (
        torch.tensor(train_idx, dtype=torch.long),
        torch.tensor(val_idx, dtype=torch.long),
        idx_test.clone(),
    )
