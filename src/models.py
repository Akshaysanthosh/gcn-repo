from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class FeatureMLP(nn.Module):
    """Feature-only two-layer baseline with the same hidden size as the GCN."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.input_layer = nn.Linear(input_dim, hidden_dim)
        self.output_layer = nn.Linear(hidden_dim, output_dim)
        self.dropout = dropout
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.input_layer.weight)
        nn.init.zeros_(self.input_layer.bias)
        nn.init.xavier_uniform_(self.output_layer.weight)
        nn.init.zeros_(self.output_layer.bias)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        return F.relu(self.input_layer(x))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        hidden = F.relu(self.input_layer(x))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.output_layer(hidden)


class GCNLayer(nn.Module):
    """Single graph convolution layer with precomputed normalized adjacency."""

    def __init__(self, input_dim: int, output_dim: int, bias: bool = True) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.empty(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.empty(output_dim))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self) -> None:
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        support = x @ self.weight
        output = adjacency @ support
        if self.bias is not None:
            output = output + self.bias
        return output


class TwoLayerGCN(nn.Module):
    """Paper-style 2-layer GCN for semi-supervised node classification."""

    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, dropout: float = 0.5) -> None:
        super().__init__()
        self.gcn1 = GCNLayer(input_dim, hidden_dim)
        self.gcn2 = GCNLayer(hidden_dim, output_dim)
        self.dropout = dropout

    def encode(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        return F.relu(self.gcn1(x, adjacency))

    def forward(self, x: torch.Tensor, adjacency: torch.Tensor) -> torch.Tensor:
        x = F.dropout(x, p=self.dropout, training=self.training)
        hidden = F.relu(self.gcn1(x, adjacency))
        hidden = F.dropout(hidden, p=self.dropout, training=self.training)
        return self.gcn2(hidden, adjacency)
