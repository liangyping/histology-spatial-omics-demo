"""
Simplified model skeleton for the histology-to-spatial omics demo.

This file provides a minimal neural network structure for demonstration only.
It does not contain the full research model or unpublished implementation details.
"""

from typing import Optional

import torch
import torch.nn as nn


class ImageFeatureEncoder(nn.Module):
    """
    A lightweight placeholder encoder for image patch features.

    In a real workflow, image features may come from a pretrained image encoder
    or from precomputed patch-level representations.
    """

    def __init__(self, input_dim: int = 512, hidden_dim: int = 256):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.LayerNorm(hidden_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.encoder(x)


class GeneExpressionPredictor(nn.Module):
    """
    Predict normalized gene expression from image-derived features.
    """

    def __init__(self, hidden_dim: int = 256, n_genes: int = 250):
        super().__init__()
        self.head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, n_genes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        return self.head(features)


class SpatialNeighborhoodAggregation(nn.Module):
    """
    Aggregate features from spatially neighboring spots using a precomputed
    kNN graph. This module allows the model to incorporate spatial context
    from surrounding tissue regions.

    Input:
        features:    (N, D) node features
        edge_index:  (2, E) neighbor pairs from kNN graph
        edge_weight: (E,) edge weights (e.g. inverse distance)

    Output:
        (N, D) spatially refined features
    """

    def __init__(self, feature_dim: int = 256, alpha: float = 0.5):
        super().__init__()
        self.alpha = alpha
        self.norm  = nn.LayerNorm(feature_dim)

    def forward(
        self,
        features: torch.Tensor,
        edge_index: torch.Tensor,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        N = features.shape[0]
        src, dst = edge_index[0], edge_index[1]

        if edge_weight is None:
            w = torch.ones(
                src.shape[0],
                device=features.device,
                dtype=features.dtype,
            )
        else:
            w = edge_weight.to(device=features.device, dtype=features.dtype)

        agg = torch.zeros_like(features)
        messages = features[src] * w.unsqueeze(1)
        agg.scatter_add_(
            dim=0,
            index=dst.unsqueeze(1).expand(-1, features.shape[1]),
            src=messages,
        )

        count = torch.zeros(N, device=features.device, dtype=features.dtype)
        count.scatter_add_(0, dst, w)
        count = count.clamp(min=1.0)

        aggregated = agg / count.unsqueeze(1)
        out = self.alpha * aggregated + (1 - self.alpha) * features
        return self.norm(out)


class DemoHistologyToOmicsModel(nn.Module):
    """
    A simplified model skeleton for histology-to-spatial-omics prediction.

    Input:
        image_features: tensor with shape (N, D)

    Output:
        predicted_expression: tensor with shape (N, G)
    """

    def __init__(
        self,
        input_dim: int = 512,
        hidden_dim: int = 256,
        n_genes: int = 250,
        use_spatial: bool = False,
    ):
        super().__init__()
        self.use_spatial = use_spatial
        self.encoder     = ImageFeatureEncoder(input_dim=input_dim, hidden_dim=hidden_dim)
        self.spatial_agg = SpatialNeighborhoodAggregation(hidden_dim) if use_spatial else None
        self.predictor   = GeneExpressionPredictor(hidden_dim=hidden_dim, n_genes=n_genes)

    def forward(
        self,
        image_features: torch.Tensor,
        edge_index: Optional[torch.Tensor] = None,
        edge_weight: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            image_features: Precomputed image patch features with shape (N, D).
            edge_index:     (2, E) kNN graph edges for spatial aggregation.
            edge_weight:    (E,) edge weights.

        Returns:
            Predicted gene expression with shape (N, G).
        """
        features = self.encoder(image_features)

        if self.use_spatial and self.spatial_agg is not None and edge_index is not None:
            features = self.spatial_agg(features, edge_index, edge_weight)

        return self.predictor(features)
