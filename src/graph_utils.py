"""
Graph construction utilities for spatial neighborhood modeling.

Builds k-nearest-neighbor graphs from spot coordinates to capture
spatial correlations between neighboring tissue regions.
"""

import numpy as np
from typing import Optional


def build_knn_graph(
    coords: np.ndarray,
    k: int = 6,
    radius: Optional[float] = None,
    self_loops: bool = False,
) -> dict:
    """
    Build a k-nearest-neighbor spatial graph from spot coordinates.

    Args:
        coords:      (N, 2) array of (x, y) spatial coordinates
        k:           number of neighbors per spot
        radius:      optional maximum distance threshold
        self_loops:  whether to include self-edges

    Returns:
        dict with keys:
            "edge_index":  (2, E) source/destination node pairs
            "edge_weight": (E,) inverse-distance edge weights
            "n_nodes":     number of spots N
    """
    from scipy.spatial import cKDTree

    tree = cKDTree(coords)
    distances, indices = tree.query(coords, k=k + 1)

    src, dst, weights = [], [], []
    for i, (dists, nbrs) in enumerate(zip(distances[:, 1:], indices[:, 1:])):
        for d, j in zip(dists, nbrs):
            if radius is not None and d > radius:
                continue
            src.append(i)
            dst.append(int(j))
            weights.append(1.0 / (d + 1e-6))

    if self_loops:
        for i in range(len(coords)):
            src.append(i)
            dst.append(i)
            weights.append(1.0)

    return {
        "edge_index":  np.array([src, dst], dtype=np.int64),
        "edge_weight": np.array(weights, dtype=np.float32),
        "n_nodes":     len(coords),
    }


def normalize_edge_weights(edge_weight: np.ndarray, edge_index: np.ndarray, n_nodes: int) -> np.ndarray:
    """
    Normalize edge weights so that incoming weights for each target node sum to 1.

    Args:
        edge_weight: (E,) raw edge weights
        edge_index:  (2, E) edge pairs
        n_nodes:     number of nodes

    Returns:
        (E,) normalized edge weights
    """
    dst = edge_index[1]
    row_sum = np.zeros(n_nodes, dtype=np.float32)
    np.add.at(row_sum, dst, edge_weight)
    row_sum = np.where(row_sum > 0, row_sum, 1.0)
    return edge_weight / row_sum[dst]


def graph_smoothing(
    features: np.ndarray,
    edge_index: np.ndarray,
    edge_weight: np.ndarray,
    n_iters: int = 1,
    alpha: float = 0.5,
) -> np.ndarray:
    """
    Smooth node features by aggregating from spatial neighbors.

    Args:
        features:    (N, D) node feature matrix
        edge_index:  (2, E) edge connectivity
        edge_weight: (E,) edge weights
        n_iters:     number of smoothing iterations
        alpha:       blend ratio (alpha * neighbor_agg + (1-alpha) * self)

    Returns:
        (N, D) smoothed feature matrix
    """
    N = features.shape[0]
    src, dst = edge_index[0], edge_index[1]
    out = features.copy()

    for _ in range(n_iters):
        agg = np.zeros_like(out)
        np.add.at(agg, dst, out[src] * edge_weight[:, None])
        count = np.zeros(N, dtype=np.float32)
        np.add.at(count, dst, edge_weight)
        count = np.where(count > 0, count, 1.0)
        out = alpha * (agg / count[:, None]) + (1 - alpha) * out

    return out
