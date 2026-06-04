"""
Basic data utilities for the histology-to-spatial omics demo.

This file provides lightweight helper functions for loading expression
matrices, spatial coordinates, and checking input shapes. It is intended
for demonstration only and does not include project-specific preprocessing.
"""

from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd


def load_expression_csv(path: str) -> Tuple[np.ndarray, list]:
    """
    Load a gene expression matrix from a CSV file.

    Expected format:
        rows    = spatial spots or samples
        columns = genes

    Args:
        path: Path to expression CSV file.

    Returns:
        expression: NumPy array with shape (N, G)
        gene_names: List of gene names.
    """
    df = pd.read_csv(path)
    gene_names = list(df.columns)
    expression = df.values.astype(np.float32)
    return expression, gene_names


def load_coordinates_csv(path: str) -> np.ndarray:
    """
    Load spatial coordinates from a CSV file.

    Expected columns:
        x, y

    Args:
        path: Path to coordinate CSV file.

    Returns:
        coordinates: NumPy array with shape (N, 2).
    """
    df = pd.read_csv(path)

    if not {"x", "y"}.issubset(df.columns):
        raise ValueError("Coordinate file must contain 'x' and 'y' columns.")

    return df[["x", "y"]].values.astype(np.float32)


def normalize_expression(expression: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normalize expression values.

    Args:
        expression: Gene expression matrix with shape (N, G).
        method: Normalization method. Options: 'minmax', 'zscore', or 'log1p'.

    Returns:
        Normalized expression matrix.
    """
    expression = expression.astype(np.float32)

    if method == "log1p":
        return np.log1p(expression)

    if method == "minmax":
        min_val = expression.min(axis=0, keepdims=True)
        max_val = expression.max(axis=0, keepdims=True)
        return (expression - min_val) / (max_val - min_val + 1e-8)

    if method == "zscore":
        mean = expression.mean(axis=0, keepdims=True)
        std = expression.std(axis=0, keepdims=True)
        return (expression - mean) / (std + 1e-8)

    raise ValueError(f"Unsupported normalization method: {method}")


def check_data_shapes(
    expression: np.ndarray,
    coordinates: np.ndarray,
    patch_features: Optional[np.ndarray] = None,
) -> None:
    """
    Check whether expression, coordinates, and optional patch features
    have compatible sample sizes.
    """
    n_samples = expression.shape[0]

    if coordinates.shape[0] != n_samples:
        raise ValueError(
            f"Expression has {n_samples} samples, but coordinates have "
            f"{coordinates.shape[0]} samples."
        )

    if patch_features is not None and patch_features.shape[0] != n_samples:
        raise ValueError(
            f"Expression has {n_samples} samples, but patch features have "
            f"{patch_features.shape[0]} samples."
        )


def ensure_dir(path: str) -> Path:
    """
    Create a directory if it does not exist.

    Args:
        path: Directory path.

    Returns:
        Path object.
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj
