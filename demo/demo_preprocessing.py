"""
Demo: Preprocessing pipeline for histology-to-spatial omics prediction.

This script illustrates the general structure of a preprocessing workflow,
using synthetic data. No real patient data or project-specific tools are used.

Usage:
    python demo/demo_preprocessing.py --config configs/example_config.yaml
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data_utils import normalize_expression, check_data_shapes
from src.graph_utils import build_knn_graph


def parse_args():
    parser = argparse.ArgumentParser(
        description="Demo preprocessing pipeline (synthetic data)"
    )
    parser.add_argument(
        "--config",
        default="configs/example_config.yaml",
        help="Path to config YAML file",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    config_path = Path(path)
    if not config_path.exists():
        return {}
    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def make_synthetic_data(n_spots: int = 200, n_genes: int = 250, seed: int = 42):
    """Generate synthetic spatial transcriptomics data for demo purposes."""
    rng = np.random.default_rng(seed)
    expression  = rng.integers(0, 100, size=(n_spots, n_genes)).astype(np.float32)
    coordinates = rng.uniform(0, 10000, size=(n_spots, 2)).astype(np.float32)
    gene_names  = [f"GENE_{i:04d}" for i in range(n_genes)]
    return expression, coordinates, gene_names


def run_demo(cfg: dict):
    n_genes = cfg.get("data", {}).get("n_genes", 250)
    knn_k   = cfg.get("graph", {}).get("knn_k", 6)

    print("=" * 55)
    print("  Histology-to-Spatial Omics Preprocessing Demo")
    print("=" * 55)

    # ── Step 1: Generate synthetic data ───────────────────
    print("\n[1] Loading data (synthetic demo)...")
    expr, coords, genes = make_synthetic_data(n_spots=200, n_genes=n_genes)
    print(f"    Spots:      {expr.shape[0]}")
    print(f"    Genes:      {expr.shape[1]}")
    print(f"    Coord range: x=[{coords[:,0].min():.0f}, {coords[:,0].max():.0f}]"
          f"  y=[{coords[:,1].min():.0f}, {coords[:,1].max():.0f}]")

    # ── Step 2: Check data shapes ──────────────────────────
    print("\n[2] Checking data consistency...")
    check_data_shapes(expr, coords)
    print("    OK")

    # ── Step 3: Normalize expression ──────────────────────
    print("\n[3] Normalizing gene expression (log1p)...")
    expr_norm = normalize_expression(expr, method="log1p")
    print(f"    Range after normalization: [{expr_norm.min():.3f}, {expr_norm.max():.3f}]")

    # ── Step 4: Build spatial kNN graph ───────────────────
    print(f"\n[4] Building spatial kNN graph (k={knn_k})...")
    graph = build_knn_graph(coords, k=knn_k)
    print(f"    Nodes: {graph['n_nodes']}")
    print(f"    Edges: {graph['edge_index'].shape[1]}")
    print(f"    Mean edge weight: {graph['edge_weight'].mean():.6f}")

    # ── Step 5: Summary ───────────────────────────────────
    print("\n[5] Preprocessing complete.")
    print("    In a full pipeline, image patch features would be extracted")
    print("    at each spot location and used as model input alongside the")
    print("    spatial graph and normalized expression values.")
    print()


def main():
    args = parse_args()
    cfg  = load_config(args.config)
    run_demo(cfg)


if __name__ == "__main__":
    main()
