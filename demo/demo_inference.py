"""
Demo: Evaluate saved predictions using the metrics suite.

Usage:
    python demo/demo_inference.py --config configs/example_config.yaml

Optional override:
    python demo/demo_inference.py --pred_dir /path/to/predictions

This script does not require GPU or model weights. It only loads example
prediction files and computes evaluation metrics.
"""

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.metrics import evaluate_all, print_results


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate demo spatial gene expression predictions"
    )
    parser.add_argument(
        "--config",
        default="configs/example_config.yaml",
        help="Path to config YAML file",
    )
    parser.add_argument(
        "--pred_dir",
        default=None,
        help="Directory containing pred.npy, target.npy, and optional genes.txt",
    )
    return parser.parse_args()


def load_config(path: str) -> dict:
    """Load YAML config if it exists; otherwise return an empty dict."""
    config_path = Path(path)
    if not config_path.exists():
        return {}

    with open(config_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def load_demo_predictions(pred_dir: str):
    """
    Load prediction files from a directory.

    Expected files:
        pred.npy      predicted expression, shape (N, G)
        target.npy    ground-truth expression, shape (N, G)
        genes.txt     optional gene names, one gene per line
    """
    pred_dir = Path(pred_dir)

    pred   = np.load(pred_dir / "pred.npy").astype(np.float32)
    target = np.load(pred_dir / "target.npy").astype(np.float32)

    gene_path = pred_dir / "genes.txt"
    if gene_path.exists():
        gene_names = [
            line.strip()
            for line in gene_path.read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]
    else:
        gene_names = [f"Gene_{i}" for i in range(pred.shape[1])]

    return pred, target, gene_names


def main():
    args = parse_args()
    cfg  = load_config(args.config)

    pred_dir = args.pred_dir or cfg.get("prediction_dir", "example_predictions")

    print(f"Config:   {args.config}")
    print(f"Pred dir: {pred_dir}\n")

    try:
        pred, target, gene_names = load_demo_predictions(pred_dir)
    except FileNotFoundError:
        print("[Note] Prediction files were not found.")
        print("Expected files:")
        print("  pred.npy")
        print("  target.npy")
        print("  genes.txt  (optional)")
        print("\nPlease provide --pred_dir or update the config file.")
        return

    print(f"Pred shape:   {pred.shape}")
    print(f"Target shape: {target.shape}")
    print(f"Genes:        {len(gene_names)}\n")

    results = evaluate_all(
        pred=pred,
        target=target,
        gene_names=gene_names,
    )

    print_results(results, model_name="Demo model")


if __name__ == "__main__":
    main()
