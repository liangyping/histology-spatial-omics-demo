"""
Evaluation metrics for spatial gene expression prediction.

Supports:
- Per-gene Pearson Correlation Coefficient (PCC)
- Spearman correlation
- RMSE
- Subset metrics: HEG / HVG / Marker gene PCC
"""

import json
import numpy as np
from scipy.stats import pearsonr, spearmanr
from typing import Optional


def per_gene_pcc(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Compute mean per-gene Pearson Correlation Coefficient.

    Args:
        pred:   (N, G) predicted gene expression
        target: (N, G) ground-truth gene expression

    Returns:
        Mean PCC across genes (skips constant genes).
    """
    assert pred.shape == target.shape, "pred and target must have the same shape"
    pccs = []
    for g in range(pred.shape[1]):
        p, t = pred[:, g], target[:, g]
        if np.std(t) < 1e-8 or np.std(p) < 1e-8:
            continue
        pccs.append(pearsonr(p, t)[0])
    return float(np.mean(pccs)) if pccs else float("nan")


def per_gene_spearman(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean per-gene Spearman correlation."""
    spearmans = []
    for g in range(pred.shape[1]):
        p, t = pred[:, g], target[:, g]
        if np.std(t) < 1e-8 or np.std(p) < 1e-8:
            continue
        spearmans.append(spearmanr(p, t)[0])
    return float(np.mean(spearmans)) if spearmans else float("nan")


def per_gene_rmse(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean per-gene RMSE."""
    rmses = [
        np.sqrt(np.mean((pred[:, g] - target[:, g]) ** 2))
        for g in range(pred.shape[1])
    ]
    return float(np.mean(rmses))


def subset_pcc(
    pred: np.ndarray,
    target: np.ndarray,
    gene_names: list,
    subset_genes: list,
) -> tuple[float, int]:
    """
    Compute mean PCC restricted to a subset of genes (e.g. HEG / HVG / Marker).

    Args:
        pred, target:   (N, G) arrays
        gene_names:     list of gene names corresponding to columns
        subset_genes:   gene names to restrict evaluation to

    Returns:
        (mean_pcc, n_genes_found)
    """
    vals = []
    for gname in subset_genes:
        if gname not in gene_names:
            continue
        g = gene_names.index(gname)
        p, t = pred[:, g], target[:, g]
        if np.std(t) < 1e-8 or np.std(p) < 1e-8:
            continue
        vals.append(pearsonr(p, t)[0])
    mean = float(np.mean(vals)) if vals else float("nan")
    return mean, len(vals)


def evaluate_all(
    pred: np.ndarray,
    target: np.ndarray,
    gene_names: list,
    heg_genes: Optional[list] = None,
    hvg_genes: Optional[list] = None,
    marker_genes: Optional[list] = None,
) -> dict:
    """
    Full evaluation suite.

    Args:
        pred, target:   (N, G) arrays
        gene_names:     list of gene names (length G)
        heg_genes:      highly expressed gene names for HEG metric
        hvg_genes:      highly variable gene names for HVG metric
        marker_genes:   cancer-type marker gene names

    Returns:
        dict with keys: PCC, Spearman, RMSE, HEG, HVG, Marker
    """
    results = {
        "PCC":      per_gene_pcc(pred, target),
        "Spearman": per_gene_spearman(pred, target),
        "RMSE":     per_gene_rmse(pred, target),
    }

    for key, gene_list in [("HEG", heg_genes), ("HVG", hvg_genes), ("Marker", marker_genes)]:
        if gene_list is not None:
            val, n = subset_pcc(pred, target, gene_names, gene_list)
            results[key] = val
            results[f"_{key.lower()}_n"] = n
        else:
            results[key] = float("nan")

    return results


def load_gene_list(json_path: str) -> list:
    """Load gene list from a JSON file with format: {"genes": [...]}"""
    with open(json_path) as f:
        return json.load(f)["genes"]


def print_results(results: dict, model_name: str = "Model") -> None:
    """Pretty-print evaluation results."""
    print(f"\n{'-'*50}")
    print(f"  {model_name}")
    print(f"{'-'*50}")
    for key in ["PCC", "Spearman", "RMSE", "HEG", "HVG", "Marker"]:
        val = results.get(key, float("nan"))
        n   = results.get(f"_{key.lower()}_n", "")
        n_str = f"  (n={n})" if n != "" else ""
        print(f"  {key:<10}: {val:.4f}{n_str}")
