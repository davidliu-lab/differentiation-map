"""Cluster statistics (pure functions), faithful to the original analysis notebook."""

from __future__ import annotations

from typing import Dict, List, Optional, Sequence

import numpy as np
from scipy.stats import mannwhitneyu

HIGH, LOW, NONE = "high", "low", "none"


def presence_matrix(tile_slide_ids: np.ndarray, clusters: np.ndarray, slide_ids: Sequence[str], n_clusters: int) -> np.ndarray:
    """Fraction of each slide's (tumor) tiles in each cluster; NaN for slides without tiles. Shape (S, K)."""
    result = np.full((len(slide_ids), n_clusters), np.nan)
    for i, slide_id in enumerate(slide_ids):
        mask = tile_slide_ids == slide_id
        total = mask.sum()
        if total > 0:
            result[i] = np.bincount(clusters[mask], minlength=n_clusters)[:n_clusters] / total
    return result


def cluster_descriptors(
    clusters: np.ndarray,
    tile_slide_ids: np.ndarray,
    n_slides: int,
    n_clusters: int,
    tile_labels: Optional[np.ndarray] = None,
    tile_label_values: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Per-cluster descriptors over (sampled, tumor) tiles.

    `n_slides` is the number of slides that contributed tiles before the tumor filter. Label-dependent
    descriptors are NaN when `tile_labels` (binarised "Yes"/"No" per tile) is not given.
    """
    k = np.arange(n_clusters)
    n = len(clusters)
    result: Dict[str, object] = {
        "size": np.array([np.sum(clusters == i) for i in k]),
        "size_index": np.array([np.sum(clusters == i) / (n / float(n_clusters)) if n else np.nan for i in k]),
        "span": np.array([len(np.unique(tile_slide_ids[clusters == i])) / n_slides for i in k]),
    }
    nan = np.full(n_clusters, np.nan)
    labelwise: Dict[str, List[np.ndarray]] = {}
    if tile_labels is None:
        result.update(
            high_tile_share=nan, median_label=nan, mean_presence_low=nan, mean_presence_high=nan,
            presence_ratio=nan, mwu_statistic=nan, p_value=nan,
        )
        result["labelwise_presence"] = labelwise
        return result

    with np.errstate(invalid="ignore", divide="ignore"):
        result["high_tile_share"] = np.array([np.mean(tile_labels[clusters == i] == "Yes") for i in k])
        if tile_label_values is not None:
            result["median_label"] = np.array(
                [np.median(tile_label_values[clusters == i]) if np.any(clusters == i) else np.nan for i in k]
            )
        else:
            result["median_label"] = nan
        for label in ("No", "Yes"):
            slides = np.unique(tile_slide_ids[tile_labels == label])
            labelwise[label] = [
                np.array([np.sum((tile_slide_ids == s) & (clusters == i)) / np.sum(tile_slide_ids == s) for s in slides])
                for i in k
            ]
        mean_low = np.array([np.mean(labelwise["No"][i]) if len(labelwise["No"][i]) else np.nan for i in k])
        mean_high = np.array([np.mean(labelwise["Yes"][i]) if len(labelwise["Yes"][i]) else np.nan for i in k])
        result["mean_presence_low"] = mean_low
        result["mean_presence_high"] = mean_high
        result["presence_ratio"] = mean_high / mean_low

    statistic, p_value = np.full(n_clusters, np.nan), np.full(n_clusters, np.nan)
    for i in k:
        if len(labelwise["No"][i]) and len(labelwise["Yes"][i]):
            test = mannwhitneyu(labelwise["No"][i], labelwise["Yes"][i])
            statistic[i], p_value[i] = test.statistic, test.pvalue
    result["mwu_statistic"] = statistic
    result["p_value"] = p_value
    result["labelwise_presence"] = labelwise
    return result


def associations(p_value: np.ndarray, mean_high: np.ndarray, mean_low: np.ndarray, alpha: float) -> List[str]:
    """`high`/`low` for clusters significantly over-represented in high/low slides, else `none`."""
    result = []
    for p, hi, lo in zip(p_value, mean_high, mean_low):
        if np.isfinite(p) and p < alpha:
            result.append(HIGH if hi > lo else LOW)
        else:
            result.append(NONE)
    return result


def high_low_presence(presence: np.ndarray, association: Sequence[str]) -> np.ndarray:
    """Summed presence of high- and low-associated clusters per slide. Shape (S, 2): [h, l]."""
    association = np.asarray(association)
    h = presence[:, association == HIGH].sum(axis=1) if np.any(association == HIGH) else np.zeros(len(presence))
    l = presence[:, association == LOW].sum(axis=1) if np.any(association == LOW) else np.zeros(len(presence))
    return np.stack([h, l], axis=1)


def binary_metrics(y_true: np.ndarray, proba: np.ndarray, threshold: float) -> Dict[str, float]:
    from sklearn.metrics import roc_auc_score

    y_true = np.asarray(y_true, dtype=int)
    y_pred = (np.asarray(proba) >= threshold).astype(int)
    tp = int(np.sum((y_pred == 1) & (y_true == 1)))
    tn = int(np.sum((y_pred == 0) & (y_true == 0)))
    fp = int(np.sum((y_pred == 1) & (y_true == 0)))
    fn = int(np.sum((y_pred == 0) & (y_true == 1)))
    auc = float(roc_auc_score(y_true, proba)) if len(np.unique(y_true)) == 2 else float("nan")
    return {
        "n": int(len(y_true)),
        "auc": auc,
        "accuracy": (tp + tn) / len(y_true) if len(y_true) else float("nan"),
        "sensitivity": tp / (tp + fn) if tp + fn else float("nan"),
        "specificity": tn / (tn + fp) if tn + fp else float("nan"),
        "confusion": {"tp": tp, "fp": fp, "tn": tn, "fn": fn},
        "threshold": threshold,
    }
