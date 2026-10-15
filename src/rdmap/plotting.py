"""Figures: cluster summary with tile mosaics, per-cluster tile samples and per-slide cluster maps."""

from __future__ import annotations

from typing import List, Optional, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import ListedColormap  # noqa: E402
from matplotlib.figure import Figure  # noqa: E402

from .export import cluster_colors  # noqa: E402
from .predict import CohortPredictions, SlidePredictions  # noqa: E402
from .storage import read_thumbnail, read_tile_images  # noqa: E402


def load_tiles(preds: CohortPredictions, slide_ids: np.ndarray, tile_indices: np.ndarray) -> np.ndarray:
    """Tile images for (slide_id, tile_index) pairs, in the given order (reads only these tiles)."""
    out: Optional[np.ndarray] = None
    for slide_id in dict.fromkeys(slide_ids):
        rows = np.flatnonzero(slide_ids == slide_id)
        images = read_tile_images(preds.cohort.tiles_path(slide_id), tile_indices[rows])
        if out is None:
            out = np.zeros((len(slide_ids), *images.shape[1:]), dtype=np.uint8)
        out[rows] = images
    return out if out is not None else np.zeros((0, 224, 224, 3), dtype=np.uint8)


def _title_case(text: str) -> str:
    return " ".join(w[:1].upper() + w[1:] for w in text.split())


def clusters_summary(preds: CohortPredictions, mosaic: int = 6, ncols: int = 4, seed: int = 1) -> Figure:
    """One panel per cluster: descriptors and a mosaic of randomly sampled tiles, sorted by presence ratio."""
    bundle = preds.bundle
    k = bundle.n_clusters
    tiles = preds.sampled_tumor()
    clusters = tiles["cluster"].astype(np.int64)
    desc = preds.descriptors()
    labelled = preds.has_labels
    ratio = desc["presence_ratio"] if labelled else bundle.descriptor("presence_ratio")
    sorting = np.argsort(ratio)
    display = _title_case(bundle.label["display"])

    nrows = min(int(np.ceil(k / ncols)), 10)
    fig, ax = plt.subplots(nrows=nrows, ncols=ncols, figsize=(ncols * 6, nrows * 10))
    ax = np.atleast_1d(ax).flatten()
    for a in ax:
        a.axis("off")
    random = np.random.RandomState(seed)
    for i in range(k):
        if i >= ncols * nrows:
            break
        c = sorting[i]
        lines = [
            f"Cluster ID = {c}",
            f"Size Index = {desc['size_index'][c]:.2f}",
            f"Cluster Span = {desc['span'][c] * 100:.2f}%",
        ]
        if labelled:
            lines += [
                f"High {display} Tiles = {desc['high_tile_share'][c] * 100:.2f}%",
                f"Mean Presence (Low) = {desc['mean_presence_low'][c] * 100:.2f}% ",
                f"Mean Presence (High) = {desc['mean_presence_high'][c] * 100:.2f}% ",
                f"Median {display} = {desc['median_label'][c]:.2f} ",
                f"P-Value = {desc['p_value'][c]:.3g}",
            ]
        ax[i].set_title("\n".join(lines))
        members = np.flatnonzero(clusters == c)
        idx = random.choice(len(members), min(mosaic * mosaic, len(members)), replace=False)
        selected = load_tiles(preds, tiles["slide_id"][members[idx]], tiles["tile_index"][members[idx]])
        X = np.full((mosaic * mosaic, 224, 224, 3), 255, dtype=np.uint8)
        if len(selected):
            X = np.full((mosaic * mosaic, *selected.shape[1:]), 255, dtype=np.uint8)
            X[: len(selected)] = selected
        w, h = X.shape[1], X.shape[2]
        X = X.reshape(mosaic, mosaic, *X.shape[1:]).transpose(0, 2, 1, 3, 4).reshape(mosaic * w, mosaic * h, 3)
        ax[i].imshow(X)
    fig.tight_layout()
    return fig


def tile_sample(preds: CohortPredictions, cluster: int, seed: int, n_tiles: int = 25, ncols: int = 5) -> Figure:
    """Random tiles of one cluster (seeded), ordered by distance to the cluster centroid."""
    tiles = preds.sampled_tumor()
    members = np.flatnonzero(tiles["cluster"].astype(np.int64) == cluster)
    random = np.random.RandomState(seed)
    selected = members[random.choice(len(members), min(n_tiles, len(members)), replace=False)]
    selected = selected[np.argsort(tiles["distance"][selected])]
    images = load_tiles(preds, tiles["slide_id"][selected], tiles["tile_index"][selected])

    fig, ax = plt.subplots(1, 1, figsize=(12, 12))
    ax.axis("off")
    nrows = (len(images) + ncols - 1) // ncols
    for i, image in enumerate(images):
        row, col = i // ncols, i % ncols
        ax.imshow(image, extent=(col, col + 1, row + 1, row), origin="upper")
    ax.set_xlim(0, ncols)
    ax.set_ylim(max(nrows, 1), 0)
    return fig


def slide_clustermap(
    preds: CohortPredictions,
    sp: SlidePredictions,
    clusters: Sequence[int],
    training_presence: Optional[pd.DataFrame] = None,
) -> Figure:
    """Thumbnail, overlay of the selected clusters, and the slide's cluster presences (all tumor tiles)."""
    bundle = preds.bundle
    clusters = list(clusters)
    thumbnail = read_thumbnail(preds.cohort.tiles_path(sp.slide.slide_id))
    height, width = int(sp.info.row.max()) + 1, int(sp.info.col.max()) + 1

    tumor_tiles = sp.tile_index[sp.tumor]
    slide_clustering = sp.cluster[sp.tumor]
    with np.errstate(invalid="ignore"):
        presences = np.array([np.mean(slide_clustering == i) for i in range(bundle.n_clusters)])
    selection = np.isin(slide_clustering, clusters)
    rows, cols = sp.info.row[tumor_tiles][selection], sp.info.col[tumor_tiles][selection]
    mapped = np.zeros(int(selection.sum()), dtype=int)
    for i, c in enumerate(clusters):
        mapped[slide_clustering[selection] == c] = i
    clustermap = np.zeros((height, width), dtype=int)
    clustermap[rows, cols] = mapped
    clustermask = np.zeros((height, width), dtype=float)
    clustermask[rows, cols] = 1

    fig, ax = plt.subplots(3, 1, figsize=(8, 15))
    ax[0].imshow(thumbnail)
    ax[0].axis("off")
    palette = cluster_colors(bundle.associations)
    colors = np.array([palette[c] for c in clusters])
    extent = (0, width, height, 0)
    ax[1].imshow(thumbnail, alpha=0.25, extent=extent)
    if clusters:
        ax[1].imshow(clustermap, cmap=ListedColormap(colors), alpha=clustermask, vmin=0,
                     vmax=max(len(clusters) - 1, 0), extent=extent)
    ax[1].axis("off")

    indices = np.arange(len(clusters))
    ax[2].bar(indices, presences[clusters] if clusters else [], color=colors)
    if training_presence is not None and clusters:
        style = dict(showfliers=False, patch_artist=False, widths=0.25, boxprops=dict(color="black"),
                     medianprops=dict(color="black"))
        for label, offset in (("No", -0.2), ("Yes", 0.2)):
            subset = training_presence[training_presence["label"] == label]
            data = [subset[f"presence_{c}"].dropna().to_numpy() for c in clusters]
            ax[2].boxplot(data, positions=indices + offset, **style)
    ax[2].set_xticks(indices)
    ax[2].set_xticklabels([f"Cluster {c}" for c in clusters])
    ax[2].set_ylabel("Cluster presence (% of tiles belonging to given cluster)")

    title = f"Slide ID = {sp.slide.slide_id}"
    if sp.slide.label is not None:
        call = "HIGH" if sp.slide.label == "Yes" else "LOW"
        if preds.label_kind == "continuous" and sp.slide.label_value is not None:
            title += f"; {bundle.label['display']} = {sp.slide.label_value:.3f} ({call})"
        else:
            title += f" ({call})"
    fig.suptitle(title)
    fig.tight_layout()
    return fig


def select_clusters(p_values: np.ndarray, max_p_value: float) -> List[int]:
    return [int(i) for i in np.flatnonzero(np.nan_to_num(p_values, nan=1.0) < max_p_value)]
