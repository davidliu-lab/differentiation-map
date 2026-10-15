"""fit(): train the PLS alignment, k-means clusters, cluster associations and the slide model (path B)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd

from . import __version__, progress
from .bundle import SLIDE_INPUTS, ModelBundle, now_iso, poly2_terms
from .cohort import list_slides, sample_tiles, with_files
from .config import Cohort, Config, ConfigError
from .embedding import backbone_info
from .stats import associations, cluster_descriptors, high_low_presence, presence_matrix
from .storage import read_features, read_tile_info
from .tumor import PROMPTS_VERSION, load_text_features, tumor_mask

DESCRIPTOR_KEYS = [
    "size", "size_index", "span", "high_tile_share", "median_label", "mean_presence_low",
    "mean_presence_high", "presence_ratio", "mwu_statistic", "p_value",
]


def _jsonable(values: np.ndarray) -> List[Any]:
    return [None if not np.isfinite(v) else float(v) for v in np.asarray(values, dtype=float)]


def load_sampled_tiles(cfg: Config, cohort: Cohort, slides, backbone: str, tumor_backbone: str, text_features):
    """Sample tiles per slide (shared RNG, slide order) and load their features and tumor mask."""
    sampling = cfg.section("sampling")
    infos = [read_tile_info(cohort.tiles_path(s.slide_id)) for s in slides]
    sampled = sample_tiles([info.background for info in infos], int(sampling["bagsize"]), int(sampling["seed"]))
    total = int(sum(len(idx) for idx in sampled))
    X, tumor, slide_ids, tile_index = [], [], [], []
    with progress.bar(total, "loading features") as bar:
        for i, (slide, idx) in enumerate(zip(slides, sampled)):
            bar.set_postfix_str(f"slide {i + 1}/{len(slides)} · {slide.slide_id}")
            conch = read_features(cohort.features_path(tumor_backbone, slide.slide_id), idx)
            X.append(read_features(cohort.features_path(backbone, slide.slide_id), idx))
            tumor.append(tumor_mask(conch, text_features))
            slide_ids.append(np.full(len(idx), slide.slide_id, dtype=object))
            tile_index.append(idx)
            bar.update(len(idx))
    return (
        np.concatenate(X), np.concatenate(tumor), np.concatenate(slide_ids), np.concatenate(tile_index), sampled, infos
    )


def fit(cfg: Config, cohort: Cohort, out_dir: Path) -> ModelBundle:
    from sklearn.cluster import KMeans
    from sklearn.cross_decomposition import PLSRegression
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import PolynomialFeatures, StandardScaler, normalize

    label_cfg, model_cfg = cfg.section("label"), cfg.section("model")
    cutoff = float(label_cfg["cutoff"])
    backbone, tumor_backbone = cfg.get("backbones", "features"), cfg.get("backbones", "tumor_filter")
    slides, kind = list_slides(cohort, cutoff, require_labels=True)
    slides = with_files(cohort, slides, [backbone, tumor_backbone])
    if len({s.label for s in slides}) < 2:
        raise ConfigError("fit needs slides of both label groups (high and low).")
    progress.info(f"Training on {len(slides)} slides of cohort '{cohort.name}' ({kind} label '{cohort.label_column}').")

    text_features = load_text_features(cfg)
    X, tumor, tile_slide_ids, tile_index, sampled, infos = load_sampled_tiles(
        cfg, cohort, slides, backbone, tumor_backbone, text_features
    )
    by_id = {s.slide_id: s for s in slides}
    labels = np.array([by_id[s].label for s in tile_slide_ids])
    values = np.array([by_id[s].label_value for s in tile_slide_ids], dtype=float)
    progress.info(f"{len(X):,} sampled tiles, {int(tumor.sum()):,} classified as tumor.")

    Xt = normalize(X) if model_cfg["l2_pre"] else X
    Xt = Xt[tumor]
    with progress.step(f"Fitting PLS ({model_cfg['n_components']} components, {len(Xt):,} tiles)"):
        # Estimator settings are spelled out (not left to defaults) so results do not depend on the sklearn version.
        pls = PLSRegression(n_components=int(model_cfg["n_components"]), scale=True, max_iter=500, tol=1e-06)
        pls.fit(Xt, values[tumor])
    Z = pls.transform(Xt)
    if model_cfg["l2_post"]:
        Z = normalize(Z)
    n_clusters = int(model_cfg["n_clusters"])
    with progress.step(f"Fitting KMeans (k={n_clusters}, {len(Z):,} tiles)"):
        kmeans = KMeans(n_clusters=n_clusters, init="k-means++", n_init=1, max_iter=300, tol=1e-4, algorithm="lloyd",
                        random_state=int(model_cfg["seed"])).fit(Z)
    clusters = kmeans.predict(Z)
    distances = np.min(kmeans.transform(Z), axis=1)

    # Cluster descriptors and high/low associations.
    t_slides, t_labels, t_values = tile_slide_ids[tumor], labels[tumor], values[tumor]
    n_slides = len(np.unique(tile_slide_ids))
    desc = cluster_descriptors(clusters, t_slides, n_slides, n_clusters, t_labels, t_values)
    alpha = float(cfg.get("rd_association", "alpha"))
    assoc = associations(desc["p_value"], desc["mean_presence_high"], desc["mean_presence_low"], alpha)
    slide_ids = [s.slide_id for s in slides]
    presence = presence_matrix(t_slides, clusters, slide_ids, n_clusters)

    # Slide model: logistic regression on degree-2 polynomial features of [h, l].
    arrays: Dict[str, np.ndarray] = {
        "x_mean": pls._x_mean, "x_std": pls._x_std, "x_rotations": pls.x_rotations_, "centroids": kmeans.cluster_centers_,
    }
    slide_cfg = cfg.section("slide_model")
    inputs = [name for name, cls in (("presence_high", "high"), ("presence_low", "low")) if cls in assoc]
    slide_model = None
    valid = ~np.isnan(presence).any(axis=1)
    y_slide = np.array([1 if s.label == "Yes" else 0 for s in slides])
    if inputs:
        Xs = high_low_presence(presence[valid], assoc)[:, [SLIDE_INPUTS[n] for n in inputs]]
        poly = PolynomialFeatures(degree=2, include_bias=False)
        scaler = StandardScaler()
        Fs = scaler.fit_transform(poly.fit_transform(Xs))
        logreg = LogisticRegression(C=float(slide_cfg["C"]), solver="lbfgs", max_iter=100).fit(Fs, y_slide[valid])
        arrays.update(
            slide_coef=np.concatenate([logreg.intercept_, logreg.coef_[0]]),
            slide_feat_mean=scaler.mean_,
            slide_feat_std=scaler.scale_,
        )
        slide_model = {
            "type": "logistic_poly2", "inputs": inputs, "terms": poly2_terms(inputs),
            "C": float(slide_cfg["C"]), "threshold": float(slide_cfg["threshold"]),
        }
    else:
        progress.log.warning("No cluster is significantly associated with high or low labels: slide model disabled.")

    uni, conch = backbone_info(backbone), backbone_info(tumor_backbone)
    tiling = cohort.tiling
    meta = {
        "name": out_dir.name,
        "created": now_iso(),
        "package_version": __version__,
        "backbone": {"name": backbone, "hf_repo": uni["hf_repo"], "hf_revision": uni["hf_revision"], "dim": uni["dim"]},
        "tumor_filter": {
            "backbone": tumor_backbone, "hf_repo": conch["hf_repo"], "hf_revision": conch["hf_revision"],
            "prompts_version": PROMPTS_VERSION,
        },
        "input": {"tile_mm": float(tiling["tile_mm"]), "output_px": int(tiling["output_px"])},
        "preprocessing": {
            "l2_pre": bool(model_cfg["l2_pre"]), "l2_post": bool(model_cfg["l2_post"]),
            "n_components": int(model_cfg["n_components"]),
        },
        "clusters": {
            "n_clusters": n_clusters,
            "descriptors": {**{k: _jsonable(desc[k]) for k in DESCRIPTOR_KEYS}, "rd_association": assoc},
        },
        "rd_association": {"test": "mannwhitneyu", "alpha": alpha},
        "slide_model": slide_model,
        "validation": None,
        "training_summary": {
            "n_slides": len(slides),
            "n_slides_high": int(sum(s.label == "Yes" for s in slides)),
            "n_slides_low": int(sum(s.label == "No" for s in slides)),
            "n_sampled_tiles": int(len(X)),
            "n_tumor_tiles": int(tumor.sum()),
            "bagsize": int(cfg.get("sampling", "bagsize")),
            "label_kind": kind,
        },
        "label": {
            "name": label_cfg["name"], "display": label_cfg["display"], "cutoff": cutoff, "positive": "Yes",
        },
    }
    bundle = ModelBundle(meta, {k: np.asarray(v, dtype=np.float64) for k, v in arrays.items()})
    _self_check(bundle, X[tumor], Z, clusters, kmeans, logreg if slide_model else None,
                Fs if slide_model else None, presence[valid])

    bundle.save(out_dir / "model")
    _write_training_stats(out_dir / "training", tile_slide_ids, tile_index, tumor, clusters, distances, labels,
                          slides, presence, desc, assoc)
    for i in range(n_clusters):
        p = desc["p_value"][i]
        progress.info(f"Cluster {i}: size {desc['size'][i]:>6}, p = {p:.2e}, {assoc[i]}")
    progress.info(f"Model written to {out_dir / 'model'}")
    return bundle


def _self_check(bundle: ModelBundle, X_tumor, Z, clusters, kmeans, logreg, Fs, presence_valid) -> None:
    """The numpy inference path must reproduce sklearn on the training tiles (D20)."""
    Z_np = bundle.transform(X_tumor)
    if not np.allclose(Z_np, Z, atol=1e-5):
        raise RuntimeError(f"Self-check failed: PLS projection differs (max {np.abs(Z_np - Z).max():.2e}).")
    clusters_np, _ = bundle.predict(Z)
    mismatches = int(np.sum(clusters_np != clusters))
    if mismatches > 0.001 * len(clusters):
        raise RuntimeError(f"Self-check failed: {mismatches} cluster assignments differ from sklearn.")
    if logreg is not None:
        proba_np = bundle.slide_proba(presence_valid)
        proba_sk = logreg.predict_proba(Fs)[:, 1]
        if not np.allclose(proba_np, proba_sk, atol=1e-9):
            raise RuntimeError("Self-check failed: slide model probabilities differ from sklearn.")
    progress.info(f"Self-check passed (numpy inference matches sklearn; {mismatches} borderline tiles differ).")


def _write_training_stats(path: Path, tile_slide_ids, tile_index, tumor, clusters, distances, labels, slides,
                          presence, desc, assoc) -> None:
    """Per-tile and per-slide training statistics. Private: never part of a released model (D21)."""
    path.mkdir(parents=True, exist_ok=True)
    cluster_all = np.full(len(tumor), -1)
    cluster_all[tumor] = clusters
    distance_all = np.full(len(tumor), np.nan)
    distance_all[tumor] = distances
    pd.DataFrame({
        "slide_id": tile_slide_ids, "tile_index": tile_index, "tumor": tumor, "cluster": cluster_all,
        "distance": distance_all, "label": labels,
    }).to_csv(path / "train_tiles.csv", index=False)
    slide_df = pd.DataFrame(presence, columns=[f"presence_{k}" for k in range(presence.shape[1])])
    slide_df.insert(0, "label", [s.label for s in slides])
    slide_df.insert(0, "slide_id", [s.slide_id for s in slides])
    slide_df.to_csv(path / "slide_presence.csv", index=False)
    stats = pd.DataFrame({k: desc[k] for k in DESCRIPTOR_KEYS})
    stats.insert(0, "cluster", np.arange(len(stats)))
    stats["rd_association"] = assoc
    stats.to_csv(path / "cluster_stats.csv", index=False)
