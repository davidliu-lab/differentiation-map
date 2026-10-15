"""predict / evaluate: per-tile cluster predictions, slide-level tables and evaluation on labelled cohorts."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from . import progress
from .bundle import ModelBundle
from .cohort import Slide, list_slides, sample_tiles, with_files
from .config import Cohort, Config, ConfigError
from .stats import associations, binary_metrics, cluster_descriptors, high_low_presence, presence_matrix
from .storage import TileInfo, read_arrays, read_features, read_features_attrs, read_tile_info, write_arrays
from .tumor import PROMPTS_VERSION, load_text_features, tumor_mask


def prediction_path(out_dir: Path, slide_id: str) -> Path:
    return out_dir / "predictions" / f"{slide_id}.h5"


def _check_contract(cohort: Cohort, slide: Slide, info: TileInfo, bundle: ModelBundle, warned: set) -> None:
    """Warn (once per issue) when tiles/features differ from the model's input contract (D23)."""
    tile_mm, expected = info.attrs.get("tile_mm"), bundle.meta["input"]["tile_mm"]
    if tile_mm is not None and abs(float(tile_mm) - expected) > 0.05 * expected and "tile_mm" not in warned:
        progress.log.warning(f"Tiles of {slide.slide_id} are {float(tile_mm):.4f} mm; the model expects {expected} mm.")
        warned.add("tile_mm")
    for backbone, spec in ((bundle.backbone, bundle.meta["backbone"]), (bundle.tumor_backbone, bundle.meta["tumor_filter"])):
        revision = read_features_attrs(cohort.features_path(backbone, slide.slide_id)).get("hf_revision")
        if revision and revision != spec["hf_revision"] and backbone not in warned:
            progress.log.warning(f"{backbone} features of {slide.slide_id} come from revision {revision[:8]}, "
                                 f"the model expects {spec['hf_revision'][:8]}.")
            warned.add(backbone)


def predict(cfg: Config, cohort: Cohort, bundle: ModelBundle, out_dir: Path, fail_fast: bool = False) -> int:
    """Predict every foreground tile of every slide (resumable, one file per slide), then write the tables."""
    if bundle.meta["tumor_filter"]["prompts_version"] != PROMPTS_VERSION:
        raise ConfigError("The model was trained with different tumor-filter prompts than this version of rdmap.")
    slides, _ = list_slides(cohort, bundle.label["cutoff"])
    slides = with_files(cohort, slides, [bundle.backbone, bundle.tumor_backbone])
    todo = [s for s in slides if not prediction_path(out_dir, s.slide_id).is_file()]
    summary = progress.StageSummary("predict", skipped=len(slides) - len(todo))
    if len(todo) < len(slides):
        progress.info(f"{len(slides) - len(todo)}/{len(slides)} slides already predicted, skipping")
    if todo:
        text_features = load_text_features(cfg)
        infos = [read_tile_info(cohort.tiles_path(s.slide_id)) for s in todo]
        total = int(sum(len(info.foreground) for info in infos))
        warned: set = set()
        with progress.bar(total, "predict") as bar:
            for i, (slide, info) in enumerate(zip(todo, infos)):
                bar.set_postfix_str(f"slide {i + 1}/{len(todo)} · {slide.slide_id}")
                try:
                    _check_contract(cohort, slide, info, bundle, warned)
                    _predict_slide(cohort, slide, info, bundle, text_features, out_dir)
                    summary.processed += 1
                    summary.tiles += len(info.foreground)
                except Exception as e:  # noqa: BLE001 - per-slide failures are reported, not fatal (D30.8)
                    if fail_fast:
                        raise
                    progress.log.error(f"predict failed for {slide.slide_id}: {e}")
                    summary.failed.append(slide.slide_id)
                bar.update(len(info.foreground))
    summary.report()
    write_tables(cfg, cohort, bundle, out_dir)
    return 1 if summary.failed else 0


def _predict_slide(cohort: Cohort, slide: Slide, info: TileInfo, bundle: ModelBundle, text_features, out_dir: Path):
    foreground = info.foreground
    conch = read_features(cohort.features_path(bundle.tumor_backbone, slide.slide_id), foreground)
    tumor = tumor_mask(conch, text_features)
    features = read_features(cohort.features_path(bundle.backbone, slide.slide_id), foreground[tumor])
    clusters, distances = bundle.predict_features(features)
    cluster_all = np.full(len(foreground), -1, dtype=np.int64)
    cluster_all[tumor] = clusters
    distance_all = np.full(len(foreground), np.nan)
    distance_all[tumor] = distances
    write_arrays(
        prediction_path(out_dir, slide.slide_id),
        {"tile_index": foreground, "tumor": tumor, "cluster": cluster_all, "distance": distance_all},
        {"slide_id": slide.slide_id, "model": bundle.name, "n_clusters": bundle.n_clusters},
    )


@dataclass
class SlidePredictions:
    slide: Slide
    info: TileInfo
    tile_index: np.ndarray  # foreground tile indices
    tumor: np.ndarray
    cluster: np.ndarray  # -1 for non-tumor
    distance: np.ndarray
    sampled: np.ndarray  # foreground tile indices in sampled order


class CohortPredictions:
    """Predictions of a cohort plus the (recomputed) per-slide tile sample, in slide order."""

    def __init__(self, cfg: Config, cohort: Cohort, bundle: ModelBundle, out_dir: Path):
        self.cohort, self.bundle, self.out_dir = cohort, bundle, out_dir
        slides, self.label_kind = list_slides(cohort, bundle.label["cutoff"])
        slides = with_files(cohort, slides, [bundle.backbone, bundle.tumor_backbone])
        missing = [s.slide_id for s in slides if not prediction_path(out_dir, s.slide_id).is_file()]
        if missing:
            raise ConfigError(f"No predictions for {len(missing)} slides (run `rdmap predict` first): {missing[:5]}")
        infos = [read_tile_info(cohort.tiles_path(s.slide_id)) for s in slides]
        sampling = cfg.section("sampling")
        sampled = sample_tiles([i.background for i in infos], int(sampling["bagsize"]), int(sampling["seed"]))
        self.slides: List[SlidePredictions] = []
        for slide, info, idx in zip(slides, infos, sampled):
            arrays = read_arrays(prediction_path(out_dir, slide.slide_id))
            self.slides.append(SlidePredictions(
                slide, info, arrays["tile_index"], arrays["tumor"].astype(bool), arrays["cluster"],
                arrays["distance"], idx,
            ))

    @property
    def has_labels(self) -> bool:
        return self.label_kind != "none" and any(sp.slide.label is not None for sp in self.slides)

    def sampled_tumor(self) -> Dict[str, np.ndarray]:
        """Sampled tumor tiles of all slides, in slide order and sampled order (the notebook's tile order)."""
        parts: Dict[str, list] = {k: [] for k in ("slide_id", "tile_index", "cluster", "distance", "label", "value")}
        for sp in self.slides:
            position = {int(t): j for j, t in enumerate(sp.tile_index)}
            rows = np.array([position[int(t)] for t in sp.sampled], dtype=np.int64)
            rows = rows[sp.tumor[rows]] if len(rows) else rows
            n = len(rows)
            parts["slide_id"].append(np.full(n, sp.slide.slide_id, dtype=object))
            parts["tile_index"].append(sp.tile_index[rows])
            parts["cluster"].append(sp.cluster[rows])
            parts["distance"].append(sp.distance[rows])
            parts["label"].append(np.full(n, sp.slide.label, dtype=object))
            value = np.nan if sp.slide.label_value is None else sp.slide.label_value
            parts["value"].append(np.full(n, value, dtype=float))
        return {k: np.concatenate(v) if v else np.zeros(0) for k, v in parts.items()}

    def n_sampled_slides(self) -> int:
        return sum(1 for sp in self.slides if len(sp.sampled) > 0)

    def presence(self) -> np.ndarray:
        tiles = self.sampled_tumor()
        return presence_matrix(tiles["slide_id"], tiles["cluster"].astype(np.int64),
                               [sp.slide.slide_id for sp in self.slides], self.bundle.n_clusters)

    def descriptors(self) -> Optional[dict]:
        """Cluster descriptors recomputed on this cohort (label-dependent ones only if the cohort has labels)."""
        tiles = self.sampled_tumor()
        labels = tiles["label"] if self.has_labels else None
        values = tiles["value"] if self.has_labels else None
        return cluster_descriptors(tiles["cluster"].astype(np.int64), tiles["slide_id"], self.n_sampled_slides(),
                                   self.bundle.n_clusters, labels, values)


def label_column_name(bundle: ModelBundle) -> str:
    return f"{bundle.label['name']}_high"


def write_tables(cfg: Config, cohort: Cohort, bundle: ModelBundle, out_dir: Path) -> None:
    preds = CohortPredictions(cfg, cohort, bundle, out_dir)
    presence = preds.presence()
    k = bundle.n_clusters
    ids = [sp.slide.slide_id for sp in preds.slides]
    labels = [sp.slide.label for sp in preds.slides]

    presences = pd.DataFrame(presence, columns=[f"cluster_{i}" for i in range(k)])
    presences.insert(0, "slide_id", ids)
    if preds.has_labels:
        presences[label_column_name(bundle)] = labels
    presences.to_csv(out_dir / "cluster_presences.csv", index=False)

    hl = high_low_presence(presence, bundle.associations)
    table = pd.DataFrame({
        "slide_id": ids,
        "mpp": [sp.info.attrs.get("mpp", np.nan) for sp in preds.slides],
        "n_foreground": [len(sp.tile_index) for sp in preds.slides],
        "n_tumor": [int(sp.tumor.sum()) for sp in preds.slides],
        "n_sampled_tumor": [int(np.isin(sp.sampled, sp.tile_index[sp.tumor]).sum()) for sp in preds.slides],
    })
    for i in range(k):
        table[f"presence_{i}"] = presence[:, i]
    table["presence_high"], table["presence_low"] = hl[:, 0], hl[:, 1]
    proba = bundle.slide_proba(presence)
    if proba is not None:
        threshold = bundle.meta["slide_model"]["threshold"]
        table["rd_high_proba"] = proba
        table["rd_high"] = np.where(np.isnan(proba), "", np.where(proba >= threshold, "Yes", "No"))
    if preds.has_labels:
        table[label_column_name(bundle)] = labels
    table.to_csv(out_dir / "slides.csv", index=False)
    progress.info(f"Wrote {out_dir / 'slides.csv'} and {out_dir / 'cluster_presences.csv'}")


def evaluate(cfg: Config, cohort: Cohort, bundle: ModelBundle, out_dir: Path) -> dict:
    preds = CohortPredictions(cfg, cohort, bundle, out_dir)
    if not preds.has_labels:
        raise ConfigError(f"Cohort '{cohort.name}' has no labels (set metadata and label_column) to evaluate.")
    result: dict = {"model": bundle.name, "cohort": cohort.name}
    presence = preds.presence()
    y = np.array([1 if sp.slide.label == "Yes" else 0 if sp.slide.label == "No" else -1 for sp in preds.slides])
    proba = bundle.slide_proba(presence)
    if proba is not None:
        ok = (y >= 0) & ~np.isnan(proba)
        result["slide_model"] = binary_metrics(y[ok], proba[ok], bundle.meta["slide_model"]["threshold"])
    desc = preds.descriptors()
    alpha = bundle.meta["rd_association"]["alpha"]
    cohort_assoc = associations(desc["p_value"], desc["mean_presence_high"], desc["mean_presence_low"], alpha)
    result["clusters"] = [
        {
            "cluster": i,
            "model_p_value": None if not np.isfinite(bundle.p_values[i]) else float(bundle.p_values[i]),
            "model_rd_association": bundle.associations[i],
            "cohort_p_value": None if not np.isfinite(desc["p_value"][i]) else float(desc["p_value"][i]),
            "cohort_rd_association": cohort_assoc[i],
            "cohort_mean_presence_low": float(desc["mean_presence_low"][i]),
            "cohort_mean_presence_high": float(desc["mean_presence_high"][i]),
        }
        for i in range(bundle.n_clusters)
    ]
    (out_dir / "evaluation.json").write_text(json.dumps(result, indent=2) + "\n")

    if "slide_model" in result:
        m = result["slide_model"]
        print(f"Slide model on {m['n']} slides: AUC {m['auc']:.3f}, accuracy {m['accuracy']:.3f}, "
              f"sensitivity {m['sensitivity']:.3f}, specificity {m['specificity']:.3f} (threshold {m['threshold']})")
        print(f"Confusion: {m['confusion']}")
    print(f"{'cluster':>7}  {'model p':>9} {'model assoc':>12}  {'cohort p':>9} {'cohort assoc':>12}")
    for c in result["clusters"]:
        mp = "-" if c["model_p_value"] is None else f"{c['model_p_value']:.2e}"
        cp = "-" if c["cohort_p_value"] is None else f"{c['cohort_p_value']:.2e}"
        print(f"{c['cluster']:>7}  {mp:>9} {c['model_rd_association']:>12}  {cp:>9} {c['cohort_rd_association']:>12}")
    print(f"Wrote {out_dir / 'evaluation.json'}")
    return result
