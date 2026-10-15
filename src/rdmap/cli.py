"""Command-line interface: `rdmap <command>`."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional

from tqdm.contrib.logging import logging_redirect_tqdm

from . import __version__, progress
from .bundle import ModelBundle, default_model_path
from .cohort import list_slides
from .config import MODELS_ROOT, Cohort, Config, ConfigError

PER_COHORT_COMMANDS = ("run", "slides", "tile", "embed", "fit", "predict", "map", "clusters", "evaluate")


# ---- argument parsing ----------------------------------------------------------------------------------------------


def _common(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("-c", "--config", action="append", default=[], metavar="FILE",
                        help="extra TOML config merged on top of configs/default.toml (repeatable)")
    parser.add_argument("--progress", choices=progress.MODES, default=None,
                        help="progress display (default: auto, or $RDMAP_PROGRESS)")
    parser.add_argument("-q", "--quiet", action="store_true", help="same as --progress off")
    parser.add_argument("--fail-fast", action="store_true", help="stop at the first failing slide")


def _cohort_args(parser: argparse.ArgumentParser) -> None:
    group = parser.add_argument_group("cohort (either --cohort from a config file, or an ad-hoc cohort)")
    group.add_argument("--cohort", metavar="NAME", help="cohort defined in a -c config file")
    group.add_argument("--wsi-dir", metavar="DIR", help="folder with whole-slide images (ad-hoc cohort)")
    group.add_argument("--glob", metavar="PAT", help="WSI file pattern(s), comma-separated (default: common formats)")
    group.add_argument("--metadata", metavar="CSV", help="slide metadata CSV (slide_id, optional label/filename/mpp)")
    group.add_argument("--label-column", metavar="COL", help="label column in the metadata CSV")
    group.add_argument("--mpp", type=float, help="fallback microns per pixel")
    group.add_argument("--name", help="ad-hoc cohort name (default: name of --wsi-dir)")


def _model_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--model", metavar="DIR", help="model folder (default: newest released model in models/)")
    parser.add_argument("--out", metavar="DIR", help="output folder (default: <output_dir>/<model name>/<cohort>)")


def _cluster_selection(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--clusters", metavar="IDS", help="comma-separated cluster IDs (overrides --max-p)")
    parser.add_argument("--max-p", type=float, metavar="P", help="select clusters with p-value < P")
    parser.add_argument("--p-values-from", choices=("model", "cohort"), default="model",
                        help="p-values for --max-p: the model's training cohort (default) or this labelled cohort")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="rdmap", description="Rambow differentiation maps from H&E whole-slide images.")
    parser.add_argument("--version", action="version", version=f"rdmap {__version__}")
    sub = parser.add_subparsers(dest="command", required=True, metavar="command")

    def add(name: str, help: str) -> argparse.ArgumentParser:
        p = sub.add_parser(name, help=help, description=help)
        _common(p)
        if name in PER_COHORT_COMMANDS:
            _cohort_args(p)
        return p

    p = add("run", "tile -> embed -> predict -> map -> clusters, skipping finished steps")
    _model_args(p)
    add("slides", "list the slide IDs of a cohort, one per line (e.g. for SLURM array jobs)")
    p = add("tile", "extract tiles from whole-slide images")
    p.add_argument("--slide-id", help="process one slide only")
    p = add("embed", "extract UNI-2 and CONCH features")
    p.add_argument("--backbone", choices=("uni2", "conch"), help="only this backbone (default: both)")
    p.add_argument("--slide-id", help="process one slide only")
    p = add("text-features", "(re)compute the CONCH tumor-filter text features")
    p.add_argument("--force", action="store_true", help="recompute even if cached")
    p = add("fit", "train a model on a labelled cohort (path B)")
    p.add_argument("--out", metavar="DIR",
                   help="run folder; the model goes to DIR/model (default: <output_dir>/<cohort>-model)")
    p = add("predict", "predict clusters for every tile and write slides.csv / cluster_presences.csv")
    _model_args(p)
    p = add("map", "map clusters onto the slides: per-tile tables, QuPath GeoJSON and slide cluster map figures")
    _model_args(p)
    p.add_argument("--formats", default="csv,geojson,png",
                   help="comma-separated: csv (tile tables), geojson (QuPath), png (slide cluster maps); default: all")
    p.add_argument("--geojson-class", choices=("rd_association", "cluster"), default="rd_association",
                   help="QuPath classification of tiles (default: rd_association)")
    _cluster_selection(p)
    p.add_argument("--training-stats", metavar="DIR",
                   help="training folder of a fit run, adds reference boxplots to the slide cluster maps")
    p.add_argument("--slide-id", help="one slide only")
    p = add("clusters", "visualize the clusters: cluster summary figure and per-cluster tile samples")
    _model_args(p)
    _cluster_selection(p)
    p.add_argument("--seeds", help="comma-separated random seeds for the tile samples (default from config)")
    p = add("evaluate", "evaluate the model on a labelled cohort")
    _model_args(p)
    p = add("release-model", "copy a trained model into models/ and write its MODEL_CARD.md")
    p.add_argument("--src", required=True, metavar="DIR", help="model folder, e.g. outputs/<run>/model")
    p.add_argument("--name", required=True, help="released model name")
    p.add_argument("--validation", metavar="JSON", help="validation metrics (JSON file) to record in the model")
    return parser


# ---- helpers -------------------------------------------------------------------------------------------------------


def resolve_cohort(cfg: Config, args: argparse.Namespace) -> Cohort:
    if args.cohort and args.wsi_dir:
        raise ConfigError("Use either --cohort or --wsi-dir, not both.")
    if args.cohort:
        return cfg.cohort(args.cohort)
    if args.wsi_dir:
        return cfg.add_adhoc_cohort(args.wsi_dir, args.name, args.glob, args.metadata, args.label_column, args.mpp)
    raise ConfigError("Specify a cohort: --cohort NAME (with -c FILE) or --wsi-dir DIR.")


def resolve_model(args: argparse.Namespace) -> ModelBundle:
    explicit = getattr(args, "model", None)
    path = Path(explicit) if explicit else default_model_path()
    bundle = ModelBundle.load(path)
    if not explicit and not getattr(args, "_model_announced", False):
        progress.info(f"Using model {bundle.name} ({path})")
        args._model_announced = True
    return bundle


def resolve_out(cfg: Config, args: argparse.Namespace, bundle: ModelBundle, cohort: Cohort) -> Path:
    out = Path(args.out) if getattr(args, "out", None) else cfg.output_dir / bundle.name / cohort.name
    out.mkdir(parents=True, exist_ok=True)
    return out


def _ints(text: Optional[str]) -> Optional[List[int]]:
    return None if not text else [int(x) for x in text.split(",") if x.strip()]


def _selected_clusters(args, preds, default_max_p: float) -> List[int]:
    from .plotting import select_clusters

    if getattr(args, "clusters", None):
        return _ints(args.clusters)
    max_p = args.max_p if getattr(args, "max_p", None) is not None else default_max_p
    if getattr(args, "p_values_from", "model") == "cohort":
        if not preds.has_labels:
            raise ConfigError("--p-values-from cohort needs a labelled cohort.")
        p_values = preds.descriptors()["p_value"]
    else:
        p_values = preds.bundle.p_values
    return select_clusters(p_values, max_p)


def _save(fig, path: Path, dpi: int) -> None:
    import matplotlib.pyplot as plt

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".partial" + path.suffix)
    fig.savefig(tmp, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    tmp.replace(path)


def _slides_for(cohort: Cohort, args) -> list:
    slides, _ = list_slides(cohort)
    slide_id = getattr(args, "slide_id", None)
    return [s for s in slides if s.slide_id == slide_id] if slide_id else slides


# ---- commands ------------------------------------------------------------------------------------------------------


class _Bars:
    """Forward progress updates to the overall bar and the current-slide bar."""

    def __init__(self, *bars):
        self.bars = bars

    def update(self, n: int = 1) -> None:
        for bar in self.bars:
            bar.update(n)


def cmd_slides(cfg: Config, cohort: Cohort, args) -> int:
    for slide in list_slides(cohort)[0]:
        print(slide.slide_id)
    return 0


def cmd_tile(cfg: Config, cohort: Cohort, args) -> int:
    from .tiling import WSI, TilingParams, grid_shape, tile_slide

    slides = _slides_for(cohort, args)
    todo = [s for s in slides if not cohort.tiles_path(s.slide_id).is_file()]
    summary = progress.StageSummary("tile", skipped=len(slides) - len(todo))
    if not todo:
        progress.info(f"tile: all {len(slides)} slides already tiled, skipping")
        return 0
    if summary.skipped:
        progress.info(f"tile: {summary.skipped}/{len(slides)} slides already tiled, skipping")

    # Plan first (cheap: WSI dimensions and resolution) so the overall bar and its ETA are counted in tiles.
    plan = []
    for slide in todo:
        try:
            if slide.wsi_path is None or not slide.wsi_path.is_file():
                raise FileNotFoundError(f"no whole-slide image found in {cohort.wsi_dir}")
            with WSI(slide.wsi_path) as wsi:
                mpp = slide.mpp or wsi.mpp() or cohort.mpp
                params = TilingParams.from_config(cohort.tiling, mpp)
                n_x, n_y = grid_shape(wsi.width, wsi.height, params.size_px, params.stride_px)
            plan.append((slide, mpp, params, n_x * n_y))
        except Exception as e:  # noqa: BLE001 - reported per slide (D30.8)
            if args.fail_fast:
                raise
            progress.log.error(f"tile failed for {slide.slide_id}: {e}")
            summary.failed.append(slide.slide_id)

    with progress.bar(sum(p[3] for p in plan), "tile") as overall:
        for i, (slide, mpp, params, n) in enumerate(plan):
            overall.set_postfix_str(f"slide {i + 1}/{len(plan)} · {slide.slide_id}")
            with progress.bar(n, "slide", inner=True) as inner:
                try:
                    with WSI(slide.wsi_path) as wsi:
                        tile_slide(wsi, cohort.tiles_path(slide.slide_id), slide.slide_id, params, mpp,
                                   _Bars(overall, inner))
                    summary.processed += 1
                    summary.tiles += n
                except Exception as e:  # noqa: BLE001
                    if args.fail_fast:
                        raise
                    progress.log.error(f"tile failed for {slide.slide_id}: {e}")
                    summary.failed.append(slide.slide_id)
    summary.report()
    return 1 if summary.failed else 0


def cmd_embed(cfg: Config, cohort: Cohort, args) -> int:
    from .embedding import embed_slide, load_backbone, resolve_device, tile_count

    backbone_arg = getattr(args, "backbone", None)
    backbones = [backbone_arg] if backbone_arg else [cfg.get("backbones", "features"),
                                                     cfg.get("backbones", "tumor_filter")]
    slides = [s for s in _slides_for(cohort, args) if cohort.tiles_path(s.slide_id).is_file()]
    device = resolve_device(cfg.get("embedding", "device", "auto"))
    batch_size = int(cfg.get("embedding", "batch_size", 20))
    status = 0
    for name in backbones:
        todo = [s for s in slides if not cohort.features_path(name, s.slide_id).is_file()]
        summary = progress.StageSummary(f"embed {name}", skipped=len(slides) - len(todo))
        if not todo:
            progress.info(f"embed {name}: all {len(slides)} tiled slides already embedded, skipping")
            continue
        if summary.skipped:
            progress.info(f"embed {name}: {summary.skipped}/{len(slides)} slides already embedded, skipping")
        progress.info(f"embed {name} ({device}, batch {batch_size})")
        with progress.step(f"Loading {name}"):
            backbone = load_backbone(name, cfg.models_dir, device)
        counts = [tile_count(cohort.tiles_path(s.slide_id)) for s in todo]
        with progress.bar(sum(counts), f"embed {name}") as overall:
            for i, (slide, n) in enumerate(zip(todo, counts)):
                overall.set_postfix_str(f"slide {i + 1}/{len(todo)} · {slide.slide_id}")
                with progress.bar(n, "slide", inner=True) as inner:
                    try:
                        embed_slide(backbone, cohort.tiles_path(slide.slide_id),
                                    cohort.features_path(name, slide.slide_id), batch_size, _Bars(overall, inner))
                        summary.processed += 1
                        summary.tiles += n
                    except Exception as e:  # noqa: BLE001
                        if args.fail_fast:
                            raise
                        progress.log.error(f"embed {name} failed for {slide.slide_id}: {e}")
                        summary.failed.append(slide.slide_id)
        summary.report()
        status |= 1 if summary.failed else 0
        del backbone
    return status


def cmd_predict(cfg, cohort, args) -> int:
    from .predict import predict

    bundle = resolve_model(args)
    return predict(cfg, cohort, bundle, resolve_out(cfg, args, bundle, cohort), fail_fast=args.fail_fast)


def cmd_map(cfg, cohort, args) -> int:
    """Per-tile tables (csv), QuPath GeoJSON (geojson) and slide cluster map figures (png)."""
    import pandas as pd

    from .export import export
    from .plotting import slide_clustermap
    from .predict import CohortPredictions

    bundle = resolve_model(args)
    out = resolve_out(cfg, args, bundle, cohort)
    formats = [f.strip() for f in (getattr(args, "formats", None) or "csv,geojson,png").split(",") if f.strip()]
    preds = CohortPredictions(cfg, cohort, bundle, out)
    if getattr(args, "slide_id", None):
        preds.slides = [sp for sp in preds.slides if sp.slide.slide_id == args.slide_id]
    tile_formats = [f for f in formats if f in ("csv", "geojson")]
    if tile_formats:
        export(preds, out, tile_formats, getattr(args, "geojson_class", None) or "rd_association")
    if "png" not in formats:
        return 0
    figures = cfg.section("figures")
    clusters = _selected_clusters(args, preds, float(figures["clustermaps"]["max_p_value"]))
    training = None
    if getattr(args, "training_stats", None):
        training = pd.read_csv(Path(args.training_stats) / "slide_presence.csv")
    summary = progress.StageSummary("slide cluster maps")
    total = int(sum(len(sp.tile_index) for sp in preds.slides))
    with progress.bar(total, "slide cluster maps") as bar:
        for i, sp in enumerate(preds.slides):
            bar.set_postfix_str(f"slide {i + 1}/{len(preds.slides)} · {sp.slide.slide_id}")
            try:
                fig = slide_clustermap(preds, sp, clusters, training)
                _save(fig, out / "slide-clustermaps" / f"{sp.slide.slide_id}.png", int(figures["dpi"]))
                summary.processed += 1
                summary.tiles += len(sp.tile_index)
            except Exception as e:  # noqa: BLE001 - per-slide failures are reported, not fatal (D30.8)
                if args.fail_fast:
                    raise
                progress.log.error(f"slide cluster map failed for {sp.slide.slide_id}: {e}")
                summary.failed.append(sp.slide.slide_id)
            bar.update(len(sp.tile_index))
    summary.report()
    return 1 if summary.failed else 0


def cmd_clusters(cfg, cohort, args) -> int:
    """Cluster summary figure (clusters.png) and per-cluster tile samples (tile-samples/)."""
    from .plotting import clusters_summary, tile_sample
    from .predict import CohortPredictions

    bundle = resolve_model(args)
    out = resolve_out(cfg, args, bundle, cohort)
    figures = cfg.section("figures")
    preds = CohortPredictions(cfg, cohort, bundle, out)
    summary_spec = figures["summary"]
    with progress.step("Drawing cluster summary"):
        fig = clusters_summary(preds, int(summary_spec["mosaic"]), int(summary_spec["ncols"]), int(summary_spec["seed"]))
        _save(fig, out / "clusters.png", int(figures["dpi"]))
    spec = figures["tile_samples"]
    clusters = _selected_clusters(args, preds, float(spec["max_p_value"]))
    seeds = _ints(getattr(args, "seeds", None)) or [int(s) for s in spec["seeds"]]
    jobs = [(c, s) for c in clusters for s in seeds]
    with progress.bar(len(jobs), "tile samples", unit="figures") as bar:
        for c, s in jobs:
            fig = tile_sample(preds, c, s, int(spec["n_tiles"]), int(spec["ncols"]))
            _save(fig, out / "tile-samples" / f"cluster_{c}_seed_{s}.png", int(figures["dpi"]))
            bar.update(1)
    progress.info(f"Wrote {out / 'clusters.png'} and {len(jobs)} tile samples (clusters {clusters}) "
                  f"to {out / 'tile-samples'}")
    return 0


def cmd_evaluate(cfg, cohort, args) -> int:
    from .predict import evaluate

    bundle = resolve_model(args)
    evaluate(cfg, cohort, bundle, resolve_out(cfg, args, bundle, cohort))
    return 0


def cmd_fit(cfg, cohort, args) -> int:
    from .training import fit

    fit(cfg, cohort, Path(args.out) if args.out else cfg.output_dir / f"{cohort.name}-model")
    return 0


def cmd_run(cfg, cohort, args) -> int:
    stages = [
        ("tile", cmd_tile), ("embed", cmd_embed), ("predict", cmd_predict), ("map", cmd_map),
        ("clusters", cmd_clusters),
    ]
    status = 0
    for i, (name, command) in enumerate(stages, start=1):
        progress.header(i, len(stages), name)
        status |= command(cfg, cohort, args)
    print(resolve_out(cfg, args, resolve_model(args), cohort))
    return status


def cmd_text_features(cfg, args) -> int:
    from .tumor import load_text_features, text_features_path

    load_text_features(cfg, force=args.force)
    print(text_features_path(cfg))
    return 0


def cmd_release_model(cfg, args) -> int:
    bundle = ModelBundle.load(Path(args.src))
    dst = MODELS_ROOT / args.name
    if dst.exists():
        raise ConfigError(f"{dst} already exists.")
    bundle.meta["name"] = args.name
    if args.validation:
        bundle.meta["validation"] = json.loads(Path(args.validation).read_text())
    bundle.save(dst)
    (dst / "MODEL_CARD.md").write_text(model_card(bundle))
    print(dst)
    return 0


def model_card(bundle: ModelBundle) -> str:
    meta = bundle.meta
    ts, label = meta["training_summary"], meta["label"]
    p_values = meta["clusters"]["descriptors"]["p_value"]
    rows = "\n".join(
        f"| {i} | {a} | {'-' if p is None else f'{p:.3g}'} |" for i, (a, p) in enumerate(zip(bundle.associations, p_values))
    )
    validation = (
        "Not recorded." if not meta.get("validation")
        else "```json\n" + json.dumps(meta["validation"], indent=2) + "\n```"
    )
    slide_model = meta.get("slide_model")
    slide_text = (
        f"Logistic regression (C = {slide_model['C']}) on standardised degree-2 polynomial features of "
        f"{' and '.join(slide_model['inputs'])}. `rd_high = rd_high_proba >= {slide_model['threshold']}`."
        if slide_model else "Disabled (no cluster was significantly associated with high or low labels)."
    )
    return f"""# Model card: {meta['name']}

## Intended use

Clustering of tumor tiles from H&E whole-slide images of in-transit melanoma metastases into clusters associated
with high or low {label['display']}, and slide-level prediction of high {label['display']}.
For non-commercial academic research only.

## Inputs

- Tiles of {meta['input']['tile_mm']} mm, resized to {meta['input']['output_px']} px.
- Features: {meta['backbone']['hf_repo']} (revision `{meta['backbone']['hf_revision']}`).
- Tumor filter: {meta['tumor_filter']['hf_repo']} (revision `{meta['tumor_filter']['hf_revision']}`),
  prompts {meta['tumor_filter']['prompts_version']}.

Results are only meaningful with these backbones, revisions and tile size. Staining and scanner differences can
shift cluster assignments.

## Training

- {ts['n_slides']} slides ({ts['n_slides_high']} high, {ts['n_slides_low']} low; high means
  {label['display']} >= {label['cutoff']}), up to {ts['bagsize']} tiles per slide, {ts['n_tumor_tiles']:,} tumor tiles.
- {meta['preprocessing']['n_components']} PLS components, {meta['clusters']['n_clusters']} k-means clusters.

## Cluster associations

Mann-Whitney U test of per-slide cluster presence between high and low training slides
(alpha = {meta['rd_association']['alpha']}). The p-values describe the training cohort, not new data.

| Cluster | Association | p-value |
|---|---|---|
{rows}

## Slide model

{slide_text}

## Validation

{validation}

## Licence

Released under Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0), like the rest of the
repository. The model is derived from UNI-2 and CONCH outputs, whose own terms (CC BY-NC-ND 4.0) also prohibit
commercial use of derivatives.
"""


COMMANDS = {
    "run": cmd_run, "slides": cmd_slides, "tile": cmd_tile, "embed": cmd_embed, "fit": cmd_fit, "predict": cmd_predict,
    "map": cmd_map, "clusters": cmd_clusters, "evaluate": cmd_evaluate,
}


def main(argv: Optional[List[str]] = None) -> int:
    args = build_parser().parse_args(argv)
    progress.setup("off" if args.quiet else args.progress)
    try:
        with logging_redirect_tqdm(loggers=[progress.log]):
            cfg = Config.load(args.config)
            if args.command == "text-features":
                return cmd_text_features(cfg, args)
            if args.command == "release-model":
                return cmd_release_model(cfg, args)
            return COMMANDS[args.command](cfg, resolve_cohort(cfg, args), args)
    except ConfigError as e:
        progress.log.error(f"error: {e}")
        return 2
    except KeyboardInterrupt:
        progress.log.error("interrupted")
        return 130


if __name__ == "__main__":
    sys.exit(main())
