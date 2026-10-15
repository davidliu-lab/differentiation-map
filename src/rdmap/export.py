"""Per-tile maps: CSV with level-0 tile rectangles and QuPath-compatible GeoJSON (written by `rdmap map`)."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd

from . import progress
from .bundle import ModelBundle
from .predict import CohortPredictions, SlidePredictions

PALETTE = {
    "red": "#DF362A", "blue": "#2BBFD9", "yellow": "#FAC802", "green": "#8AB365", "pink": "#C670D2",
    "grey": "#788591", "dark red": "#7C2D27", "dark blue": "#006D8F", "dark yellow": "#AE7E1E",
    "dark green": "#008F6B", "dark pink": "#815FA5", "dark grey": "#434A51", "light red": "#E79892",
    "light blue": "#96D3DE", "light yellow": "#F5E294", "light green": "#C0E2A1", "light pink": "#DDAAE4",
    "light grey": "#8D96A0",
}
ASSOCIATION_COLORS = {"high": PALETTE["red"], "low": PALETTE["green"], "none": PALETTE["grey"]}
RD_CLASSES = {association: (f"RD {association}", color) for association, color in ASSOCIATION_COLORS.items()}
_CLUSTER_SHADES = {
    "high": ["red", "dark red", "light red"],
    "low": ["green", "dark green", "light green"],
    "none": ["blue", "yellow", "pink", "dark blue", "dark yellow", "dark pink", "light blue", "light yellow",
             "light pink", "dark grey", "light grey"],
}


def cluster_colors(associations: Sequence[str]) -> list[str]:
    """One colour per cluster: reds for high-associated, greens for low-associated, other hues for the rest."""
    used = {"high": 0, "low": 0, "none": 0}
    colors = []
    for association in associations:
        shades = _CLUSTER_SHADES[association]
        colors.append(PALETTE[shades[used[association] % len(shades)]])
        used[association] += 1
    return colors


def _rgb(hex_color: str) -> list[int]:
    return [int(hex_color[i : i + 2], 16) for i in (1, 3, 5)]


def tile_geometry(sp: SlidePredictions) -> tuple[np.ndarray, np.ndarray, int]:
    """Level-0 top-left corners (x, y) of the foreground tiles and the tile size in pixels."""
    attrs = sp.info.attrs
    if "size_px" not in attrs or "stride_px" not in attrs:
        raise ValueError(f"Tile file of {sp.slide.slide_id} lacks size_px/stride_px attributes.")
    stride, size = int(attrs["stride_px"]), int(attrs["size_px"])
    x = sp.info.col[sp.tile_index] * stride
    y = sp.info.row[sp.tile_index] * stride
    return x, y, size


def _write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    tmp.write_text(text)
    os.replace(tmp, path)


def export_slide(sp: SlidePredictions, bundle: ModelBundle, out_dir: Path, formats: Sequence[str], geojson_class: str):
    x, y, size = tile_geometry(sp)
    assoc = np.array(bundle.associations, dtype=object)
    palette = cluster_colors(bundle.associations)
    association = np.where(sp.cluster >= 0, assoc[np.clip(sp.cluster, 0, None)], "")
    if "csv" in formats:
        df = pd.DataFrame({
            "x": x, "y": y, "w": size, "h": size,
            "row": sp.info.row[sp.tile_index], "col": sp.info.col[sp.tile_index],
            "tumor": sp.tumor,
            "cluster": pd.array(np.where(sp.cluster >= 0, sp.cluster, -1), dtype="Int64"),
            "rd_association": association,
        })
        df.loc[df["cluster"] < 0, "cluster"] = pd.NA
        path = out_dir / "tile-tables" / f"{sp.slide.slide_id}.csv"
        _write_text(path, df.to_csv(index=False))
    if "geojson" in formats:
        features = []
        for i in np.flatnonzero(sp.tumor):
            x0, y0, c = int(x[i]), int(y[i]), int(sp.cluster[i])
            if geojson_class == "cluster":
                name, color = f"Cluster {c}", palette[c]
            else:
                name, color = RD_CLASSES[association[i]]
            features.append({
                "type": "Feature",
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [[[x0, y0], [x0 + size, y0], [x0 + size, y0 + size], [x0, y0 + size], [x0, y0]]],
                },
                "properties": {
                    "objectType": "tile",
                    "classification": {"name": name, "color": _rgb(color)},
                    "measurements": {"cluster": c},
                },
            })
        path = out_dir / "qupath" / f"{sp.slide.slide_id}.geojson"
        _write_text(path, json.dumps({"type": "FeatureCollection", "features": features}, separators=(",", ":")))


def export(preds: CohortPredictions, out_dir: Path, formats: Sequence[str], geojson_class: str) -> None:
    total = int(sum(len(sp.tile_index) for sp in preds.slides))
    summary = progress.StageSummary("tile tables / QuPath")
    with progress.bar(total, "tile tables / QuPath") as bar:
        for i, sp in enumerate(preds.slides):
            bar.set_postfix_str(f"slide {i + 1}/{len(preds.slides)} · {sp.slide.slide_id}")
            export_slide(sp, preds.bundle, out_dir, formats, geojson_class)
            summary.processed += 1
            summary.tiles += len(sp.tile_index)
            bar.update(len(sp.tile_index))
    summary.report()
