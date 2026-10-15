"""Cohorts: slide discovery, labels from the optional metadata CSV, and the per-slide tile sampler."""

from __future__ import annotations

import fnmatch
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

from . import progress
from .config import Cohort, ConfigError

_MULTI_EXTENSIONS = (".ome.tiff", ".ome.tif")
_BINARY_LABELS = {
    "yes": "Yes", "y": "Yes", "true": "Yes", "high": "Yes", "1": "Yes", "1.0": "Yes",
    "no": "No", "n": "No", "false": "No", "low": "No", "0": "No", "0.0": "No",
}


@dataclass
class Slide:
    slide_id: str
    wsi_path: Optional[Path] = None
    label: Optional[str] = None  # binarised: "Yes" (high) / "No" (low)
    label_value: Optional[float] = None  # continuous score, or 1.0/0.0 for binary labels
    mpp: Optional[float] = None


def slide_id_from_filename(name: str) -> str:
    lower = name.lower()
    for ext in _MULTI_EXTENSIONS:
        if lower.endswith(ext):
            return name[: -len(ext)]
    return Path(name).stem


def discover_wsis(cohort: Cohort) -> Dict[str, Path]:
    if cohort.wsi_dir is None or not cohort.wsi_dir.is_dir():
        return {}
    found: Dict[str, Path] = {}
    for path in sorted(cohort.wsi_dir.iterdir()):
        if path.is_file() and any(fnmatch.fnmatch(path.name, pattern) for pattern in cohort.wsi_glob):
            found.setdefault(slide_id_from_filename(path.name), path)
    return found


def parse_labels(values: pd.Series, cutoff: float) -> Tuple[List[Optional[str]], List[Optional[float]], str]:
    """Return binarised labels, numeric label values and the label kind ('continuous' or 'binary')."""
    numeric = pd.to_numeric(values, errors="coerce")
    non_missing = values.notna() & (values.astype(str).str.strip() != "")
    if non_missing.any() and numeric[non_missing].notna().all():
        unique = set(numeric[non_missing].unique())
        kind = "binary" if unique <= {0.0, 1.0} and len(unique) == 2 else "continuous"
        labels = [None if pd.isna(v) else ("Yes" if v >= cutoff else "No") for v in numeric]
        return labels, [None if pd.isna(v) else float(v) for v in numeric], kind
    labels, label_values = [], []
    for raw in values:
        if pd.isna(raw) or str(raw).strip() == "":
            labels.append(None)
            label_values.append(None)
            continue
        key = str(raw).strip().lower()
        if key not in _BINARY_LABELS:
            raise ConfigError(f"Cannot interpret label value '{raw}' (expected a number or Yes/No).")
        labels.append(_BINARY_LABELS[key])
        label_values.append(1.0 if _BINARY_LABELS[key] == "Yes" else 0.0)
    return labels, label_values, "binary"


def list_slides(cohort: Cohort, cutoff: float = 0.5, require_labels: bool = False) -> Tuple[List[Slide], str]:
    """Slides of a cohort, in metadata-CSV order (or sorted by ID without a CSV), and the label kind."""
    wsis = discover_wsis(cohort)
    kind = "none"
    if cohort.metadata is not None:
        if not cohort.metadata.is_file():
            raise ConfigError(f"Metadata CSV not found for cohort '{cohort.name}': {cohort.metadata}")
        df = pd.read_csv(cohort.metadata, dtype={cohort.slide_id_column: str})
        if cohort.slide_id_column not in df.columns:
            raise ConfigError(f"Column '{cohort.slide_id_column}' not found in {cohort.metadata}.")
        labels: List[Optional[str]] = [None] * len(df)
        values: List[Optional[float]] = [None] * len(df)
        if cohort.label_column:
            if cohort.label_column not in df.columns:
                raise ConfigError(f"Label column '{cohort.label_column}' not found in {cohort.metadata}.")
            labels, values, kind = parse_labels(df[cohort.label_column], cutoff)
        slides = []
        for i, record in enumerate(df.to_dict("records")):
            slide_id = str(record[cohort.slide_id_column])
            filename = record.get(cohort.filename_column) if cohort.filename_column in df.columns else None
            filename = filename if isinstance(filename, str) and filename.strip() else None
            wsi_path = cohort.wsi_dir / filename if (filename and cohort.wsi_dir) else wsis.get(slide_id)
            mpp = record.get("mpp") if "mpp" in df.columns else None
            mpp = float(mpp) if mpp is not None and not pd.isna(mpp) else None
            slides.append(Slide(slide_id, wsi_path, labels[i], values[i], mpp))
    elif wsis:
        slides = [Slide(slide_id, path) for slide_id, path in sorted(wsis.items())]
    elif cohort.tiles_dir.is_dir():
        slides = [Slide(p.stem) for p in sorted(cohort.tiles_dir.glob("*.h5"))]
    else:
        raise ConfigError(f"No slides found for cohort '{cohort.name}' (no metadata CSV, WSIs or tiles).")
    if require_labels:
        if kind == "none":
            raise ConfigError(f"Cohort '{cohort.name}' needs a metadata CSV and label_column for this command.")
        dropped = [s.slide_id for s in slides if s.label is None]
        if dropped:
            progress.info(f"{len(dropped)} slides without a label are ignored: {', '.join(dropped)}")
        slides = [s for s in slides if s.label is not None]
    return slides, kind


def with_files(cohort: Cohort, slides: Iterable[Slide], backbones: Sequence[str]) -> List[Slide]:
    """Keep slides that have a tile file and feature files for all `backbones`; warn about the rest."""
    kept, missing = [], []
    for slide in slides:
        paths = [cohort.tiles_path(slide.slide_id)] + [cohort.features_path(b, slide.slide_id) for b in backbones]
        (kept if all(p.is_file() for p in paths) else missing).append(slide)
    if missing:
        progress.log.warning(
            f"{len(missing)} slides skipped (missing tiles or {'/'.join(backbones)} features): "
            + ", ".join(s.slide_id for s in missing)
        )
    return kept


def sample_tiles(backgrounds: Sequence[np.ndarray], bagsize: int, seed: int) -> List[np.ndarray]:
    """Per slide, sample up to `bagsize` foreground tiles without replacement.

    One RandomState is shared across slides in the given order and the sampled order is kept, so the result
    is reproducible given the slide order.
    """
    random = np.random.RandomState(seed=seed)
    sampled = []
    for background in backgrounds:
        indices = np.nonzero(np.logical_not(background))[0]
        if bagsize > 0 and len(indices) > bagsize:
            indices = random.choice(indices, size=bagsize, replace=False)
        sampled.append(indices)
    return sampled
