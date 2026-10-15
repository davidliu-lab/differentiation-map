"""Tiling: read whole-slide images, cut a regular grid of tiles, flag background/artifact tiles, store as HDF5.

Ported from the original `extract-tiles.py` (HDF5 mode). OpenSlide reads `.svs`-like formats; tifffile + zarr
reads `.tif`/`.tiff` (including flat OME-TIFFs).
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

import numpy as np
from PIL import Image
from scipy import ndimage

from .storage import atomic_h5

Range = Tuple[int, int]
RESAMPLE = Image.Resampling.BICUBIC  # Pillow's default for RGB resize, spelled out for version independence


def _pairs(flat: Sequence[int]) -> List[Range]:
    if len(flat) % 2:
        raise ValueError(f"HSV ranges must be (lo, hi) pairs, got {list(flat)}.")
    return [(int(flat[i]), int(flat[i + 1])) for i in range(0, len(flat), 2)]


@dataclass
class TilingParams:
    size_px: int
    stride_px: int
    output_px: int
    threshold: float
    background: List[Tuple[Range, Range, Range]]  # (hue, saturation, value) per rule
    artifact: List[Tuple[Range, Range, Range]]

    @classmethod
    def from_config(cls, tiling: dict, mpp: Optional[float]) -> "TilingParams":
        size = tiling.get("size_px")
        if size is None:
            if mpp is None:
                raise ValueError("The slide resolution (mpp) is unknown; set an mpp column/value or tiling.size_px.")
            size = round(float(tiling["tile_mm"]) * 1000.0 / mpp)
        stride = int(tiling.get("stride_px", size))
        return cls(
            size_px=int(size),
            stride_px=stride,
            output_px=int(tiling["output_px"]),
            threshold=float(tiling["background_threshold"]),
            background=list(zip(_pairs(tiling["background_hue"]), _pairs(tiling["background_sat"]),
                                _pairs(tiling["background_val"]))),
            artifact=list(zip(_pairs(tiling["artifact_hue"]), _pairs(tiling["artifact_sat"]),
                              _pairs(tiling["artifact_val"]))),
        )


def grid_shape(width: int, height: int, size: int, stride: int) -> Tuple[int, int]:
    return max(0, (width - size) // stride + 1), max(0, (height - size) // stride + 1)


class WSI:
    """Minimal whole-slide image reader with the original script's thumbnail and tile semantics."""

    def __init__(self, path: Path):
        self.path = Path(path)
        self.is_tiff = self.path.name.lower().endswith((".tif", ".tiff"))
        self._full: Optional[np.ndarray] = None
        if self.is_tiff:
            import tifffile
            import zarr

            self._store = tifffile.imread(str(self.path), aszarr=True)
            self._zarr = zarr.open(self._store, mode="r")
            self._is_group = isinstance(self._zarr, zarr.Group)
            self._data = self._zarr["0"] if self._is_group else self._zarr
            self._channels_first = self._data.shape[0] == 3
            self.height, self.width = self._data.shape[1:3] if self._channels_first else self._data.shape[:2]
        else:
            import openslide

            self._slide = openslide.OpenSlide(str(self.path))
            self.width, self.height = self._slide.level_dimensions[0]

    def close(self) -> None:
        if self.is_tiff:
            self._store.close()
        else:
            self._slide.close()
        self._full = None

    def __enter__(self) -> "WSI":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def mpp(self) -> Optional[float]:
        """Microns per pixel from the file metadata, if available."""
        if not self.is_tiff:
            value = self._slide.properties.get("openslide.mpp-x")
            return float(value) if value else None
        import tifffile

        with tifffile.TiffFile(str(self.path)) as tif:
            ome = tif.ome_metadata or ""
            match = re.search(r'PhysicalSizeX="([0-9.eE+-]+)"', ome)
            if match:
                unit = re.search(r'PhysicalSizeXUnit="([^"]+)"', ome)
                factor = {"nm": 1e-3, "mm": 1e3, "cm": 1e4}.get(unit.group(1) if unit else "µm", 1.0)
                return float(match.group(1)) * factor
            page = tif.pages[0]
            tag, unit = page.tags.get("XResolution"), page.tags.get("ResolutionUnit")
            if tag is not None and unit is not None and int(unit.value) in (2, 3):
                numerator, denominator = tag.value
                if numerator:
                    per_unit = 25400.0 if int(unit.value) == 2 else 10000.0  # inch / cm in microns
                    return per_unit * denominator / numerator
        return None

    def thumbnail(self) -> Image.Image:
        if not self.is_tiff:
            level = self._slide.level_count - 1
            return self._slide.read_region((0, 0), level=level, size=self._slide.level_dimensions[-1]).convert("RGB")
        if self._is_group:
            data = self._zarr[max(self._zarr.keys())][:]
            if data.shape[0] == 3:
                data = data.transpose(1, 2, 0)
            return Image.fromarray(data)
        # Flat TIFF: read the full image once (it is also reused for fast tile reads), zoom to 1024 px.
        self._full = self._data[:]
        data = self._full.transpose(1, 2, 0) if self._channels_first else self._full
        zoom = 1024 / max(data.shape[0], data.shape[1])
        return Image.fromarray(ndimage.zoom(data, (zoom, zoom, 1), order=1))

    def read(self, x: int, y: int, size: int) -> Image.Image:
        if not self.is_tiff:
            return self._slide.read_region(location=(x, y), level=0, size=(size, size)).convert("RGB")
        source = self._full if self._full is not None else self._data
        if self._channels_first:
            data = np.asarray(source[:, y : y + size, x : x + size]).transpose(1, 2, 0)
        else:
            data = np.asarray(source[y : y + size, x : x + size, :])
        return Image.fromarray(data)


def _hsv_mask(hue, sat, val, rules) -> np.ndarray:
    mask = np.zeros(hue.shape, dtype=bool)
    for (h0, h1), (s0, s1), (v0, v1) in rules:
        mask |= (hue >= h0) & (hue <= h1) & (sat >= s0) & (sat <= s1) & (val >= v0) & (val <= v1)
    return mask


def background_probability(tile: Image.Image, params: TilingParams) -> float:
    """Fraction of background/artifact pixels on a 100x100 HSV downsample (median/min/max filtered)."""
    hsv = np.array(tile.resize((100, 100), RESAMPLE).convert("HSV"))
    hue, sat, val = hsv[..., 0], hsv[..., 1], hsv[..., 2]
    background = _hsv_mask(hue, sat, val, params.background)
    background = ndimage.median_filter(background, size=5)
    background = ndimage.minimum_filter(background, size=20)
    artifact = _hsv_mask(hue, sat, val, params.artifact)
    artifact = ndimage.minimum_filter(artifact, size=10)
    artifact = ndimage.maximum_filter(artifact, size=100)
    return float(np.mean(background | artifact))


def tile_slide(wsi: WSI, out_path: Path, slide_id: str, params: TilingParams, mpp: Optional[float], bar=None) -> int:
    """Tile one slide into `out_path` (written atomically). Returns the number of tiles."""
    n_x, n_y = grid_shape(wsi.width, wsi.height, params.size_px, params.stride_px)
    n = n_x * n_y
    thumbnail = np.asarray(wsi.thumbnail(), dtype=np.uint8)
    out = params.output_px
    with atomic_h5(out_path) as f:
        tiles = f.create_dataset("tiles", shape=(n, out, out, 3), dtype=np.uint8, chunks=(1, out, out, 3))
        rows = f.create_dataset("row", shape=(n,), dtype=np.int64)
        cols = f.create_dataset("col", shape=(n,), dtype=np.int64)
        background = f.create_dataset("background", shape=(n,), dtype=bool)
        f.create_dataset("thumbnail", data=thumbnail)
        f.attrs.update(
            slide_id=slide_id, mpp=float("nan") if mpp is None else float(mpp),
            tile_mm=float("nan") if mpp is None else params.size_px * mpp / 1000.0,
            size_px=params.size_px, stride_px=params.stride_px, output_px=out,
            slide_width=int(wsi.width), slide_height=int(wsi.height),
        )
        for idx, (i, j) in enumerate(np.ndindex((n_x, n_y))):
            tile = wsi.read(i * params.stride_px, j * params.stride_px, params.size_px)
            cols[idx], rows[idx] = i, j
            background[idx] = background_probability(tile, params) > params.threshold
            tiles[idx] = np.asarray(tile.resize((out, out), RESAMPLE))
            if bar is not None:
                bar.update(1)
    return n
