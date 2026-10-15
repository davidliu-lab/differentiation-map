"""HDF5 storage for tiles, features and predictions. This is the only module that uses h5py."""

from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterator, Mapping, Optional

import h5py
import numpy as np


@contextmanager
def atomic_h5(path: Path) -> Iterator[h5py.File]:
    """Write an HDF5 file to `<path>.partial` and rename it on success, so interrupted writes never look done."""
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial")
    if tmp.exists():
        tmp.unlink()
    file = h5py.File(tmp, "w")
    try:
        yield file
    except BaseException:
        file.close()
        tmp.unlink(missing_ok=True)
        raise
    file.close()
    os.replace(tmp, path)


def _attrs(obj: h5py.HLObject) -> Dict[str, Any]:
    result: Dict[str, Any] = {}
    for key, value in obj.attrs.items():
        if isinstance(value, bytes):
            value = value.decode()
        elif isinstance(value, np.generic):
            value = value.item()
        result[key] = value
    return result


def _read_rows(dataset: h5py.Dataset, indices: Optional[np.ndarray]) -> np.ndarray:
    """Read rows in arbitrary order (h5py needs increasing, unique indices)."""
    if indices is None:
        return dataset[()]
    indices = np.asarray(indices, dtype=np.int64)
    if len(indices) == 0:
        return dataset[:0]
    unique, inverse = np.unique(indices, return_inverse=True)
    return dataset[unique][inverse]


@dataclass
class TileInfo:
    row: np.ndarray
    col: np.ndarray
    background: np.ndarray
    attrs: Dict[str, Any]

    @property
    def n(self) -> int:
        return len(self.row)

    @property
    def foreground(self) -> np.ndarray:
        return np.flatnonzero(~self.background)


def read_tile_info(path: Path) -> TileInfo:
    with h5py.File(path, "r") as f:
        return TileInfo(
            row=f["row"][()].astype(np.int64),
            col=f["col"][()].astype(np.int64),
            background=f["background"][()].astype(bool),
            attrs=_attrs(f),
        )


def read_tile_images(path: Path, indices: np.ndarray) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return np.asarray(_read_rows(f["tiles"], indices), dtype=np.uint8)


def read_thumbnail(path: Path) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return np.asarray(f["thumbnail"][()], dtype=np.uint8)


def read_features(path: Path, indices: Optional[np.ndarray] = None) -> np.ndarray:
    with h5py.File(path, "r") as f:
        return _read_rows(f["features"], indices)


def read_features_attrs(path: Path) -> Dict[str, Any]:
    with h5py.File(path, "r") as f:
        return _attrs(f)


def write_features(path: Path, features: np.ndarray, attrs: Mapping[str, Any]) -> None:
    with atomic_h5(path) as f:
        f.create_dataset("features", data=np.asarray(features, dtype=np.float32))
        f.attrs.update(dict(attrs))


def write_arrays(path: Path, arrays: Mapping[str, np.ndarray], attrs: Mapping[str, Any]) -> None:
    with atomic_h5(path) as f:
        for name, value in arrays.items():
            f.create_dataset(name, data=value)
        f.attrs.update(dict(attrs))


def read_arrays(path: Path) -> Dict[str, np.ndarray]:
    with h5py.File(path, "r") as f:
        return {name: f[name][()] for name in f.keys()}


def read_attrs(path: Path) -> Dict[str, Any]:
    with h5py.File(path, "r") as f:
        return _attrs(f)
