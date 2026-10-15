"""Configuration: TOML loading/merging, ${VAR:-default} expansion and cohort definitions."""

from __future__ import annotations

import copy
import os
import re
import tomllib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = REPO_ROOT / "configs" / "default.toml"
MODELS_ROOT = REPO_ROOT / "models"
DEFAULT_WSI_GLOBS = ["*.svs", "*.tif", "*.tiff", "*.ndpi", "*.mrxs", "*.scn"]

_VAR_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)(?::-([^}]*))?\}")


class ConfigError(ValueError):
    pass


def expand_vars(value: str) -> str:
    """Expand ${VAR} and ${VAR:-default}; unset (or empty) variables without a default are an error."""

    def replace(match: re.Match) -> str:
        name, default = match.group(1), match.group(2)
        current = os.environ.get(name)
        if current:
            return current
        if default is not None:
            return default
        raise ConfigError(f"Environment variable {name} is not set (referenced as {match.group(0)}).")

    return _VAR_PATTERN.sub(replace, value)


def _expand_tree(value: Any) -> Any:
    if isinstance(value, str):
        return expand_vars(value)
    if isinstance(value, dict):
        return {k: _expand_tree(v) for k, v in value.items()}
    if isinstance(value, list):
        return [_expand_tree(v) for v in value]
    return value


def deep_merge(base: Dict[str, Any], override: Dict[str, Any]) -> Dict[str, Any]:
    result = copy.deepcopy(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = deep_merge(result[key], value)
        else:
            result[key] = copy.deepcopy(value)
    return result


def to_path(value: str | os.PathLike) -> Path:
    return Path(os.path.expanduser(str(value)))


@dataclass
class Cohort:
    name: str
    root: Path
    tiling: Dict[str, Any]
    wsi_dir: Optional[Path] = None
    wsi_glob: List[str] = field(default_factory=lambda: list(DEFAULT_WSI_GLOBS))
    metadata: Optional[Path] = None
    label_column: Optional[str] = None
    slide_id_column: str = "slide_id"
    filename_column: str = "filename"
    mpp: Optional[float] = None

    @property
    def tiles_dir(self) -> Path:
        return self.root / "tiles"

    def features_dir(self, backbone: str) -> Path:
        return self.root / "features" / backbone

    def tiles_path(self, slide_id: str) -> Path:
        return self.tiles_dir / f"{slide_id}.h5"

    def features_path(self, backbone: str, slide_id: str) -> Path:
        return self.features_dir(backbone) / f"{slide_id}.h5"


class Config:
    """Merged configuration. `configs/default.toml` is always the base layer."""

    def __init__(self, raw: Dict[str, Any]):
        self.raw = raw

    @classmethod
    def load(cls, files: Sequence[str | os.PathLike] = ()) -> "Config":
        if not DEFAULT_CONFIG_PATH.is_file():
            raise ConfigError(f"Default configuration not found at {DEFAULT_CONFIG_PATH}.")
        raw: Dict[str, Any] = {}
        for path in [DEFAULT_CONFIG_PATH, *map(Path, files)]:
            if not path.is_file():
                raise ConfigError(f"Configuration file not found: {path}")
            with open(path, "rb") as f:
                raw = deep_merge(raw, tomllib.load(f))
        return cls(_expand_tree(raw))

    def section(self, name: str) -> Dict[str, Any]:
        return self.raw.get(name, {})

    def get(self, section: str, key: str, default: Any = None) -> Any:
        return self.section(section).get(key, default)

    @property
    def data_dir(self) -> Path:
        return to_path(self.get("paths", "data_dir", "data"))

    @property
    def output_dir(self) -> Path:
        return to_path(self.get("paths", "output_dir", "outputs"))

    @property
    def models_dir(self) -> Path:
        return to_path(self.get("paths", "models_dir", "~/.cache/rdmap"))

    @property
    def text_features_dir(self) -> Path:
        return self.data_dir / "text_features"

    def cohort_names(self) -> List[str]:
        return list(self.section("cohorts").keys())

    def cohort(self, name: str) -> Cohort:
        cohorts = self.section("cohorts")
        if name not in cohorts:
            known = ", ".join(cohorts) or "none"
            raise ConfigError(f"Unknown cohort '{name}' (defined cohorts: {known}).")
        return self._make_cohort(name, cohorts[name])

    def add_adhoc_cohort(
        self,
        wsi_dir: str,
        name: Optional[str] = None,
        glob: Optional[str] = None,
        metadata: Optional[str] = None,
        label_column: Optional[str] = None,
        mpp: Optional[float] = None,
    ) -> Cohort:
        wsi_path = to_path(wsi_dir).resolve()
        name = name or wsi_path.name
        entry: Dict[str, Any] = {"wsi_dir": str(wsi_path)}
        if glob:
            entry["wsi_glob"] = glob
        if metadata:
            entry["metadata"] = metadata
        if label_column:
            entry["label_column"] = label_column
        if mpp is not None:
            entry["mpp"] = mpp
        self.raw.setdefault("cohorts", {})[name] = deep_merge(self.section("cohorts").get(name, {}), entry)
        return self.cohort(name)

    def _make_cohort(self, name: str, entry: Dict[str, Any]) -> Cohort:
        tiling = deep_merge(self.section("tiling"), entry.get("tiling", {}))
        globs = entry.get("wsi_glob", DEFAULT_WSI_GLOBS)
        if isinstance(globs, str):
            globs = [g.strip() for g in globs.split(",") if g.strip()]
        return Cohort(
            name=name,
            root=to_path(entry.get("root", self.data_dir / name)),
            tiling=tiling,
            wsi_dir=to_path(entry["wsi_dir"]) if entry.get("wsi_dir") else None,
            wsi_glob=list(globs),
            metadata=to_path(entry["metadata"]) if entry.get("metadata") else None,
            label_column=entry.get("label_column"),
            slide_id_column=entry.get("slide_id_column", "slide_id"),
            filename_column=entry.get("filename_column", "filename"),
            mpp=float(entry["mpp"]) if entry.get("mpp") is not None else None,
        )
