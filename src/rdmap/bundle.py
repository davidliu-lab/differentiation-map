"""Model bundle: inference-only parameters stored as model.json + model.npz (no pickle), and numpy inference."""

from __future__ import annotations

import hashlib
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import MODELS_ROOT, ConfigError
from .stats import HIGH, LOW, high_low_presence

FORMAT_VERSION = 1
SLIDE_INPUTS = {"presence_high": 0, "presence_low": 1}


def l2_normalize(X: np.ndarray) -> np.ndarray:
    """Row-wise L2 normalisation in the input dtype (same arithmetic as sklearn.preprocessing.normalize)."""
    norms = np.sqrt(np.einsum("ij,ij->i", X, X))
    norms[norms < 10 * np.finfo(norms.dtype).eps] = 1.0
    return X / norms[:, np.newaxis]


def poly2_terms(inputs: Sequence[str]) -> List[str]:
    """Degree-2 polynomial term names in sklearn's PolynomialFeatures order (without bias)."""
    terms = list(inputs)
    for i in range(len(inputs)):
        for j in range(i, len(inputs)):
            terms.append(f"{inputs[i]}^2" if i == j else f"{inputs[i]} {inputs[j]}")
    return terms


def poly2_features(X: np.ndarray) -> np.ndarray:
    columns = [X[:, i] for i in range(X.shape[1])]
    for i in range(X.shape[1]):
        for j in range(i, X.shape[1]):
            columns.append(X[:, i] * X[:, j])
    return np.stack(columns, axis=1)


class ModelBundle:
    def __init__(self, meta: Dict[str, Any], arrays: Dict[str, np.ndarray], path: Optional[Path] = None):
        self.meta = meta
        self.arrays = arrays
        self.path = path

    # ---- persistence -------------------------------------------------------------------------------------------

    @classmethod
    def load(cls, path: Path) -> "ModelBundle":
        path = Path(path)
        json_path, npz_path = path / "model.json", path / "model.npz"
        if not json_path.is_file() or not npz_path.is_file():
            raise ConfigError(f"Not a model directory (model.json/model.npz missing): {path}")
        meta = json.loads(json_path.read_text())
        if meta.get("format_version") != FORMAT_VERSION:
            raise ConfigError(f"Unsupported model format version {meta.get('format_version')} in {path}.")
        digest = hashlib.sha256(npz_path.read_bytes()).hexdigest()
        if digest != meta.get("arrays_sha256"):
            raise ConfigError(f"model.npz does not match the checksum in model.json: {path}")
        with np.load(npz_path, allow_pickle=False) as npz:
            arrays = {k: npz[k] for k in npz.files}
        return cls(meta, arrays, path)

    def save(self, path: Path) -> None:
        path = Path(path)
        path.mkdir(parents=True, exist_ok=True)
        npz_path = path / "model.npz"
        tmp = path / "model.partial.npz"
        np.savez(tmp, **{k: np.asarray(v, dtype=np.float64) for k, v in self.arrays.items()})
        tmp.replace(npz_path)
        self.meta["format_version"] = FORMAT_VERSION
        self.meta["arrays_sha256"] = hashlib.sha256(npz_path.read_bytes()).hexdigest()
        (path / "model.json").write_text(json.dumps(self.meta, indent=2) + "\n")
        self.path = path

    # ---- metadata ----------------------------------------------------------------------------------------------

    @property
    def name(self) -> str:
        return self.meta["name"]

    @property
    def n_clusters(self) -> int:
        return int(self.meta["clusters"]["n_clusters"])

    @property
    def backbone(self) -> str:
        return self.meta["backbone"]["name"]

    @property
    def tumor_backbone(self) -> str:
        return self.meta["tumor_filter"]["backbone"]

    @property
    def label(self) -> Dict[str, Any]:
        return self.meta["label"]

    def descriptor(self, key: str) -> np.ndarray:
        return np.array([np.nan if v is None else v for v in self.meta["clusters"]["descriptors"][key]], dtype=float)

    @property
    def p_values(self) -> np.ndarray:
        return self.descriptor("p_value")

    @property
    def associations(self) -> List[str]:
        # Early bundles called the third association "undetermined".
        return ["none" if a == "undetermined" else a for a in self.meta["clusters"]["descriptors"]["rd_association"]]

    def clusters_with(self, association: str) -> List[int]:
        return [i for i, a in enumerate(self.associations) if a == association]

    # ---- inference ---------------------------------------------------------------------------------------------

    def transform(self, X: np.ndarray) -> np.ndarray:
        """Features → L2 → PLS projection → L2, mirroring sklearn's arithmetic (including float32 centring)."""
        pre = self.meta["preprocessing"]
        X = np.array(X, copy=True)
        if X.dtype not in (np.float32, np.float64):
            X = X.astype(np.float64)
        if pre["l2_pre"]:
            X = l2_normalize(X)
        X -= self.arrays["x_mean"]
        X /= self.arrays["x_std"]
        Z = np.dot(X, self.arrays["x_rotations"])
        if pre["l2_post"]:
            Z = l2_normalize(Z)
        return Z

    def predict(self, Z: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Nearest centroid and its Euclidean distance."""
        centroids = self.arrays["centroids"]
        d2 = ((Z[:, np.newaxis, :] - centroids[np.newaxis, :, :]) ** 2).sum(axis=2)
        clusters = d2.argmin(axis=1)
        return clusters.astype(np.int64), np.sqrt(d2[np.arange(len(Z)), clusters])

    def predict_features(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        if len(X) == 0:
            return np.zeros(0, dtype=np.int64), np.zeros(0)
        return self.predict(self.transform(X))

    @property
    def has_slide_model(self) -> bool:
        return self.meta.get("slide_model") is not None

    def slide_proba(self, presence: np.ndarray) -> Optional[np.ndarray]:
        """P(high) per slide from the per-cluster presence matrix (S, K); None if the model has no slide model."""
        if not self.has_slide_model:
            return None
        spec = self.meta["slide_model"]
        hl = high_low_presence(presence, self.associations)
        X = hl[:, [SLIDE_INPUTS[name] for name in spec["inputs"]]]
        features = (poly2_features(X) - self.arrays["slide_feat_mean"]) / self.arrays["slide_feat_std"]
        coef = self.arrays["slide_coef"]
        logit = coef[0] + features @ coef[1:]
        return 1.0 / (1.0 + np.exp(-logit))


def default_model_path() -> Path:
    """The newest released bundle under the repository's `models/` directory (D29.2)."""
    candidates = []
    for json_path in MODELS_ROOT.glob("*/model.json"):
        try:
            created = json.loads(json_path.read_text()).get("created", "")
        except (OSError, ValueError):
            continue
        candidates.append((created, json_path.parent))
    if not candidates:
        raise ConfigError(
            f"No released model found in {MODELS_ROOT}. Pass --model (e.g. a model trained with `rdmap fit`)."
        )
    return sorted(candidates)[-1][1]


def now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


__all__ = ["ModelBundle", "default_model_path", "l2_normalize", "poly2_features", "poly2_terms", "HIGH", "LOW"]
