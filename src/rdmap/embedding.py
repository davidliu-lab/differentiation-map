"""Foundation-model backbones (UNI-2, CONCH): registry with pinned Hugging Face revisions, and feature extraction.

Heavy dependencies (torch, timm, conch) are imported lazily inside functions.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict

import numpy as np

from . import progress

IMAGENET_MEAN, IMAGENET_STD = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225)
OPENAI_MEAN, OPENAI_STD = (0.48145466, 0.4578275, 0.40821073), (0.26862954, 0.26130258, 0.27577711)
READ_CHUNK = 1000  # tiles read from disk at a time (a multiple of the default batch size)

BACKBONES: Dict[str, Dict[str, Any]] = {
    "uni2": {
        "hf_repo": "MahmoodLab/UNI2-h",
        "hf_revision": "d517a8dd47902dd7c308b3c36f63bce47e7b9a43",
        "filename": "pytorch_model.bin",
        "dim": 1536,
        "input_px": 224,
        "mean": IMAGENET_MEAN,
        "std": IMAGENET_STD,
    },
    "conch": {
        "hf_repo": "MahmoodLab/CONCH",
        "hf_revision": "f9ca9f877171a28ade80228fb195ac5d79003357",
        "filename": "pytorch_model.bin",
        "dim": 512,
        "input_px": 448,
        "mean": OPENAI_MEAN,
        "std": OPENAI_STD,
    },
}


def backbone_info(name: str) -> Dict[str, Any]:
    if name not in BACKBONES:
        raise ValueError(f"Unknown backbone '{name}' (available: {', '.join(BACKBONES)}).")
    return BACKBONES[name]


def hf_token() -> str | None:
    return os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or None


def download_weights(name: str, models_dir: Path) -> Path:
    """Download (or reuse from cache) the pinned checkpoint of a gated backbone."""
    from huggingface_hub import hf_hub_download

    info = backbone_info(name)
    try:
        path = hf_hub_download(
            info["hf_repo"],
            filename=info["filename"],
            revision=info["hf_revision"],
            cache_dir=str(models_dir),
            token=hf_token(),
        )
    except Exception as e:  # noqa: BLE001 - re-raised with actionable context
        raise RuntimeError(
            f"Could not download {info['hf_repo']} (revision {info['hf_revision'][:8]}). The model is gated: "
            f"request access at https://huggingface.co/{info['hf_repo']} and set HF_TOKEN. Original error: {e}"
        ) from e
    return Path(path)


def load_conch(models_dir: Path, device: str = "cpu"):
    from conch.open_clip_custom import create_model_from_pretrained

    checkpoint = download_weights("conch", models_dir)
    model = create_model_from_pretrained("conch_ViT-B-16", checkpoint_path=str(checkpoint), return_transform=False)
    return model.to(device).eval()


def resolve_device(setting: str = "auto") -> str:
    import torch

    if setting == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return setting


@dataclass
class Backbone:
    name: str
    model: Any
    transform: Callable
    forward: Callable
    device: str


def load_backbone(name: str, models_dir: Path, device: str) -> Backbone:
    """Load a backbone with the preprocessing and forward pass of the original feature pipelines."""
    import torch
    from torchvision.transforms import v2 as transforms

    info = backbone_info(name)
    transform = transforms.Compose([
        transforms.Resize(info["input_px"], antialias=True),
        transforms.CenterCrop(info["input_px"]),
        transforms.Normalize(mean=info["mean"], std=info["std"]),
    ])
    if name == "uni2":
        import timm

        checkpoint = download_weights("uni2", models_dir)
        model = timm.create_model(
            "vit_giant_patch14_224", img_size=224, patch_size=14, depth=24, num_heads=24, init_values=1e-5,
            embed_dim=1536, mlp_ratio=2.66667 * 2, num_classes=0, no_embed_class=True,
            mlp_layer=timm.layers.SwiGLUPacked, act_layer=torch.nn.SiLU, reg_tokens=8, dynamic_img_size=True,
        )
        model.load_state_dict(torch.load(checkpoint, map_location="cpu"), strict=True)
        model = model.to(device).eval()
        return Backbone(name, model, transform, lambda m, x: m(x), device)
    if name == "conch":
        model = load_conch(models_dir, device)
        return Backbone(name, model, transform,
                        lambda m, x: m.encode_image(x, proj_contrast=False, normalize=False), device)
    raise ValueError(f"Unknown backbone '{name}'.")


def embed_tiles(backbone: Backbone, tiles: np.ndarray, batch_size: int) -> np.ndarray:
    """Features for uint8 HWC tiles. Halves the batch size on CUDA out-of-memory errors."""
    import torch

    X = tiles.astype(np.float32).transpose(0, 3, 1, 2) / 255.0
    while True:
        try:
            results = []
            for start in range(0, len(X), batch_size):
                batch = backbone.transform(torch.tensor(X[start : start + batch_size]).to(backbone.device))
                with torch.inference_mode():
                    results.append(backbone.forward(backbone.model, batch).float().cpu().numpy())
            return np.concatenate(results) if results else np.zeros((0, backbone_info(backbone.name)["dim"]))
        except torch.cuda.OutOfMemoryError:
            if batch_size == 1:
                raise
            batch_size = max(1, batch_size // 2)
            torch.cuda.empty_cache()
            progress.log.warning(f"CUDA out of memory; retrying with batch size {batch_size}")


def embed_slide(backbone: Backbone, tiles_path: Path, out_path: Path, batch_size: int, bar=None) -> int:
    """Embed every tile of a slide (background included, so rows align with the tile file); atomic write."""
    import h5py

    from .storage import write_features

    info = backbone_info(backbone.name)
    with h5py.File(tiles_path, "r") as f:
        dataset = f["tiles"]
        n = len(dataset)
        features = np.zeros((n, info["dim"]), dtype=np.float32)
        for start in range(0, n, READ_CHUNK):
            chunk = np.asarray(dataset[start : start + READ_CHUNK], dtype=np.uint8)
            features[start : start + len(chunk)] = embed_tiles(backbone, chunk, batch_size)
            if bar is not None:
                bar.update(len(chunk))
    write_features(out_path, features, {"backbone": backbone.name, "hf_repo": info["hf_repo"],
                                        "hf_revision": info["hf_revision"]})
    return n


def tile_count(tiles_path: Path) -> int:
    import h5py

    with h5py.File(tiles_path, "r") as f:
        return len(f["tiles"])


__all__ = ["BACKBONES", "Backbone", "backbone_info", "download_weights", "embed_slide", "embed_tiles",
           "load_backbone", "load_conch", "resolve_device", "tile_count"]
