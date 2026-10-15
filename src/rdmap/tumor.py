"""Zero-shot CONCH tumor filter: prompts, class text features (auto-computed and cached) and the tumor mask."""

from __future__ import annotations

from pathlib import Path

import numpy as np

from . import progress
from .config import Config

PROMPTS_VERSION = "v1"
CLASSES = ["tumor", "normal", "other"]
TUMOR_CLASS = 0

PROMPTS = {
    "tumor": [
        "an H&E histopathology tile showing melanoma tumor nests in the dermis",
        "a digitized WSI patch (20×) of malignant melanocytic tumor on H&E",
        "an H&E photomicrograph of cancer tissue with pleomorphic nuclei",
        "an H&E histology patch of invasive melanoma tumor region",
        "an H&E tile of tumor cells with high nuclear-to-cytoplasm ratio",
        "an H&E tumor region (overstained)",
        "an H&E tumor region (understained)",
    ],
    "normal": [
        "an H&E histopathology tile of normal skin without tumor",
        "a WSI patch (20×) of benign dermis and epidermis on H&E",
        "an H&E photomicrograph of normal stroma without atypia",
        "an H&E tile of benign adnexal structures (hair follicle, sebaceous gland)",
        "an H&E tile of normal collagenous dermis with scattered lymphocytes",
        "an H&E normal tissue region (eosin-rich)",
        "an H&E normal tissue region (hematoxylin-dense)",
    ],
    "other": [
        "an H&E WSI patch with stained background and no tissue structures",
        "an H&E slide artifact: pen ink marking with no tissue",
        "an H&E slide artifact: tissue fold causing duplicated stained bands",
        "an H&E slide artifact: out-of-focus blurry region, no cellular detail",
        "an H&E slide artifact: air bubble under coverslip",
        "an H&E slide artifact: dust or debris on the slide",
        "an H&E slide artifact: scanner stripe / banding lines",
        "an H&E slide artifact: coverslip edge / label area, no tissue",
        "an H&E slide artifact: glare or reflection with no histologic structures",
    ],
}


def text_features_path(cfg: Config) -> Path:
    return cfg.text_features_dir / f"conch-{PROMPTS_VERSION}.npy"


def _tokenize(texts):
    """CONCH tokens exactly as `conch.open_clip_custom.tokenize` produced them with transformers 4.

    The tokenizer is driven directly through the `tokenizers` library so results do not depend on the transformers
    version: transformers 4 ignored the file's `add_prefix_space=True` (first word tokenized without a leading space)
    while transformers 5 honours it, which changes the text features and the tumor filter. Context length is 128,
    with the last position reserved for <cls>.
    """
    import torch
    from conch.open_clip_custom import custom_tokenizer
    from tokenizers import Tokenizer, pre_tokenizers

    tokenizer = Tokenizer.from_file(str(custom_tokenizer.TOKENIZER_DIR / custom_tokenizer.DEFAULT_TOKENIZER))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, trim_offsets=True, use_regex=True)
    pad_id = tokenizer.token_to_id("<pad>")
    tokenizer.enable_truncation(max_length=127)
    tokenizer.enable_padding(length=127, pad_id=pad_id, pad_token="<pad>")
    ids = [encoding.ids + [pad_id] for encoding in tokenizer.encode_batch(list(texts))]
    return torch.tensor(ids, dtype=torch.long)


def compute_text_features(cfg: Config) -> np.ndarray:
    """Per class: mean of the L2-normalised CONCH text embeddings of its prompts, re-normalised."""
    import torch

    from .embedding import load_conch

    with progress.step("Loading CONCH for tumor-filter text features"):
        model = load_conch(cfg.models_dir, device="cpu")
    features = []
    with progress.step("Computing tumor-filter text features"), torch.no_grad():
        for cls in CLASSES:
            tokens = _tokenize(PROMPTS[cls])
            emb = model.encode_text(tokens)
            emb = emb / emb.norm(dim=-1, keepdim=True)
            mean = emb.mean(dim=0)
            features.append(mean / mean.norm())
    return torch.stack(features, dim=0).numpy().astype(np.float32)


def load_text_features(cfg: Config, force: bool = False) -> np.ndarray:
    path = text_features_path(cfg)
    if path.is_file() and not force:
        return np.load(path, allow_pickle=False)
    features = compute_text_features(cfg)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".partial.npy")
    np.save(tmp, features, allow_pickle=False)
    tmp.replace(path)
    progress.info(f"Saved tumor-filter text features to {path}")
    return features


def tumor_mask(conch_features: np.ndarray, text_features: np.ndarray) -> np.ndarray:
    """Tiles whose CONCH image features are closest to the 'tumor' prompts (argmax of dot products)."""
    return np.dot(conch_features, text_features.T).argmax(axis=1) == TUMOR_CLASS
