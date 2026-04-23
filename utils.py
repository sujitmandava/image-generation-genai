"""Shared utilities: paths, dataset, transforms, viz, and checkpoint helpers.

Everything that would otherwise be copy-pasted across the six notebooks lives
here. Keep this file small - domain logic (training, loss, evaluation) stays
in the notebooks themselves.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Iterable
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
CHECKPOINTS_DIR = PROJECT_ROOT / "checkpoints"
OUTPUTS_DIR = PROJECT_ROOT / "outputs"

for _d in (DATA_RAW, DATA_PROCESSED, CHECKPOINTS_DIR, OUTPUTS_DIR):
    _d.mkdir(parents=True, exist_ok=True)


DEFAULT_STYLES: tuple[str, ...] = (
    "Impressionism",
    "Cubism",
    "Ukiyo_e",
    "Baroque",
    "Pop_Art",
    "Abstract_Expressionism",
    "Realism",
    "Northern_Renaissance",
)

IMAGE_SIZE = 128
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)


def get_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def build_transform(image_size: int = IMAGE_SIZE, train: bool = False) -> transforms.Compose:
    """Resize + center-crop + optional flip + normalize to [-1, 1]."""
    ops = [transforms.Resize(image_size, antialias=True),
           transforms.CenterCrop(image_size)]
    if train:
        ops.append(transforms.RandomHorizontalFlip(0.5))
    ops += [transforms.ToTensor(), transforms.Normalize(NORM_MEAN, NORM_STD)]
    return transforms.Compose(ops)


def denormalize(x: torch.Tensor) -> torch.Tensor:
    """Undo `build_transform` normalization back into [0, 1]."""
    mean = torch.tensor(NORM_MEAN, device=x.device).view(1, -1, 1, 1)
    std = torch.tensor(NORM_STD, device=x.device).view(1, -1, 1, 1)
    if x.dim() == 3:
        mean, std = mean.squeeze(0), std.squeeze(0)
    return (x * std + mean).clamp(0.0, 1.0)


class WikiArtDataset(Dataset):
    """Image dataset backed by an index CSV.

    Required columns: ``filepath`` (absolute, or relative to ``root_dir``),
    ``label`` (int). Optional: ``style`` (str human-readable).
    """

    def __init__(self, index_csv: str | os.PathLike,
                 root_dir: str | os.PathLike | None = None,
                 transform: transforms.Compose | None = None) -> None:
        self.index = pd.read_csv(index_csv)
        missing = {"filepath", "label"} - set(self.index.columns)
        if missing:
            raise ValueError(f"{index_csv} missing columns {missing}")
        self.root_dir = Path(root_dir) if root_dir is not None else None
        self.transform = transform or build_transform()

    def __len__(self) -> int:
        return len(self.index)

    def _resolve(self, fp: str) -> Path:
        p = Path(fp)
        return p if p.is_absolute() or self.root_dir is None else self.root_dir / p

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, int]:
        row = self.index.iloc[idx]
        with Image.open(self._resolve(row["filepath"])) as im:
            im = im.convert("RGB")
            x = self.transform(im)
        return x, int(row["label"])

    @property
    def labels(self) -> np.ndarray:
        return self.index["label"].to_numpy()

    @property
    def style_names(self) -> list[str]:
        if "style" not in self.index.columns:
            return [str(i) for i in sorted(self.index["label"].unique())]
        pairs = self.index[["label", "style"]].drop_duplicates().sort_values("label")
        return pairs["style"].tolist()


def show_grid(images: torch.Tensor, titles: Iterable[str] | None = None,
              ncols: int = 8, figsize: tuple[float, float] | None = None,
              suptitle: str | None = None):
    if images.dim() != 4:
        raise ValueError(f"Expected NCHW, got {tuple(images.shape)}")
    n = images.size(0)
    nrows = int(np.ceil(n / ncols))
    figsize = figsize or (ncols * 1.6, nrows * 1.6)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize)
    axes = np.array(axes).reshape(-1)
    titles = list(titles) if titles is not None else [None] * n
    imgs = images.detach().cpu()
    if imgs.min() < -0.01:
        imgs = denormalize(imgs)
    for i, ax in enumerate(axes):
        ax.axis("off")
        if i < n:
            ax.imshow(imgs[i].permute(1, 2, 0).numpy())
            if titles[i] is not None:
                ax.set_title(titles[i], fontsize=8)
    if suptitle:
        fig.suptitle(suptitle)
    fig.tight_layout()
    return fig


def count_parameters(module: torch.nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# Checkpoint helpers (save/resume latest)
# ---------------------------------------------------------------------------


def save_checkpoint(path: Path, *, epoch: int, model_state: dict,
                    optimizer_states: dict[str, dict] | None = None,
                    scheduler_states: dict[str, dict] | None = None,
                    history: list[dict] | None = None,
                    best_metric: float | None = None,
                    extra: dict | None = None) -> None:
    """Save a 'latest' checkpoint that downstream training can resume from."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "epoch": epoch,
        "model_state": model_state,
        "optimizer_states": optimizer_states or {},
        "scheduler_states": scheduler_states or {},
        "history": history or [],
        "best_metric": best_metric,
    }
    if extra:
        payload.update(extra)
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(payload, tmp)
    tmp.replace(path)


def load_checkpoint(path: Path, map_location: str | torch.device = "cpu") -> dict | None:
    """Return checkpoint dict, or None if the file does not exist."""
    if not Path(path).exists():
        return None
    return torch.load(path, map_location=map_location)
