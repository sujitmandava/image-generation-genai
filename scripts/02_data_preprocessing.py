"""Step 2 - Data preprocessing.

Reads `data/raw/` (produced by step 1), resizes + center-crops every image
to IMAGE_SIZE using multiple worker processes, writes the result to
`data/processed/`, and produces stratified train/val/test CSVs.
"""

from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd
import torch
from PIL import Image
from sklearn.model_selection import train_test_split
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from utils import (  # noqa: E402
    DATA_PROCESSED, DATA_RAW, OUTPUTS_DIR,
    WikiArtDataset, build_transform, set_seed, show_grid,
)


def _process_one(task: tuple[str, str, int]) -> tuple[str, bool]:
    src, dst, size = task
    try:
        with Image.open(src) as im:
            im = im.convert("RGB")
            short = min(im.size)
            scale = size / short
            new_size = (int(round(im.size[0] * scale)),
                        int(round(im.size[1] * scale)))
            im = im.resize(new_size, Image.LANCZOS)
            left = (im.size[0] - size) // 2
            top  = (im.size[1] - size) // 2
            im = im.crop((left, top, left + size, top + size))
            Path(dst).parent.mkdir(parents=True, exist_ok=True)
            im.save(dst, format="JPEG", quality=92)
        return dst, True
    except Exception as e:
        return f"{src}: {e}", False


def preprocess_all(raw_index: pd.DataFrame, size: int, workers: int,
                   force: bool) -> pd.DataFrame:
    processed_rows = []
    tasks: list[tuple[str, str, int]] = []
    for _, row in raw_index.iterrows():
        src = PROJECT_ROOT / row["filepath"]
        name = Path(row["filepath"]).name
        dst = DATA_PROCESSED / row["style"] / name
        processed_rows.append({
            "filepath": str(dst.relative_to(PROJECT_ROOT)),
            "style": row["style"],
            "label": int(row["label"]),
        })
        if dst.exists() and not force:
            continue
        tasks.append((str(src), str(dst), size))

    if tasks:
        with ProcessPoolExecutor(max_workers=workers) as ex:
            desc = f"preprocess ({workers} workers)"
            for res, ok in tqdm(ex.map(_process_one, tasks),
                                total=len(tasks), desc=desc):
                if not ok:
                    print("  WARN:", res)
    else:
        print("All images already preprocessed.")
    return pd.DataFrame(processed_rows)


def make_splits(index: pd.DataFrame, val_frac: float, test_frac: float,
                splits_dir: Path) -> dict[str, pd.DataFrame]:
    splits_dir.mkdir(exist_ok=True)
    train_df, temp_df = train_test_split(
        index, test_size=val_frac + test_frac,
        stratify=index["label"], random_state=42,
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=test_frac / (val_frac + test_frac),
        stratify=temp_df["label"], random_state=42,
    )
    splits = {"train": train_df, "val": val_df, "test": test_df}
    for name, sub in splits.items():
        out = splits_dir / f"{name}.csv"
        sub.reset_index(drop=True).to_csv(out, index=False)
        print(f"{name:5s}  n={len(sub):5d}  -> {out}")
    return splits


def sanity_grid(image_size: int, splits_dir: Path) -> None:
    ds = WikiArtDataset(splits_dir / "train.csv", root_dir=PROJECT_ROOT,
                        transform=build_transform(image_size, train=True))
    xs, ys = zip(*(ds[i] for i in range(16)))
    fig = show_grid(torch.stack(xs),
                    titles=[ds.style_names[y] for y in ys],
                    ncols=8, suptitle="Sanity check: preprocessed training images")
    fig.savefig(OUTPUTS_DIR / "02_sanity.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--image-size", type=int, default=128)
    p.add_argument("--val-frac",   type=float, default=0.10)
    p.add_argument("--test-frac",  type=float, default=0.10)
    p.add_argument("--workers",    type=int, default=4)
    p.add_argument("--force",      action="store_true",
                   help="Re-process images even if they already exist.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(42)

    raw_manifest = json.loads((DATA_RAW / "manifest.json").read_text())
    styles = raw_manifest["styles"]
    print(f"raw: {raw_manifest['n_images']} images, {len(styles)} styles")

    raw_index = pd.read_csv(DATA_RAW / "index.csv")
    processed_index = preprocess_all(raw_index, args.image_size,
                                     args.workers, args.force)
    processed_index.to_csv(DATA_PROCESSED / "index.csv", index=False)
    print(f"Processed index: {len(processed_index)} rows -> "
          f"{DATA_PROCESSED / 'index.csv'}")

    splits_dir = DATA_PROCESSED / "splits"
    splits = make_splits(processed_index, args.val_frac, args.test_frac,
                         splits_dir)
    summary = pd.DataFrame({k: v["style"].value_counts()
                            for k, v in splits.items()}).fillna(0).astype(int)
    print(summary)

    sanity_grid(args.image_size, splits_dir)

    manifest = {
        "image_size": args.image_size,
        "styles": styles,
        "label_to_style": raw_manifest["label_to_style"],
        "n_train": int(len(splits["train"])),
        "n_val":   int(len(splits["val"])),
        "n_test":  int(len(splits["test"])),
    }
    (DATA_PROCESSED / "manifest.json").write_text(json.dumps(manifest, indent=2))
    print("Wrote", DATA_PROCESSED / "manifest.json")


if __name__ == "__main__":
    main()
