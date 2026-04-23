"""Step 1 - Data acquisition.

Downloads a subset of `huggan/wikiart` to `data/raw/<style>/*.jpg` and writes
`data/raw/index.csv`. Configure the run by editing the constants below.
"""

from __future__ import annotations

import json
import os
import sys
from collections import Counter
from pathlib import Path
from utils import DATA_RAW, DEFAULT_STYLES, OUTPUTS_DIR, set_seed
import matplotlib.pyplot as plt
import pandas as pd
from datasets import load_dataset
from dotenv import load_dotenv
from PIL import Image
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Load variables from .env at the project root (HF_TOKEN, etc.).
load_dotenv(PROJECT_ROOT / ".env")
HF_TOKEN = os.environ.get("HF_TOKEN")
if HF_TOKEN is None:
    print("Warning: HF_TOKEN not found in environment or .env - "
          "dataset access will be anonymous.")

# ---------------------------------------------------------------------------
# Configuration - edit these to change how much data is downloaded.
# ---------------------------------------------------------------------------

FRACTION              = 0.10       # Fraction of FULL_SUBSET_PER_STYLE to keep.
N_STYLES              = 8          # Number of styles from utils.DEFAULT_STYLES.
MAX_SIDE_PX           = 512        # Resize so max(w, h) <= MAX_SIDE_PX.
FORCE                 = False      # Re-download even if index.csv already exists.
SEED                  = 42
FULL_SUBSET_PER_STYLE = 1250       # Target images per style at FRACTION == 1.0.

set_seed(SEED)

STYLES = list(DEFAULT_STYLES[:N_STYLES])
PER_STYLE = max(50, int(round(FULL_SUBSET_PER_STYLE * FRACTION)))
STYLE_TO_LABEL = {s: i for i, s in enumerate(STYLES)}

print(f"Fraction:          {FRACTION:.2%}")
print(f"Images per style:  {PER_STYLE}")
print(f"Total target:      {PER_STYLE * N_STYLES}")
print(f"Styles:            {STYLES}")
print(f"Destination:       {DATA_RAW}")

# Download images from Hugging Face.
index_csv = DATA_RAW / "index.csv"
ds = load_dataset("huggan/wikiart", split="train", streaming=True, token=HF_TOKEN)

lbl_feat = ds.features.get("style")
id_to_name = ({i: lbl_feat.int2str(i).replace(" ", "_") for i in range(lbl_feat.num_classes)} if lbl_feat is not None else None)

counts: Counter[str] = Counter()
rows: list[dict] = []
pbar = tqdm(total=PER_STYLE * len(STYLES), desc="Collecting")
for ex in ds:
    name = id_to_name[ex["style"]] if id_to_name is not None else (ex.get("style") or ex.get("genre"))
    if name not in STYLE_TO_LABEL or counts[name] >= PER_STYLE:
        if all(counts[s] >= PER_STYLE for s in STYLES):
            break
        continue
    img = ex["image"]
    if not isinstance(img, Image.Image):
        continue
    out_path = DATA_RAW / name / f"{name}_{counts[name]:05d}.jpg"
    try:
        img = img.convert("RGB")
        if max(img.size) > MAX_SIDE_PX:
            img.thumbnail((MAX_SIDE_PX, MAX_SIDE_PX), Image.LANCZOS)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(out_path, format="JPEG", quality=92)
    except Exception:
        continue
    counts[name] += 1
    rows.append({"filepath": str(out_path.relative_to(PROJECT_ROOT)),
                    "style": name,
                    "label": STYLE_TO_LABEL[name]})
    pbar.update(1)
pbar.close()

df = pd.DataFrame(rows)
df.to_csv(index_csv, index=False)
print(f"Saved {len(df)} rows to {index_csv}")

# EDA: per-style counts bar chart.

counts_by_style = df["style"].value_counts().reindex(STYLES)
fig, ax = plt.subplots(figsize=(7, 3.5))
ax.barh(counts_by_style.index, counts_by_style.values, color="steelblue")
ax.set_title("Images per style")
ax.set_xlabel("count")
fig.tight_layout()
fig.savefig(OUTPUTS_DIR / "01_style_counts.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# EDA: sample paintings per style.
fig, axes = plt.subplots(len(STYLES), 4, figsize=(8, len(STYLES) * 2))
for r, s in enumerate(STYLES):
    sub = df[df["style"] == s]
    sample = sub.sample(min(4, len(sub)), random_state=r)
    for c, (_, row) in enumerate(sample.iterrows()):
        ax = axes[r, c]
        with Image.open(PROJECT_ROOT / row["filepath"]) as im:
            ax.imshow(im)
        ax.axis("off")
        if c == 0:
            ax.set_ylabel(s.replace("_", " "), fontsize=9,
                          rotation=0, ha="right", va="center", labelpad=40)
fig.suptitle("Sample paintings per style", y=1.0)
fig.tight_layout()
fig.savefig(OUTPUTS_DIR / "01_samples.png", dpi=120, bbox_inches="tight")
plt.close(fig)

# Write manifest.
manifest = {
    "data_fraction": FRACTION,
    "images_per_style": PER_STYLE,
    "n_images": int(len(df)),
    "styles": STYLES,
    "label_to_style": {i: s for i, s in enumerate(STYLES)},
    "max_side_px": MAX_SIDE_PX,
}
(DATA_RAW / "manifest.json").write_text(json.dumps(manifest, indent=2))
print("Wrote", DATA_RAW / "manifest.json")
