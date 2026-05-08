# WikiArt Style Transfer - Disentangled VAE study

A small comparative study of latent-variable models for art-style transfer
on the [`huggan/wikiart`](https://huggingface.co/datasets/huggan/wikiart)
dataset. Three architectures are implemented and trained; two are compared
quantitatively, one is reported as a negative result.

For the **full write-up** (architecture rationale, training details,
metrics, and figure-by-figure analysis of `outputs/`), see
[`report.md`](report.md). This file is the operational guide: how to set
up the repo, run the pipeline, and what the CLI flags do.

## Status

| Model | Role | Compared in `04`? |
|---|---|---|
| **Vanilla VAE** | Reconstruction / generation baseline. | Yes - test recon MSE. |
| **Disentangled VAE** | Two-head VAE with `z_c` (content) and `z_s` (style); style transfer = swap `z_s`. | Yes - recon MSE + transfer accuracy + LPIPS. |
| **StarGAN** | One generator conditioned on a one-hot style, WGAN-GP + style-CE + cycle-L1. | **No** - mode-collapsed; see `outputs/gan_translations.png`. |

> **Important:** The model checkpoints currently in `checkpoints/` were
> trained *without* the LPIPS perceptual term. The training scripts now
> support `--lpips-weight` (default `0.5`), but the saved weights pre-date
> that change. Retraining is left as future work; see `report.md` §
> "Future work".

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Git LFS (model checkpoints)

`checkpoints/*.pt` and other binary model files are stored via Git LFS
(see `.gitattributes`). Install LFS once per machine before cloning:

```bash
# macOS:   brew install git-lfs
# Debian:  sudo apt install git-lfs
git lfs install
git clone git@github.com:sujitmandava/image-generation-genai.git
```

If you already cloned without LFS, run `git lfs pull` from inside the
repo to fetch the actual binary blobs (otherwise the `.pt` files will
just be tiny pointer text files).

### HuggingFace token

`01_data_acquisition.py` reads `HF_TOKEN` from a project-root `.env` to
download `huggan/wikiart`. Anonymous access works for most images but
hitting rate limits is more likely without a token. Create one at
[hf.co/settings/tokens](https://huggingface.co/settings/tokens) and put
it in `.env`:

```
HF_TOKEN=hf_...
```

## Layout

```
.
|- utils.py                         shared paths, dataset, transforms, ckpt helpers
|- models.py                        VanillaVAE, DisentangledVAE, StarGAN, configs, losses
|- scripts/
|  |- 01_data_acquisition.py        download from HF (configured by constants at the top of the file)
|  |- 02_data_preprocessing.py      resize / center-crop, stratified train/val/test splits
|  |- 03a_vae_baseline.py           VAE     - Optuna tune (or --skip-tune) + train + early stop
|  |- 03b_gan_baseline.py           StarGAN - Optuna tune (or --skip-tune) + train + early stop
|  |- 03c_disentangled_vae.py       DisVAE  - Optuna tune (or --skip-tune) + train + early stop
|  |- 04_result_analysis.py         load best.pt, train style judge, compute metrics, render figures
|- data/
|  |- raw/                          (created by 01)
|  |- processed/                    (created by 02; train.csv / val.csv / test.csv + manifest.json)
|- checkpoints/                     <model>_best.pt + <model>_latest.pt per model (LFS)
|- outputs/                         figures and CSVs produced by 03a/b/c and 04
|- slurm/                           HPC job scripts
|- report.md                        full write-up (architecture, training, results)
|- requirements.txt
```

## Running order

All scripts are standalone and take `--help`. Run from the project root.

```bash
# 1) Download to data/raw/. NOTE: 01 currently uses module-level constants,
#    not argparse. Edit DOWNLOAD_ALL_STYLES / N_STYLES / MAX_SIDE_PX at the
#    top of the file before running.
python scripts/01_data_acquisition.py

# 2) Resize + center-crop to 128x128 with 4 workers, stratified 80/10/10 splits.
python scripts/02_data_preprocessing.py --image-size 128 --workers 4

# 3) Train. By default each script runs its own Optuna search and then
#    trains the best config with checkpoint resume + early stopping.
python scripts/03a_vae_baseline.py     --final-epochs 25
python scripts/03b_gan_baseline.py     --final-epochs 20
python scripts/03c_disentangled_vae.py --final-epochs 30

# Same training, no Optuna search, hyperparameters set by hand:
python scripts/03a_vae_baseline.py --skip-tune \
    --latent-dim 256 --beta 0.6 --lr 1.5e-4 \
    --lpips-weight 0.5 --final-epochs 25 --patience 8

# 4) Evaluate on the test split. Compares VAE-2 vs DisVAE; the GAN
#    is excluded (mode-collapsed, see report.md).
python scripts/04_result_analysis.py
```

Every 3x script can be **interrupted and re-run**: it detects
`checkpoints/<model>_latest.pt` and continues from the next epoch as
long as the saved config still matches the current one (any keys
present in the saved config must agree; new keys added later are
tolerated). Pass `--force-restart` to start over.

Link to models: https://drive.google.com/drive/folders/1TKWGW4s3fuwEctROYnlkanNOgGZWlpyq?usp=sharing

## CLI flags

### Training scripts (03a / 03b / 03c) - common

| Flag | Default | Notes |
|---|---|---|
| `--final-epochs` | 25 / 20 / 30 | Hard upper bound on training epochs. |
| `--patience` | `8` | Early-stop after N epochs without val-metric improvement. `0` disables. |
| `--batch-size` | 64 / 32 / 64 | |
| `--n-loader-workers` | `4` | DataLoader workers. |
| `--force-restart` | off | Discard `_latest.pt` and start from epoch 1. |
| `--tune-trials` / `--tune-epochs` / `--tune-steps` / `--tune-subset` / `--n-jobs` | varies | Optuna budget. Ignored when `--skip-tune` is set. |
| `--skip-tune` | off | Bypass Optuna; use the manual HP flags below. |

### Training scripts - manual hyperparameters (after `--skip-tune`)

| Script | Manual HP flags |
|---|---|
| 03a | `--latent-dim`, `--beta`, `--lr`, `--lpips-weight` |
| 03b | `--lr-g`, `--lr-d`, `--lambda-cls`, `--lambda-rec`, `--n-res-blocks` |
| 03c | `--latent-content`, `--latent-style`, `--beta-content`, `--beta-style`, `--style-clf-w`, `--adv-w`, `--lr`, `--lpips-weight` |

The two VAE scripts also accept `--lpips-weight W` whether or not
`--skip-tune` is set. The training objective becomes
`MSE + beta * KL + W * LPIPS(x_hat, x)` using a frozen VGG16 LPIPS
network. Pass `0` to disable. **The shipped checkpoints were trained
with `W = 0`.**

### Other scripts

| Script | Flags |
|---|---|
| 01 | (none - edit constants in the file) |
| 02 | `--image-size`, `--val-frac`, `--test-frac`, `--workers`, `--force` |
| 04 | `--n-transfer-pairs`, `--clf-epochs`, `--batch-size`, `--n-loader-workers` |

## Multiprocessing

- `DataLoader(num_workers=--n-loader-workers)` parallelizes image decoding
  and augmentation.
- Optuna runs trials concurrently via `study.optimize(..., n_jobs=--n-jobs)`.
  Keep `--n-jobs 1` on a single GPU.
- `02_data_preprocessing.py` uses `ProcessPoolExecutor(max_workers=--workers)`
  to resize / center-crop in parallel.

## Dataset

8 broad styles from `huggan/wikiart`, totalling **38 252** images:

| Style | Count |
|---|---|
| Impressionism | ~13 000 |
| Realism | ~10 600 |
| Baroque | ~4 200 |
| Abstract_Expressionism | ~2 800 |
| Northern_Renaissance | ~2 600 |
| Cubism | ~2 200 |
| Pop_Art | ~1 500 |
| Ukiyo_e | ~1 200 |

The styles are picked by `DEFAULT_STYLES` in `utils.py`. The
preprocessing step writes a stratified 80 / 10 / 10 train / val / test
split with `random_state=42`: 30 601 / 3 825 / 3 826 images.

The class imbalance is significant (roughly 11x between Impressionism
and Ukiyo_e). See `outputs/01_style_counts.png` and
`report.md` § "Dataset" for the implications.

## Outputs

Files written by each script. See `report.md` for what each one shows
and how to read it.

| Producer | File | Content |
|---|---|---|
| 01 | `outputs/01_style_counts.png` | Per-style image count bar chart. |
| 01 | `outputs/01_samples.png` | Random sample paintings per style. |
| 02 | `outputs/02_sanity.png` | 16 random pre-processed training images. |
| 03a | `outputs/vae_optuna_trials.csv`, `vae_optuna_scatter.png` | HP search log + scatter. |
| 03a | `outputs/vae_training_curves.png`, `vae_recon.png`, `vae_samples.png` | Train curves, reconstructions, prior samples. |
| 03b | `outputs/gan_optuna_trials.csv`, `gan_training_curves.png`, `gan_translations.png` | Same set for the GAN. |
| 03c | `outputs/disvae_optuna_trials.csv`, `disvae_training_curves.png`, `disvae_transfer.png`, `disvae_test_image_transfer_labeled.png` (when `--test-image` is given) | Same set for the DisVAE. |
| 04 | `outputs/compare_training_curves.png` | VAE-2 vs DisVAE training panel. |
| 04 | `outputs/recon_mse_compare.png` | Test-set reconstruction MSE bar chart. |
| 04 | `outputs/qualitative_reconstruction.png` | 6 paintings, original vs each model's reconstruction. |
| 04 | `outputs/qualitative_style_transfer.png` | 6 (content, style ref, DisVAE transfer) triplets. |
| 04 | `outputs/style_transfer_per_style.png` | Per-target-style transfer accuracy + content LPIPS. |
| 04 | `outputs/per_style_metrics.csv` | Same data, tabular. |
| 04 | `outputs/summary.csv` | Final headline metrics. |
