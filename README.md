# WikiArt Style Transfer - Disentangled VAE study

Artistic style transfer on WikiArt. Three models are trained; the head-to-head
comparison in `04_result_analysis.py` is between the two VAEs (the GAN is
trained but reported separately as a negative result, see "Scope" below).

| Model | What it does | Status in `04` |
|---|---|---|
| **Vanilla VAE** | Single-latent reconstruction / generation baseline. | Compared (recon MSE + qualitative). |
| **Disentangled VAE** | Two latents per image, `z_c` (content) and `z_s` (style). Style transfer = swap `z_s`. | Compared (recon + style-transfer accuracy + LPIPS). |
| **StarGAN** | One generator conditioned on a one-hot style, WGAN-GP + style-CE + cycle loss. | Trained but **excluded from `04`** - mode collapse, see `outputs/gan_translations.png`. |

### Scope

`04_result_analysis.py` loads `vae2_best.pt` and `disvae_best.pt` and reports
reconstruction MSE on the test split plus style-transfer accuracy / LPIPS for
the DisVAE. The StarGAN baseline (`03b_gan_baseline.py`) is run end-to-end
but its outputs collapse to a near-constant translation regardless of target
style; rather than hide that, we ship `outputs/gan_translations.png` as a
documented failure mode and leave it out of the quantitative comparison. To
re-include it, retrain `03b` with the perceptual / cycle improvements
discussed in the script and uncomment the GAN-loading block in `04`.

## Layout

```
.
|- utils.py                            shared paths, dataset, transforms, checkpoint helpers
|- models.py                           VanillaVAE, DisentangledVAE, StarGAN, configs, losses
|- scripts/
|  |- 01_data_acquisition.py          download (configurable --fraction)
|  |- 02_data_preprocessing.py        resize/crop, splits, manifest
|  |- 03a_vae_baseline.py             HP tune + train + resume
|  |- 03b_gan_baseline.py             HP tune + train + resume
|  |- 03c_disentangled_vae.py         HP tune + train + resume
|  |- 04_result_analysis.py           comparison + metrics
|- data/
|  |- raw/        (created by step 1)
|  |- processed/  (created by step 2)
|- checkpoints/   (<model>_latest.pt + <model>_best.pt per model)
|- outputs/       (figures and CSVs)
|- requirements.txt
```

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

## Running order

Every script is a standalone Python file. Run them from the project root.
All scripts accept `--help` for the full list of flags.

```bash
# 1) Download ~10% of the per-style target (~100 imgs/style x 8 styles)
python scripts/01_data_acquisition.py --fraction 0.10 --n-styles 8

# 2) Resize + center-crop to 128x128 using 4 worker processes
python scripts/02_data_preprocessing.py --image-size 128 --workers 4

# 3) Train the three models. By default each runs its own Optuna search and
#    then trains the best config with resume-from-checkpoint + early
#    stopping. Use --skip-tune + manual HP flags to bypass tuning.
python scripts/03a_vae_baseline.py       --final-epochs 25
python scripts/03b_gan_baseline.py       --final-epochs 20
python scripts/03c_disentangled_vae.py   --final-epochs 30

# Same training, no Optuna search, hyperparameters set by hand:
python scripts/03a_vae_baseline.py --skip-tune \
    --latent-dim 256 --beta 0.6 --lr 1.5e-4 --lpips-weight 0.5 \
    --final-epochs 25 --patience 8

# 4) Load *_best.pt, train a style judge, and compare on the test set.
#    Compares VAE-2 vs DisVAE; StarGAN is excluded (see Scope above).
python scripts/04_result_analysis.py
```

Every 3x script can be **interrupted and re-run**: it detects
`checkpoints/<model>_latest.pt` and continues from the next epoch as long
as the config hasn't changed. Pass `--force-restart` to start over.

### Perceptual loss (VAEs only)

The two VAEs accept `--lpips-weight W` (default `0.5`). When `W > 0` the
training loss is `MSE + beta * KL + W * LPIPS(x_hat, x)` using a frozen
VGG16 LPIPS network. This noticeably reduces the "everything is blurry"
failure mode of pure pixel-MSE VAEs. Pass `--lpips-weight 0` to recover the
old behavior.

### Early stopping

All three training scripts accept `--patience N` (default `8`). Training
stops if the watched validation metric (recon MSE for the VAEs, cycle L1
for the GAN) does not improve for `N` consecutive epochs. Pass
`--patience 0` to disable.

### Manual hyperparameters (`--skip-tune`)

Pass `--skip-tune` plus the model-specific HP flags listed below to bypass
the Optuna search entirely. Useful for re-running a known-good config or
running on machines where you don't want to spend the tuning budget.

| Script | `--skip-tune` HP flags |
|---|---|
| 03a | `--latent-dim`, `--beta`, `--lr`, `--lpips-weight` |
| 03b | `--lr-g`, `--lr-d`, `--lambda-cls`, `--lambda-rec`, `--n-res-blocks` |
| 03c | `--latent-content`, `--latent-style`, `--beta-content`, `--beta-style`, `--style-clf-w`, `--adv-w`, `--lr`, `--lpips-weight` |

## Multiprocessing

- `DataLoader(num_workers=--n-loader-workers)` parallelizes image decoding
  and augmentation.
- `study.optimize(..., n_jobs=--n-jobs)` runs Optuna trials concurrently.
  Keep `--n-jobs 1` on a single GPU; raise it on CPU or multi-GPU.
- `02_data_preprocessing.py` uses `ProcessPoolExecutor(max_workers=--workers)`
  to resize/crop in parallel.

## Key CLI flags

| Script | Flags |
|---|---|
| 01 | `--fraction`, `--n-styles`, `--max-side-px`, `--force`, `--seed` |
| 02 | `--image-size`, `--val-frac`, `--test-frac`, `--workers`, `--force` |
| 3a / 3b / 3c | `--tune-trials`, `--tune-epochs` (or `--tune-steps`), `--tune-subset`, `--final-epochs`, `--batch-size`, `--n-jobs`, `--n-loader-workers`, `--force-restart`, `--skip-tune`, `--patience` (+ model-specific HP flags - see "Manual hyperparameters" above) |
| 04 | `--n-transfer-pairs`, `--clf-epochs`, `--batch-size`, `--n-loader-workers` |

## Dataset

[`huggan/wikiart`](https://huggingface.co/datasets/huggan/wikiart), streamed.
By default we keep eight broad styles:
Impressionism, Cubism, Ukiyo_e, Baroque, Pop_Art, Abstract_Expressionism,
Realism, Northern_Renaissance. Override via `DEFAULT_STYLES` in `utils.py`.

## Outputs

- `outputs/*_optuna_trials.csv` - all tuning trials
- `outputs/*_training_curves.png` - per-model curves
- `outputs/vae_recon.png`, `vae_samples.png`, `disvae_transfer.png`,
  `gan_translations.png`
- `outputs/compare_training_curves.png`, `per_style_metrics.(csv|png)`,
  `qualitative_comparison.png`, `summary.csv`
