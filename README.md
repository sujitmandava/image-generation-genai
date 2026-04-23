# WikiArt Style Transfer - VAE vs GAN

Artistic style transfer on WikiArt. Three models are trained and compared:

| Model | What it does |
|---|---|
| **Vanilla VAE** | Learns a single latent that compresses paintings. Reconstruction / generation baseline. |
| **Disentangled VAE** | Two latents per image: `z_c` (content) and `z_s` (style). Style transfer = swap `z_s`. |
| **StarGAN** | Single generator conditioned on a one-hot style vector, trained with WGAN-GP + style classification + cycle loss. |

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

## Running order

Every script is a standalone Python file. Run them from the project root.
All scripts accept `--help` for the full list of flags.

```bash
# 1) Download ~10% of the per-style target (~100 imgs/style x 8 styles)
python scripts/01_data_acquisition.py --fraction 0.10 --n-styles 8

# 2) Resize + center-crop to 128x128 using 4 worker processes
python scripts/02_data_preprocessing.py --image-size 128 --workers 4

# 3) Train the three models. Each runs its own Optuna search, then trains
#    the best config with resume-from-checkpoint.
python scripts/03a_vae_baseline.py       --final-epochs 25
python scripts/03b_gan_baseline.py       --final-epochs 20
python scripts/03c_disentangled_vae.py   --final-epochs 30

# 4) Load all *_best.pt, train a style judge, and compare on the test set
python scripts/04_result_analysis.py
```

Every 3x script can be **interrupted and re-run**: it detects
`checkpoints/<model>_latest.pt` and continues from the next epoch as long
as the config hasn't changed. Pass `--force-restart` to start over.

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
| 3a / 3b / 3c | `--tune-trials`, `--tune-epochs` (or `--tune-steps`), `--tune-subset`, `--final-epochs`, `--batch-size`, `--n-jobs`, `--n-loader-workers`, `--force-restart` |
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
