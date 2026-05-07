# WikiArt Style Transfer - Project Report

A comparative study of three latent-variable models for art-style
transfer on `huggan/wikiart`: a vanilla beta-VAE, a disentangled VAE
with separate content (`z_c`) and style (`z_s`) latents, and a
StarGAN. The two VAEs are compared in `04_result_analysis.py`; the
StarGAN is reported as a documented negative result.

For setup and CLI flags see [`README.md`](README.md). This report
focuses on what is actually in `outputs/`.

## 1. Dataset

8 of the 27 WikiArt styles, totalling **38 252** images, restricted
this way for **compute**: the full dataset at 128x128 with three
architectures and Optuna search did not fit our single-GPU budget.
The 8 chosen styles are visually distinct (figurative, expressive,
abstract, graphic, dramatic) so style transfer between any two is
meaningful. Stratified 80/10/10 split: 30 601 / 3 825 / 3 826.

The class distribution is heavily imbalanced (~11x ratio):

| Impressionism | Realism | Baroque | Abs.Exp. | N.Renaissance | Cubism | Pop_Art | Ukiyo_e |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 13 000 | 10 600 | 4 200 | 2 800 | 2 600 | 2 200 | 1 500 | 1 200 |

We do not re-balance; the GAN's auxiliary classifier and the ResNet18
judge in `04` consequently have a strong prior toward Impressionism
and Realism. See `outputs/01_style_counts.png`.

## 2. Architecture choices (`models.py`)

All three models share a 128 -> 4x4 spatial bottleneck so latent
capacities are comparable. Shared blocks: 4x4 stride-2 conv +
GroupNorm + SiLU, plus a single self-attention block at the 4x4
resolution. GroupNorm is preferred over BatchNorm because of the
class imbalance and small batches.

- **Vanilla VAE.** 5-conv encoder -> attention -> linear `mu`/`log_var`
  -> 4-deconv decoder + `Tanh`. Defaults: `latent_dim=384`, `beta=1.0`.
  Loss: `MSE + beta * KL`.
- **Disentangled VAE.** Same backbone, two heads: `z_c` (128-dim,
  content) and `z_s` (32-dim, style). Decoder takes `cat(z_c, z_s)`.
  Asymmetric sizes act as an information bottleneck on style. Two
  classifiers: a positive one on `z_s` (pulls style *into* `z_s`) and
  an adversarial one on `z_c` (pushes style *out of* `z_c`, trained
  alternately as a one-step gradient-reversal proxy).
- **StarGAN.** Direct port of [Choi et al. 2018](https://arxiv.org/abs/1711.09020)
  with WGAN-GP swapping in for LSGAN. Generator concatenates the
  one-hot target style spatially; discriminator has src + cls heads;
  cycle-L1 reconstruction.

## 3. Scripts

| Script | What it does |
|---|---|
| `01_data_acquisition.py` | Stream `huggan/wikiart`, filter to 8 styles, downsize, save to `data/raw/`. Configured by constants at the top of the file (no `argparse`). |
| `02_data_preprocessing.py` | Resize/center-crop to 128x128, save to `data/processed/`, write stratified 80/10/10 splits + manifest. Multiprocessed. |
| `03a_vae_baseline.py` | Optuna TPE over `(latent_dim, beta, lr)` (or `--skip-tune`), then full training with cosine LR, checkpoint resume, `--patience` early stopping. |
| `03b_gan_baseline.py` | Same skeleton; sweeps `(lr_g, lr_d, lambda_cls, lambda_rec)`; trains G/D with `n_critic=5`. |
| `03c_disentangled_vae.py` | Same skeleton; two optimizers (main + adversarial), alternated each step. |
| `04_result_analysis.py` | Loads `vae2_best.pt` + `disvae_best.pt`, trains a ResNet18 style judge, scores recon MSE on the test split + style-transfer accuracy + LPIPS-to-content for the DisVAE. |

> **Stored checkpoints were trained without the LPIPS perceptual term.**
> The training scripts now support `--lpips-weight W`
> (`MSE + beta*KL + W*LPIPS(x_hat, x)`) but the saved weights pre-date
> that change. Retraining with `W=0.5` is the highest-leverage
> remaining task and is left as future work.
>
> Also: the saved `disvae_best.pt` is from an earlier `models.py`
> revision (no `_ResBlockGN` / `_SelfAttention2d`) and is loaded via a
> compatibility shim (`_LegacyDisentangledVAE`) inside `04`. The shim
> can be deleted once DisVAE is retrained.

## 4. Results

| Model | Test recon MSE | Style-transfer acc | Content LPIPS |
|---|---:|---:|---:|
| VAE-2 | **0.0361** | -- | -- |
| Disentangled VAE | 0.0538 | 0.18 | 0.6942 |

Random chance on 8 styles = 0.125; DisVAE is ~5pp above. Per-style:

| Target style | Acc | LPIPS |
|---|---:|---:|
| Impressionism | 0.05 | 0.72 |
| Cubism | 0.00 | 0.69 |
| Ukiyo_e | 0.00 | 0.69 |
| Baroque | 0.00 | 0.68 |
| Pop_Art | 0.09 | 0.70 |
| **Abstract_Expressionism** | **0.76** | 0.68 |
| **Realism** | **0.54** | 0.70 |
| Northern_Renaissance | 0.00 | 0.70 |

Reading the figures in `outputs/`:

- **`qualitative_reconstruction.png` / `vae_recon.png`.** Both VAEs
  recover dominant colour and rough composition but lose all
  high-frequency detail (faces -> smears, brushwork -> flat blocks).
  Standard pure-MSE Gaussian-decoder failure mode; LPIPS would fix it.
- **`recon_mse_compare.png`.** VAE-2 (0.036) edges out DisVAE (0.054)
  on pixel MSE - expected, since DisVAE divides capacity between
  `z_c` / `z_s` and pays for two classifier terms.
- **`vae_samples.png`.** Sampling `z ~ N(0, I)` produces brown/cream
  noise. KL is tiny and *increasing* during training (see
  `vae_training_curves.png`); with `beta=0.256` (Optuna's pick) the
  prior is unusable as a generator.
- **`compare_training_curves.png`.** VAE-2 train/val curves are tight
  and still descending at epoch 30 (undertrained). DisVAE's
  `z_s -> style` classifier plateaus at **0.39** (above chance, but
  weak) while the adversarial `z_c -> style` plateaus at **0.16**
  (near chance). Reading: `z_c` is roughly style-invariant (good),
  `z_s` only weakly captures style (bad). Suggests `style_clf_w`
  should be raised in the next sweep.
- **`qualitative_style_transfer.png` + `style_transfer_per_style.png`.**
  4/8 target styles transfer at exactly 0%. Two textural styles
  (Abs.Exp., Realism) score suspiciously high; without judge
  validation on real images we can't tell whether the model succeeded
  on those classes or the judge is firing on global colour stats.
  LPIPS-to-content is uniform at ~0.69 across all targets, which is
  itself diagnostic - real style transfer would preserve content
  unevenly across "easy" and "hard" target styles. Combined with the
  qualitative panel (transfers don't look like the target style and
  partially erase content), the picture is: the decoder is collapsing
  to a *style prototype* rather than re-rendering content.
- **`gan_translations.png` + `gan_training_curves.png`.** Mode
  collapse. Every row is near-identical regardless of target style;
  within each row the generator outputs only 2-3 distinct images. D
  loss bounces chaotically after epoch 5; cycle-L1 still descends but
  is satisfied trivially by `G(x, c) ≈ x`. Excluded from the
  comparison in `04`.
- **`vae_optuna_scatter.png`.** Best `beta ≈ 0.69`, best
  `lr ≈ 1.4e-4`. `latent_dim` only weakly correlated with val recon -
  capacity is not the bottleneck, reconstruction quality is.

Headline conclusions:

| Question | Answer |
|---|---|
| VAE reconstructs WikiArt? | Yes in MSE, no qualitatively (blurry). |
| VAE generates from prior? | No (KL too small under low `beta`). |
| DisVAE disentangles style/content? | Partially - `z_c` clean, `z_s` weak. |
| DisVAE transfers art style? | **No.** 18% mean acc; 4/8 styles at 0%. |
| StarGAN works? | **No.** Mode-collapsed. |

## 5. Caveats

These results do not let us claim more than the above because:

1. **The judge is not validated on real test images.** Without that
   ceiling, transfer-accuracy numbers are not interpretable in
   absolute terms.
2. **The shipped DisVAE uses a legacy compat shim** in `04` rather
   than the current architecture.
3. **No FID / SSIM / perceptual recon metric** in `summary.csv`
   despite `pytorch-fid` being a dependency.
4. **N=200 transfer pairs**; Ukiyo_e and Pop_Art per-style accuracies
   are based on ~25 samples each.
5. **No latent-space probe** (t-SNE/UMAP of `z_s`/`z_c`) - the
   "DisVAE achieves disentanglement" claim rests on two scalar
   classifier accuracies.

## 6. Future work (priority order)

1. **Retrain VAEs with `--lpips-weight 0.5`.** Single-GPU overnight;
   fixes the dominant blur failure mode.
2. **Validate the style judge** on real test images (one line in
   `04`).
3. **Per-style FID** using `pytorch-fid` (already a dep).
4. **UMAP of `z_s` / `z_c`** colored by ground-truth style - direct
   evidence for or against disentanglement.
5. **Strengthen `style_clf_w`** in DisVAE; current value caps `z_s`
   accuracy at ~0.39.
6. **Retrain DisVAE** on the current architecture so the legacy shim
   can be deleted.
7. **Recover the StarGAN** with a perceptual cycle term + lower D LR,
   or replace with CycleGAN per pair.
8. **LPIPS-to-style-reference** alongside LPIPS-to-content.
9. **Confusion matrix** for style transfer (shows which prototype
   the model collapses onto).
10. **Latent traversal grids** for DisVAE.

Items 1-4 between them would convert this from a documented set of
failure modes into a publishable comparative study.
