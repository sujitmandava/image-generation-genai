# WikiArt Style Transfer - Project Report

A comparative study of latent-variable models for art-style
transfer on `huggan/wikiart`: a vanilla beta-VAE, a disentangled VAE
with separate content (`z_c`) and style (`z_s`) latents (in two
training variants - pure MSE and MSE + LPIPS perceptual loss), and a
StarGAN. The three VAEs are compared in `04_result_analysis.py`; the
StarGAN is reported as a documented negative result.

For setup and CLI flags see [`README.md`](README.md). This report
focuses on the findings, difficulties, and future work related to this project.

## 1. Dataset

8 of the 27 WikiArt styles, totalling **38 252** images, restricted
this way for **compute**: the full dataset at 128x128 with three
architectures and Optuna search did not fit the single-GPU budget.
The 8 chosen styles are visually distinct (figurative, expressive,
abstract, graphic, dramatic) so style transfer between any two is
meaningful. Stratified 80/10/10 split: 30,601 / 3,825 / 3,826.

The class distribution is heavily imbalanced (~11x ratio):

| Impressionism | Realism | Baroque | Abs.Exp. | N.Renaissance | Cubism | Pop_Art | Ukiyo_e |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 13 000 | 10 600 | 4 200 | 2 800 | 2 600 | 2 200 | 1 500 | 1 200 |

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
  content) and `z_s` (16- or 32-dim, style). Decoder takes
  `cat(z_c, z_s)`. Asymmetric sizes act as an information bottleneck
  on style. Two classifiers: a positive one on `z_s` (pulls style
  *into* `z_s`) and an adversarial one on `z_c` (pushes style *out
  of* `z_c`, trained alternately as a one-step gradient-reversal
  proxy). Two checkpoints are compared:
  - **DisVAE (MSE)**: `MSE + beta_c*KL_c + beta_s*KL_s + style_clf_w*CE - adv_w*CE`.
  - **DisVAE + LPIPS**: same loss + `lpips_w * LPIPS(x_hat, x)` with
    `lpips_w = 0.5`. Trains the modern (`_ResBlockGN` +
    `_SelfAttention2d`) backbone end-to-end; the MSE-only checkpoint
    pre-dates that refactor and is loaded via a small compatibility
    shim in `04`.
- **StarGAN.** Direct port of [Choi et al. 2018](https://arxiv.org/abs/1711.09020)
  with WGAN-GP swapping in for LSGAN. Generator concatenates the
  one-hot target style spatially; discriminator has src + cls heads;
  cycle-L1 reconstruction.

## 3. Scripts

| Script | What it does |
|---|---|
| `01_data_acquisition.py` | Stream `huggan/wikiart`, filter to 8 styles, downsize, save to `data/raw/`.|
| `02_data_preprocessing.py` | Resize/center-crop to 128x128, save to `data/processed/`, write stratified 80/10/10 splits + manifest. Multiprocessed. |
| `03a_vae_baseline.py` | Optuna TPE for hyperparameter tuning, then full training with cosine LR, checkpoint resume, `--patience` early stopping. |
| `03b_gan_baseline.py` | Same skeleton; trains Generator/Discriminator with `n_critic=5`. |
| `03c_disentangled_vae.py` | Same skeleton; two optimizers (main + adversarial), alternated each step. |
| `04_result_analysis.py` | Loads `vae2_best.pt`, `disvae_best.pt`, and `disvae_best2.pt` (the LPIPS-trained DisVAE), trains a ResNet18 style judge, and scores test-split recon MSE for all three plus style-transfer accuracy + LPIPS-to-content for the two DisVAE variants. |

> **Checkpoint provenance.** `vae2_best.pt` and `disvae_best.pt`
> pre-date the LPIPS-aware training loop and were trained with pure
> MSE+KL. The `disvae_best.pt` weights additionally pre-date the
> `_ResBlockGN` / `_SelfAttention2d` refactor in `models.py`, and
> are loaded via a `_LegacyDisentangledVAE` compatibility shim
> inside `04`. The third checkpoint -
> `disvae_best2.pt` - was retrained
> end-to-end on the current architecture with `lpips_w = 0.5` and
> loads through the standard `DisentangledVAE` class, so the shim is
> only needed for the older MSE-only DisVAE.

## 4. Results

Aggregate metrics on the held-out test split (N=200 transfer pairs):

| Model | Test recon MSE | Style-transfer acc | Content LPIPS |
|---|---:|---:|---:|
| VAE-2 | **0.0361** | -- | -- |
| DisVAE (MSE) | 0.0538 | **0.180** | 0.694 |
| DisVAE + LPIPS | 0.0625 | 0.145 | **0.529** |

Random chance on 8 styles = 0.125. Both DisVAE variants land just
above chance on overall transfer accuracy. The LPIPS-trained variant
has higher pixel MSE but ~24% lower content-LPIPS - exactly the
perceptual-vs-pixel trade-off the term is designed to introduce.

Per target style (acc / content-LPIPS, lower is better for LPIPS):

| Target style | DisVAE (MSE) acc | DisVAE (MSE) LPIPS | DisVAE + LPIPS acc | DisVAE + LPIPS LPIPS |
|---|---:|---:|---:|---:|
| Impressionism | 0.05 | 0.72 | **0.30** | **0.53** |
| Cubism | 0.00 | 0.69 | 0.00 | **0.52** |
| Ukiyo_e | 0.00 | 0.69 | 0.00 | **0.53** |
| Baroque | 0.00 | 0.68 | 0.00 | **0.49** |
| Pop_Art | 0.09 | 0.70 | 0.06 | **0.55** |
| **Abstract_Expressionism** | **0.76** | 0.68 | 0.36 | **0.53** |
| **Realism** | **0.54** | 0.70 | 0.50 | **0.54** |
| Northern_Renaissance | 0.00 | 0.70 | 0.00 | **0.52** |

LPIPS preservation is uniformly stronger across every target style
(~0.05-0.20 lower); accuracy redistributes rather than improves -
gains on Impressionism, losses on Abstract_Expressionism.

Reading the figures in `outputs/`:

- **`qualitative_reconstruction.png` / `vae_recon.png`.** All three
  VAEs recover dominant colour and rough composition. VAE-2 and the
  MSE-only DisVAE produce the standard low-frequency blur of a
  pure-MSE Gaussian decoder (faces -> smears, brushwork -> flat
  blocks). The LPIPS-trained DisVAE row visibly recovers more
  texture and edges - notice the canvas weave returning - at the
  cost of slight high-frequency artefacts.
- **`recon_mse_compare.png`.** VAE-2 (0.036) < DisVAE-MSE (0.054) <
  DisVAE-LPIPS (0.063) on pixel MSE. The ordering between the two
  DisVAEs is *expected and not a regression*: optimising perceptual
  similarity allows individual pixels to drift as long as VGG
  feature responses match.
- **`vae_samples.png`.** Sampling `z ~ N(0, I)` produces brown/cream
  noise. KL is tiny and *increasing* during training (see
  `vae_training_curves.png`); with `beta=0.256` (Optuna's pick) the
  prior is unusable as a generator.
- **`compare_training_curves.png`.** VAE-2 train/val curves are
  tight and still descending at epoch 30 (undertrained). DisVAE's
  `z_s -> style` classifier plateaus at **0.39** (above chance, but
  weak) while the adversarial `z_c -> style` plateaus at **0.16**
  (near chance). The DisVAE+LPIPS panel additionally overlays the
  perceptual-loss curve, which descends smoothly throughout
  training - i.e. the LPIPS term does not stall under the same
  Optuna-picked weights as the MSE-only run. Reading: `z_c` is
  roughly style-invariant (good), `z_s` only weakly captures style
  (bad). `style_clf_w` should be raised in the next sweep.
- **`qualitative_style_transfer.png` + `style_transfer_per_style.png`.**
  Both DisVAE variants leave 4-5 of 8 target styles at exactly 0%.
  Two textural styles (Abs.Exp., Realism) score above chance for
  both; the LPIPS variant additionally lifts Impressionism from 5%
  to 30%. Without judge validation on real images these are hard to
  call as real successes vs. global-colour shortcuts. The qualitative
  panel shows the same story as the metrics: the LPIPS row is
  sharper and closer to the content image, while the MSE row is
  blurrier but occasionally swings further toward the target style.
  The decoder still mostly collapses to a *style prototype* rather
  than re-rendering content.
- **`gan_translations.png` + `gan_training_curves.png`.** Mode
  collapse. Every row is near-identical regardless of target style;
  within each row the generator outputs only 2-3 distinct images. D
  loss bounces chaotically after epoch 5; cycle-L1 still descends
  but is satisfied trivially by `G(x, c) ≈ x`. Excluded from the
  comparison in `04`.
- **`vae_optuna_scatter.png`.** Best `beta ≈ 0.69`, best
  `lr ≈ 1.4e-4`. `latent_dim` only weakly correlated with val recon -
  capacity is not the bottleneck, reconstruction quality is.

Conclusions:

| Question | Answer |
|---|---|
| VAE reconstructs WikiArt? | Yes in MSE, no qualitatively (blurry without LPIPS). |
| VAE generates from prior? | No (KL too small under low `beta`). |
| DisVAE disentangles style/content? | Partially - `z_c` clean, `z_s` weak. Can be improved upon with changes to model architecture and training approach.|
| LPIPS helps DisVAE? | **For content preservation, yes** (-24% LPIPS); for transfer accuracy, no (15% vs 18%). |
| DisVAE transfers art style? | **No.** Both variants ~chance overall; only Abs.Exp. and Realism work. |
| StarGAN works? | **No.** Mode-collapsed. |

## 5. Caveats

These results do not support claims beyond the above because:

1. **The judge is not validated on real test images.** Without that
   ceiling, transfer-accuracy numbers are not interpretable in
   absolute terms - the gap between "the model succeeded" and "the
   judge fired on global colour stats" cannot be told apart.
2. **The MSE-only DisVAE uses a legacy compat shim** in `04` rather
   than the current architecture; only the LPIPS variant exercises
   the modern `_ResBlockGN` + `_SelfAttention2d` backbone.
3. **No FID / SSIM / perceptual recon metric** in `summary.csv`
   despite `pytorch-fid` being a dependency.
4. **N=200 transfer pairs**; Ukiyo_e and Pop_Art per-style accuracies
   are based on ~25 samples each.
5. **No latent-space probe** (t-SNE/UMAP of `z_s`/`z_c`) - the
   "DisVAE achieves disentanglement" claim rests on two scalar
   classifier accuracies.
6. **The two DisVAE variants are not architecturally identical.**
   The MSE-only checkpoint is the legacy backbone; the LPIPS
   checkpoint is the modern backbone. Some of the LPIPS-vs-MSE delta
   is therefore confounded with the architectural delta.

## 6. Future work

### A. Sharpen reconstructions (fix the blur)

The dominant qualitative failure mode of both VAEs is low-frequency
blur. The DisVAE+LPIPS run already showed that a perceptual loss
recovers texture. The remaining work is to bring the vanilla VAE to
parity and explore how far perceptual training can be pushed.

1. **Retrain the vanilla VAE with a perceptual loss.** Same recipe
   that worked on the DisVAE; expected to remove most of the VAE-2
   blur in a single overnight run.
2. **Sweep the perceptual weight.** A single value (0.5) was tried;
   a small sweep would tell whether texture recovery has further
   headroom or whether other terms (e.g. total variation) are
   needed.

### B. Optimize learning of the latent space

The disentanglement claim currently rests on two scalar classifier
accuracies. The latents need to be both *measured* better and
*pushed* harder.

3. **Strengthen the disentanglement signal.** The current sweep
   ranges leave the style classifier weak and the adversary too
   gentle - results in a `z_s` that holds only loose style
   information. Re-tune with wider hyperparameter ranges so the two
   latents actually specialize.
4. **Look inside the latent space.** Add a latent-space probe (a 2D
   projection of `z_c` and `z_s` colored by ground-truth style, plus
   a few traversal grids). Either confirms the disentanglement story
   or shows where it breaks.

### C. Better style transfer

Both DisVAE variants flatline near chance because the training
objective never grades the operation that evaluation measures.
Closing that gap is the single biggest accuracy lever left.

6. **Grade the model on the swap, not just the autoencode.** Add a
   style-classification loss on the transferred output and a
   content cycle-consistency term, so the model is rewarded when
   "encode content + encode style + decode" actually produces an
   image of the target style with the source's content.
7. **Make the evaluation trustworthy and informative.** Validate
   the style judge on real test images (sets a ceiling for
   transfer-accuracy numbers), use a mean-style reference instead
   of a single random one, and report a confusion matrix so it's
   visible which target style the model collapses onto.

## 7. Difficulties faced

1. **Hardware and runtime constraints.** End-to-end training was slow on a local MacBook Air, while repeated Colab runs frequently exhausted compute limits before experiments could finish. HPC jobs also stalled intermittently without actionable error logs, which introduced debugging overhead and reduced iteration speed.
2. **Architecture iteration under tight budgets.** Early/simple baselines performed poorly, but testing stronger alternatives required multi-run sweeps that were expensive in both time and compute. Several architecture iterations yielded only marginal gains, making it difficult to separate promising ideas from noise.
3. **Optimization instability in style-transfer objectives.** Adversarial and disentanglement losses were sensitive to weighting and scheduling. Small hyperparameter changes often shifted behavior from weak style learning to unstable training, especially for GAN experiments where mode collapse appeared early.
4. **Class imbalance in the selected WikiArt subset.** The ~11x skew across styles made it harder to learn minority-style signals and likely contributed to uneven per-style transfer quality. This also complicated fair comparison across target styles.