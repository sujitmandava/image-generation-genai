"""Step 3a - Vanilla beta-VAE baseline.

1. Short Optuna hyperparameter search (concurrent trials via n_jobs), or
   skip the search entirely with --skip-tune + manual HP flags.
2. Final training with resume from checkpoints/vae_latest.pt and optional
   early stopping (--patience).
3. Save checkpoints/vae_best.pt on val-loss improvement.

The training objective is `MSE + beta * KL + lpips_w * LPIPS(x_hat, x)`.
When `lpips_w > 0` (the default), a frozen VGG16 LPIPS network is used as
a perceptual term so reconstructions are not pure pixel-MSE blur.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import lpips
import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader, Subset

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import VanillaVAE, VAEConfig, vae_loss  # noqa: E402
from utils import (  # noqa: E402
    CHECKPOINTS_DIR, DATA_PROCESSED, OUTPUTS_DIR,
    WikiArtDataset, build_transform, count_parameters,
    get_device, load_checkpoint, save_checkpoint, set_seed, show_grid,
)


def make_loader(ds, bs, shuffle, workers):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                      num_workers=workers, drop_last=shuffle,
                      persistent_workers=workers > 0)


def build_lpips(device, weight: float):
    """Return a frozen LPIPS-VGG module if `weight > 0`, else None."""
    if weight <= 0.0:
        return None
    fn = lpips.LPIPS(net="vgg").to(device).eval()
    for p in fn.parameters():
        p.requires_grad_(False)
    return fn


def run_epoch(model, loader, opt, cfg, device, train: bool,
              lpips_fn=None) -> dict:
    model.train(train)
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "lpips": 0.0, "n": 0}
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            out = model(x)
            losses = vae_loss(out, x, cfg.beta,
                              lpips_fn=lpips_fn, lpips_w=cfg.lpips_w)
            if train:
                opt.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            bs = x.size(0)
            for k in ("loss", "recon", "kl", "lpips"):
                totals[k] += losses[k].item() * bs
            totals["n"] += bs
    return {k: totals[k] / totals["n"]
            for k in ("loss", "recon", "kl", "lpips")}


def cfg_matches(saved: dict | None, current: dict) -> bool:
    """Lenient cfg-equality: every key in `saved` must match `current`.

    New fields added to the dataclass since `saved` was written are tolerated
    so that adding e.g. `lpips_w` doesn't force every old run to restart.
    """
    if not saved:
        return False
    for k, v in saved.items():
        if k not in current or current[k] != v:
            return False
    return True


def tune(train_ds, val_ds, args, device, lpips_fn) -> optuna.Study:
    rng = np.random.default_rng(0)
    tune_train = Subset(train_ds, rng.choice(len(train_ds),
                        min(args.tune_subset, len(train_ds)), replace=False).tolist())
    tune_val = Subset(val_ds, rng.choice(len(val_ds),
                      min(args.tune_subset // 4, len(val_ds)), replace=False).tolist())

    def objective(trial: optuna.Trial) -> float:
        cfg = VAEConfig(
            latent_dim=trial.suggest_categorical("latent_dim", [128, 256, 512]),
            beta=trial.suggest_float("beta", 0.25, 4.0, log=True),
            lr=trial.suggest_float("lr", 5e-5, 5e-4, log=True),
            lpips_w=args.lpips_weight,
            batch_size=args.batch_size,
            epochs=args.tune_epochs,
        )
        model = VanillaVAE(cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
        tl = make_loader(tune_train, cfg.batch_size, True, args.n_loader_workers)
        vl = make_loader(tune_val,   cfg.batch_size, False, args.n_loader_workers)
        for _ in range(cfg.epochs):
            run_epoch(model, tl, opt, cfg, device, train=True,
                      lpips_fn=lpips_fn)
        return run_epoch(model, vl, opt, cfg, device, train=False,
                         lpips_fn=lpips_fn)["recon"]

    study = optuna.create_study(direction="minimize",
                                sampler=TPESampler(seed=42),
                                study_name="vae_baseline")
    study.optimize(objective, n_trials=args.tune_trials, n_jobs=args.n_jobs,
                   show_progress_bar=True)
    print("Best params:", study.best_params,
          "  best recon:", round(study.best_value, 4))
    return study


def plot_trials(study: optuna.Study) -> None:
    df = study.trials_dataframe(attrs=("number", "value", "params"))
    df.to_csv(OUTPUTS_DIR / "vae_optuna_trials.csv", index=False)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    for ax, p in zip(axes, ["beta", "lr", "latent_dim"]):
        col = f"params_{p}"
        if col in df:
            ax.scatter(df[col], df["value"], alpha=0.75)
            ax.set_xlabel(p); ax.set_ylabel("val recon")
            if p in ("beta", "lr"): ax.set_xscale("log")
    fig.suptitle("VAE Optuna trials (lower = better)")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "vae_optuna_scatter.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def load_or_init(best_cfg: VAEConfig, device, force_restart: bool):
    latest = CHECKPOINTS_DIR / "vae_latest.pt"
    model = VanillaVAE(best_cfg).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=best_cfg.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=best_cfg.epochs)
    start_epoch, history, best_val = 1, [], float("inf")

    ckpt = None if force_restart else load_checkpoint(latest, map_location=device)
    if ckpt is not None:
        if not cfg_matches(ckpt.get("cfg"), best_cfg.to_dict()):
            print("Latest checkpoint cfg differs from current best cfg, restarting.")
        else:
            model.load_state_dict(ckpt["model_state"])
            opt.load_state_dict(ckpt["optimizer_states"]["opt"])
            sched.load_state_dict(ckpt["scheduler_states"]["sched"])
            history = ckpt["history"]
            start_epoch = ckpt["epoch"] + 1
            best_val = ckpt.get("best_metric", float("inf"))
            print(f"Resumed from epoch {ckpt['epoch']} (best_val={best_val:.4f})")
    return model, opt, sched, start_epoch, history, best_val


def train_final(model, opt, sched, train_loader, val_loader,
                start_epoch, history, best_val, cfg, device,
                lpips_fn=None, patience: int = 0) -> list[dict]:
    """Train with optional early stopping.

    `patience` is the number of consecutive epochs without val-loss
    improvement after which training stops. `patience <= 0` disables
    early stopping (train for all `cfg.epochs`).
    """
    latest = CHECKPOINTS_DIR / "vae_latest.pt"
    best = CHECKPOINTS_DIR / "vae_best.pt"
    epochs_no_improve = 0
    for epoch in range(start_epoch, cfg.epochs + 1):
        tr = run_epoch(model, train_loader, opt, cfg, device,
                       train=True, lpips_fn=lpips_fn)
        va = run_epoch(model, val_loader,   opt, cfg, device,
                       train=False, lpips_fn=lpips_fn)
        sched.step()
        history.append({"epoch": epoch,
                        **{f"tr_{k}": v for k, v in tr.items()},
                        **{f"va_{k}": v for k, v in va.items()}})
        improved = va["loss"] < best_val
        if improved:
            best_val = va["loss"]
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
        print(f"ep {epoch:02d} | tr loss={tr['loss']:.4f} (recon={tr['recon']:.4f}, "
              f"lpips={tr['lpips']:.4f}) "
              f"| va loss={va['loss']:.4f} (recon={va['recon']:.4f}, "
              f"lpips={va['lpips']:.4f})"
              + ("  [best]" if improved
                 else f"  [no-improve {epochs_no_improve}/{patience}]"
                 if patience > 0 else ""))

        save_checkpoint(
            latest, epoch=epoch, model_state=model.state_dict(),
            optimizer_states={"opt": opt.state_dict()},
            scheduler_states={"sched": sched.state_dict()},
            history=history, best_metric=best_val,
            extra={"cfg": cfg.to_dict()},
        )
        if improved:
            save_checkpoint(
                best, epoch=epoch, model_state=model.state_dict(),
                history=history, best_metric=best_val,
                extra={"cfg": cfg.to_dict()},
            )
        if patience > 0 and epochs_no_improve >= patience:
            print(f"Early stopping at epoch {epoch} "
                  f"(no improvement for {patience} epochs).")
            break
    print("Done. Best val_loss:", best_val)
    return history


def plot_curves_and_samples(history: list[dict], cfg: VAEConfig,
                            val_loader, device) -> None:
    hdf = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(hdf["epoch"], hdf["tr_loss"], label="train")
    axes[0].plot(hdf["epoch"], hdf["va_loss"], label="val")
    axes[0].set_title("Total loss"); axes[0].legend()
    axes[1].plot(hdf["epoch"], hdf["tr_recon"])
    axes[1].plot(hdf["epoch"], hdf["va_recon"])
    axes[1].set_title("Reconstruction MSE")
    axes[2].plot(hdf["epoch"], hdf["tr_kl"])
    axes[2].plot(hdf["epoch"], hdf["va_kl"])
    axes[2].set_title("KL")
    for ax in axes: ax.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "vae_training_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    best_ckpt = load_checkpoint(CHECKPOINTS_DIR / "vae_best.pt", map_location=device)
    model = VanillaVAE(cfg).to(device)
    model.load_state_dict(best_ckpt["model_state"]); model.eval()

    x, _ = next(iter(val_loader))
    x = x[:8].to(device)
    with torch.no_grad():
        out = model(x)
        samples = model.decode(torch.randn(8, cfg.latent_dim, device=device))
    recon_grid = torch.cat([x.cpu(), out["x_hat"].cpu()], dim=0)
    recon_titles = [f"INPUT_{i+1}" for i in range(8)] + [f"RECON_{i+1}" for i in range(8)]
    fig = show_grid(recon_grid, titles=recon_titles,
                    ncols=8,
                    suptitle="VAE reconstruction grid (top: input, bottom: reconstruction)")
    fig.savefig(OUTPUTS_DIR / "vae_recon.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    sample_titles = [f"PRIOR_SAMPLE_{i+1}" for i in range(8)]
    fig = show_grid(samples.cpu(), titles=sample_titles, ncols=8,
                    suptitle="VAE prior samples")
    fig.savefig(OUTPUTS_DIR / "vae_samples.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tune-trials",       type=int, default=12)
    p.add_argument("--tune-epochs",       type=int, default=3)
    p.add_argument("--tune-subset",       type=int, default=2000)
    p.add_argument("--final-epochs",      type=int, default=25)
    p.add_argument("--batch-size",        type=int, default=64)
    p.add_argument("--n-jobs",            type=int, default=1,
                   help="Concurrent Optuna trials (keep 1 on single GPU).")
    p.add_argument("--n-loader-workers",  type=int, default=4)
    p.add_argument("--force-restart",     action="store_true")
    p.add_argument("--patience",          type=int, default=8,
                   help="Early-stop after N epochs without val-loss "
                        "improvement. 0 disables early stopping.")
    p.add_argument("--lpips-weight",      type=float, default=0.5,
                   help="Weight on the LPIPS perceptual term in the VAE "
                        "loss. 0 disables (pure MSE+KL).")

    skip = p.add_argument_group(
        "skip-tune", "Bypass Optuna and train with manual hyperparameters.")
    skip.add_argument("--skip-tune", action="store_true",
                      help="Skip the Optuna search and use the manual HPs below.")
    skip.add_argument("--latent-dim", type=int, default=256)
    skip.add_argument("--beta",       type=float, default=1.0)
    skip.add_argument("--lr",         type=float, default=2e-4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(0)
    device = get_device()
    print("device:", device)

    splits = DATA_PROCESSED / "splits"
    train_ds = WikiArtDataset(splits / "train.csv", root_dir=PROJECT_ROOT,
                              transform=build_transform(128, train=True))
    val_ds   = WikiArtDataset(splits / "val.csv",   root_dir=PROJECT_ROOT,
                              transform=build_transform(128, train=False))
    print(f"train={len(train_ds)}  val={len(val_ds)}")

    lpips_fn = build_lpips(device, args.lpips_weight)
    if lpips_fn is not None:
        print(f"Perceptual loss enabled (lpips_w={args.lpips_weight}).")

    if args.skip_tune:
        print(f"--skip-tune: using manual HPs "
              f"latent_dim={args.latent_dim}, beta={args.beta}, lr={args.lr}.")
        best_cfg = VAEConfig(
            latent_dim=args.latent_dim,
            beta=args.beta,
            lr=args.lr,
            lpips_w=args.lpips_weight,
            batch_size=args.batch_size,
            epochs=args.final_epochs,
        )
    else:
        study = tune(train_ds, val_ds, args, device, lpips_fn)
        plot_trials(study)
        best_cfg = VAEConfig(
            latent_dim=study.best_params["latent_dim"],
            beta=study.best_params["beta"],
            lr=study.best_params["lr"],
            lpips_w=args.lpips_weight,
            batch_size=args.batch_size,
            epochs=args.final_epochs,
        )
    print("Final config:", best_cfg)

    model, opt, sched, start_epoch, history, best_val = load_or_init(
        best_cfg, device, args.force_restart)
    print(f"Training epochs {start_epoch}..{best_cfg.epochs}  "
          f"({count_parameters(model)/1e6:.2f}M params)")

    train_loader = make_loader(train_ds, best_cfg.batch_size, True, args.n_loader_workers)
    val_loader   = make_loader(val_ds,   best_cfg.batch_size, False, args.n_loader_workers)
    history = train_final(model, opt, sched, train_loader, val_loader,
                          start_epoch, history, best_val, best_cfg, device,
                          lpips_fn=lpips_fn, patience=args.patience)
    plot_curves_and_samples(history, best_cfg, val_loader, device)


if __name__ == "__main__":
    main()
