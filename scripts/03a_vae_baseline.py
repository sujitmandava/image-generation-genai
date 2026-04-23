"""Step 3a - Vanilla beta-VAE baseline.

1. Short Optuna hyperparameter search (concurrent trials via n_jobs).
2. Final training with resume from checkpoints/vae_latest.pt.
3. Save checkpoints/vae_best.pt on val-loss improvement.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

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


def run_epoch(model, loader, opt, cfg, device, train: bool) -> dict:
    model.train(train)
    totals = {"loss": 0.0, "recon": 0.0, "kl": 0.0, "n": 0}
    ctx = torch.enable_grad() if train else torch.no_grad()
    with ctx:
        for x, _ in loader:
            x = x.to(device, non_blocking=True)
            out = model(x)
            losses = vae_loss(out, x, cfg.beta)
            if train:
                opt.zero_grad(set_to_none=True)
                losses["loss"].backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                opt.step()
            bs = x.size(0)
            for k in ("loss", "recon", "kl"):
                totals[k] += losses[k].item() * bs
            totals["n"] += bs
    return {k: totals[k] / totals["n"] for k in ("loss", "recon", "kl")}


def tune(train_ds, val_ds, args, device) -> optuna.Study:
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
            batch_size=args.batch_size,
            epochs=args.tune_epochs,
        )
        model = VanillaVAE(cfg).to(device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg.lr, weight_decay=1e-5)
        tl = make_loader(tune_train, cfg.batch_size, True, args.n_loader_workers)
        vl = make_loader(tune_val,   cfg.batch_size, False, args.n_loader_workers)
        for _ in range(cfg.epochs):
            run_epoch(model, tl, opt, cfg, device, train=True)
        return run_epoch(model, vl, opt, cfg, device, train=False)["recon"]

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
        if ckpt.get("cfg") != best_cfg.to_dict():
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
                start_epoch, history, best_val, cfg, device) -> list[dict]:
    latest = CHECKPOINTS_DIR / "vae_latest.pt"
    best = CHECKPOINTS_DIR / "vae_best.pt"
    for epoch in range(start_epoch, cfg.epochs + 1):
        tr = run_epoch(model, train_loader, opt, cfg, device, train=True)
        va = run_epoch(model, val_loader,   opt, cfg, device, train=False)
        sched.step()
        history.append({"epoch": epoch,
                        **{f"tr_{k}": v for k, v in tr.items()},
                        **{f"va_{k}": v for k, v in va.items()}})
        improved = va["loss"] < best_val
        if improved:
            best_val = va["loss"]
        print(f"ep {epoch:02d} | tr loss={tr['loss']:.4f} (recon={tr['recon']:.4f}) "
              f"| va loss={va['loss']:.4f} (recon={va['recon']:.4f})"
              + ("  [best]" if improved else ""))

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
    fig = show_grid(torch.cat([x.cpu(), out["x_hat"].cpu()], dim=0),
                    ncols=8,
                    suptitle="VAE top=input | bottom=reconstruction")
    fig.savefig(OUTPUTS_DIR / "vae_recon.png", dpi=120, bbox_inches="tight")
    plt.close(fig)
    fig = show_grid(samples.cpu(), ncols=8, suptitle="VAE prior samples")
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

    study = tune(train_ds, val_ds, args, device)
    plot_trials(study)

    best_cfg = VAEConfig(
        latent_dim=study.best_params["latent_dim"],
        beta=study.best_params["beta"],
        lr=study.best_params["lr"],
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
                          start_epoch, history, best_val, best_cfg, device)
    plot_curves_and_samples(history, best_cfg, val_loader, device)


if __name__ == "__main__":
    main()
