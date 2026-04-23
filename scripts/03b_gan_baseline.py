"""Step 3b - StarGAN-style multi-domain GAN baseline.

One generator G(x, c) conditioned on a one-hot style, one discriminator D
with a WGAN-GP src head and a style-classification head. Same skeleton as
3a: Optuna tune + resume-from-checkpoint final training.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import GANConfig, StarGANDiscriminator, StarGANGenerator  # noqa: E402
from utils import (  # noqa: E402
    CHECKPOINTS_DIR, DATA_PROCESSED, OUTPUTS_DIR,
    WikiArtDataset, build_transform, count_parameters,
    get_device, load_checkpoint, save_checkpoint, set_seed, show_grid,
)


def make_loader(ds, bs, shuffle, workers):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                      num_workers=workers, drop_last=shuffle,
                      persistent_workers=workers > 0)


def one_hot(y: torch.Tensor, n: int) -> torch.Tensor:
    return F.one_hot(y, n).float()


def gradient_penalty(D, real, fake):
    bs = real.size(0)
    eps = torch.rand(bs, 1, 1, 1, device=real.device)
    interp = (eps * real + (1 - eps) * fake).requires_grad_(True)
    src, _ = D(interp)
    grads = torch.autograd.grad(src.sum(), interp,
                                create_graph=True, retain_graph=True)[0]
    grads = grads.view(bs, -1)
    return ((grads.norm(2, dim=1) - 1) ** 2).mean()


def gan_step(G, D, opt_g, opt_d, x, c_real, cfg: GANConfig, device,
             update_g: bool) -> dict:
    bs = x.size(0)
    c_real_oh = one_hot(c_real, cfg.n_styles)
    c_trg = torch.randint(0, cfg.n_styles, (bs,), device=device)
    c_trg_oh = one_hot(c_trg, cfg.n_styles)

    src_real, cls_real = D(x)
    with torch.no_grad():
        x_fake = G(x, c_trg_oh)
    src_fake, _ = D(x_fake)
    d_adv = -src_real.mean() + src_fake.mean()
    d_gp = gradient_penalty(D, x, x_fake)
    d_cls = F.cross_entropy(cls_real, c_real)
    d_loss = d_adv + cfg.lambda_gp * d_gp + cfg.lambda_cls * d_cls
    opt_d.zero_grad(set_to_none=True)
    d_loss.backward()
    opt_d.step()

    g_info: dict = {}
    if update_g:
        x_fake = G(x, c_trg_oh)
        src_fake, cls_fake = D(x_fake)
        x_rec = G(x_fake, c_real_oh)
        g_adv = -src_fake.mean()
        g_cls = F.cross_entropy(cls_fake, c_trg)
        g_rec = F.l1_loss(x, x_rec)
        g_loss = g_adv + cfg.lambda_cls * g_cls + cfg.lambda_rec * g_rec
        opt_g.zero_grad(set_to_none=True)
        g_loss.backward()
        opt_g.step()
        g_info = {"g_loss": g_loss.item(), "g_cls": g_cls.item(),
                  "g_rec": g_rec.item()}
    return {"d_loss": d_loss.item(), "d_cls": d_cls.item(), **g_info}


def tune(train_ds, args, n_styles: int, device) -> optuna.Study:
    rng = np.random.default_rng(0)
    tune_train = Subset(train_ds, rng.choice(len(train_ds),
                        min(args.tune_subset, len(train_ds)), replace=False).tolist())

    def objective(trial: optuna.Trial) -> float:
        cfg = GANConfig(
            n_styles=n_styles, batch_size=args.batch_size,
            lr_g=trial.suggest_float("lr_g", 5e-5, 3e-4, log=True),
            lr_d=trial.suggest_float("lr_d", 5e-5, 3e-4, log=True),
            lambda_cls=trial.suggest_float("lambda_cls", 0.25, 4.0, log=True),
            lambda_rec=trial.suggest_float("lambda_rec", 2.0, 20.0, log=True),
            n_res_blocks=3,
        )
        G = StarGANGenerator(cfg).to(device)
        D = StarGANDiscriminator(cfg).to(device)
        opt_g = torch.optim.Adam(G.parameters(), lr=cfg.lr_g,
                                 betas=(cfg.beta1, cfg.beta2))
        opt_d = torch.optim.Adam(D.parameters(), lr=cfg.lr_d,
                                 betas=(cfg.beta1, cfg.beta2))
        loader = make_loader(tune_train, cfg.batch_size, True, args.n_loader_workers)
        it = itertools.cycle(loader)
        rec_window: list[float] = []
        for step in range(args.tune_steps):
            x, y = next(it)
            info = gan_step(G, D, opt_g, opt_d, x.to(device), y.to(device),
                            cfg, device, update_g=(step + 1) % cfg.n_critic == 0)
            if "g_rec" in info:
                rec_window.append(info["g_rec"])
        return float(np.mean(rec_window[-50:])) if rec_window else 1e9

    study = optuna.create_study(direction="minimize",
                                sampler=TPESampler(seed=42),
                                study_name="gan_baseline")
    study.optimize(objective, n_trials=args.tune_trials, n_jobs=args.n_jobs,
                   show_progress_bar=True)
    print("Best params:", study.best_params,
          "  best mean rec:", round(study.best_value, 4))
    study.trials_dataframe(attrs=("number", "value", "params")).to_csv(
        OUTPUTS_DIR / "gan_optuna_trials.csv", index=False)
    return study


def load_or_init(cfg: GANConfig, device, force_restart: bool):
    latest = CHECKPOINTS_DIR / "gan_latest.pt"
    G = StarGANGenerator(cfg).to(device)
    D = StarGANDiscriminator(cfg).to(device)
    opt_g = torch.optim.Adam(G.parameters(), lr=cfg.lr_g,
                             betas=(cfg.beta1, cfg.beta2))
    opt_d = torch.optim.Adam(D.parameters(), lr=cfg.lr_d,
                             betas=(cfg.beta1, cfg.beta2))
    start_epoch, history, best_rec = 1, [], float("inf")
    ckpt = None if force_restart else load_checkpoint(latest, map_location=device)
    if ckpt is not None and ckpt.get("cfg") == cfg.to_dict():
        G.load_state_dict(ckpt["model_state"]["G"])
        D.load_state_dict(ckpt["model_state"]["D"])
        opt_g.load_state_dict(ckpt["optimizer_states"]["g"])
        opt_d.load_state_dict(ckpt["optimizer_states"]["d"])
        start_epoch = ckpt["epoch"] + 1
        history = ckpt["history"]
        best_rec = ckpt.get("best_metric", float("inf"))
        print(f"Resumed from epoch {ckpt['epoch']} (best_rec={best_rec:.4f})")
    elif ckpt is not None:
        print("Config changed, discarding latest checkpoint.")
    return G, D, opt_g, opt_d, start_epoch, history, best_rec


def train_final(G, D, opt_g, opt_d, loader, start_epoch, history, best_rec,
                cfg: GANConfig, device) -> list[dict]:
    latest = CHECKPOINTS_DIR / "gan_latest.pt"
    best   = CHECKPOINTS_DIR / "gan_best.pt"
    for epoch in range(start_epoch, cfg.epochs + 1):
        G.train(); D.train()
        step = 0
        sums = {"d_loss": 0.0, "g_loss": 0.0, "g_rec": 0.0, "n_g": 0, "n_d": 0}
        for x, y in tqdm(loader, desc=f"epoch {epoch}/{cfg.epochs}", leave=False):
            step += 1
            info = gan_step(G, D, opt_g, opt_d, x.to(device), y.to(device),
                            cfg, device, update_g=(step % cfg.n_critic == 0))
            sums["d_loss"] += info["d_loss"]; sums["n_d"] += 1
            if "g_loss" in info:
                sums["g_loss"] += info["g_loss"]
                sums["g_rec"]  += info["g_rec"]
                sums["n_g"] += 1

        d_avg = sums["d_loss"] / max(1, sums["n_d"])
        g_avg = sums["g_loss"] / max(1, sums["n_g"])
        r_avg = sums["g_rec"]  / max(1, sums["n_g"])
        history.append({"epoch": epoch, "d_loss": d_avg, "g_loss": g_avg, "g_rec": r_avg})
        improved = r_avg < best_rec
        if improved:
            best_rec = r_avg
        print(f"ep {epoch:02d} | d={d_avg:.3f}  g={g_avg:.3f}  rec={r_avg:.4f}"
              + ("  [best]" if improved else ""))

        save_checkpoint(
            latest, epoch=epoch,
            model_state={"G": G.state_dict(), "D": D.state_dict()},
            optimizer_states={"g": opt_g.state_dict(), "d": opt_d.state_dict()},
            history=history, best_metric=best_rec, extra={"cfg": cfg.to_dict()},
        )
        if improved:
            save_checkpoint(
                best, epoch=epoch,
                model_state={"G": G.state_dict(), "D": D.state_dict()},
                history=history, best_metric=best_rec, extra={"cfg": cfg.to_dict()},
            )
    print("Done. Best rec:", best_rec)
    return history


def plot_and_translate(history, cfg: GANConfig, val_ds, styles, device) -> None:
    hdf = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(hdf["epoch"], hdf["d_loss"]); axes[0].set_title("D loss")
    axes[1].plot(hdf["epoch"], hdf["g_loss"]); axes[1].set_title("G loss")
    axes[2].plot(hdf["epoch"], hdf["g_rec"]);  axes[2].set_title("Cycle L1")
    for ax in axes: ax.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "gan_training_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    best_ckpt = load_checkpoint(CHECKPOINTS_DIR / "gan_best.pt", map_location=device)
    G = StarGANGenerator(cfg).to(device)
    G.load_state_dict(best_ckpt["model_state"]["G"]); G.eval()

    loader = make_loader(val_ds, 8, True, 0)
    x, _ = next(iter(loader))
    x = x.to(device)
    grids = [x.cpu()]
    with torch.no_grad():
        for s in range(cfg.n_styles):
            c = F.one_hot(torch.full((x.size(0),), s, device=device),
                          cfg.n_styles).float()
            grids.append(G(x, c).cpu())
    grid = torch.cat(grids, dim=0)
    fig = show_grid(grid, ncols=8,
                    suptitle=f"GAN rows: [input] + {styles}")
    fig.savefig(OUTPUTS_DIR / "gan_translations.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tune-trials",      type=int, default=8)
    p.add_argument("--tune-steps",       type=int, default=400)
    p.add_argument("--tune-subset",      type=int, default=2000)
    p.add_argument("--final-epochs",     type=int, default=20)
    p.add_argument("--batch-size",       type=int, default=32)
    p.add_argument("--n-jobs",           type=int, default=1)
    p.add_argument("--n-loader-workers", type=int, default=4)
    p.add_argument("--force-restart",    action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(0)
    device = get_device()
    manifest = json.loads((DATA_PROCESSED / "manifest.json").read_text())
    styles = manifest["styles"]
    n_styles = len(styles)
    print("device:", device, " styles:", n_styles)

    splits = DATA_PROCESSED / "splits"
    train_ds = WikiArtDataset(splits / "train.csv", root_dir=PROJECT_ROOT,
                              transform=build_transform(128, train=True))
    val_ds   = WikiArtDataset(splits / "val.csv",   root_dir=PROJECT_ROOT,
                              transform=build_transform(128, train=False))

    study = tune(train_ds, args, n_styles, device)

    best_cfg = GANConfig(
        n_styles=n_styles, batch_size=args.batch_size,
        epochs=args.final_epochs,
        lr_g=study.best_params["lr_g"],
        lr_d=study.best_params["lr_d"],
        lambda_cls=study.best_params["lambda_cls"],
        lambda_rec=study.best_params["lambda_rec"],
    )
    print("Final cfg:", best_cfg)

    G, D, opt_g, opt_d, start_epoch, history, best_rec = load_or_init(
        best_cfg, device, args.force_restart)
    print(f"Training epochs {start_epoch}..{best_cfg.epochs}  "
          f"(G={count_parameters(G)/1e6:.2f}M, D={count_parameters(D)/1e6:.2f}M)")

    loader = make_loader(train_ds, best_cfg.batch_size, True, args.n_loader_workers)
    history = train_final(G, D, opt_g, opt_d, loader, start_epoch, history,
                          best_rec, best_cfg, device)
    plot_and_translate(history, best_cfg, val_ds, styles, device)


if __name__ == "__main__":
    main()
