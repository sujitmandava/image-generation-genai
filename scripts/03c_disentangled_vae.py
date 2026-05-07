"""Step 3c - Disentangled VAE (content + style) baseline.

Two-head VAE with a style classifier on z_s and an adversarial classifier on
z_c. Same skeleton as 3a / 3b: Optuna HP tune + resume-from-checkpoint.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import optuna
import pandas as pd
import torch
import torch.nn.functional as F
from PIL import Image
from optuna.samplers import TPESampler
from torch.utils.data import DataLoader, Subset
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import DisentangledVAE, DisVAEConfig, disvae_loss  # noqa: E402
from utils import (  # noqa: E402
    CHECKPOINTS_DIR, DATA_PROCESSED, OUTPUTS_DIR,
    WikiArtDataset, build_transform, count_parameters,
    get_device, load_checkpoint, save_checkpoint, set_seed, show_grid,
)


def make_loader(ds, bs, shuffle, workers):
    return DataLoader(ds, batch_size=bs, shuffle=shuffle,
                      num_workers=workers, drop_last=shuffle,
                      persistent_workers=workers > 0)


def split_params(model: DisentangledVAE):
    adv = list(model.adv_clf.parameters())
    adv_ids = {id(p) for p in adv}
    main = [p for p in model.parameters() if id(p) not in adv_ids]
    return main, adv


def train_step(model, x, y, cfg, opt_main, opt_adv) -> dict:
    out = model(x)
    losses = disvae_loss(out, x, y, cfg)
    opt_main.zero_grad(set_to_none=True)
    losses["loss"].backward()
    main_params = [p for g in opt_main.param_groups for p in g["params"]]
    torch.nn.utils.clip_grad_norm_(main_params, 5.0)
    opt_main.step()

    with torch.no_grad():
        z_c = model.encode(x)["mu_c"].detach()
    adv_logits = model.adv_clf(z_c)
    adv_loss = F.cross_entropy(adv_logits, y)
    opt_adv.zero_grad(set_to_none=True)
    adv_loss.backward()
    opt_adv.step()
    return {"loss": losses["loss"].item(), "recon": losses["recon"].item(),
            "style_ce": losses["style_ce"].item(), "adv_ce": adv_loss.item()}


@torch.no_grad()
def eval_epoch(model, loader, device) -> dict:
    model.eval()
    tot = {"recon": 0.0, "sty_hit": 0, "adv_hit": 0, "n": 0}
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        out = model(x)
        tot["recon"] += F.mse_loss(out["x_hat"], x, reduction="mean").item() * x.size(0)
        tot["sty_hit"] += (out["style_logits"].argmax(1) == y).sum().item()
        tot["adv_hit"] += (out["adv_logits"].argmax(1) == y).sum().item()
        tot["n"] += x.size(0)
    return {"recon": tot["recon"] / tot["n"],
            "sty_acc": tot["sty_hit"] / tot["n"],
            "adv_acc": tot["adv_hit"] / tot["n"]}


def tune(train_ds, val_ds, args, n_styles: int, device) -> optuna.Study:
    rng = np.random.default_rng(0)
    tune_train = Subset(train_ds, rng.choice(len(train_ds),
                        min(args.tune_subset, len(train_ds)), replace=False).tolist())
    tune_val = Subset(val_ds, rng.choice(len(val_ds),
                      min(args.tune_subset // 4, len(val_ds)), replace=False).tolist())
    chance = 1.0 / n_styles

    def short_train(cfg: DisVAEConfig) -> dict:
        model = DisentangledVAE(cfg).to(device)
        main_ps, adv_ps = split_params(model)
        opt_main = torch.optim.AdamW(main_ps, lr=cfg.lr, weight_decay=1e-5)
        opt_adv  = torch.optim.AdamW(adv_ps,  lr=cfg.lr, weight_decay=1e-5)
        tl = make_loader(tune_train, cfg.batch_size, True, args.n_loader_workers)
        vl = make_loader(tune_val,   cfg.batch_size, False, args.n_loader_workers)
        for _ in range(cfg.epochs):
            model.train()
            for x, y in tl:
                train_step(model, x.to(device), y.to(device), cfg, opt_main, opt_adv)
        return eval_epoch(model, vl, device)

    def objective(trial: optuna.Trial) -> float:
        cfg = DisVAEConfig(
            n_styles=n_styles, batch_size=args.batch_size, epochs=args.tune_epochs,
            latent_content=trial.suggest_categorical("latent_content", [64, 128, 192]),
            latent_style  =trial.suggest_categorical("latent_style",   [16, 32, 64]),
            beta_content =trial.suggest_float("beta_content", 0.5, 4.0, log=True),
            beta_style   =trial.suggest_float("beta_style",   0.25, 2.0, log=True),
            style_clf_w  =trial.suggest_float("style_clf_w",  0.5, 3.0, log=True),
            adv_w        =trial.suggest_float("adv_w",        0.05, 0.5, log=True),
            lr           =trial.suggest_float("lr",           5e-5, 5e-4, log=True),
        )
        r = short_train(cfg)
        trial.set_user_attr("val_recon",   r["recon"])
        trial.set_user_attr("val_sty_acc", r["sty_acc"])
        trial.set_user_attr("val_adv_acc", r["adv_acc"])
        return r["recon"] + 0.5 * max(0.0, r["adv_acc"] - chance) - 0.05 * (r["sty_acc"] - chance)

    study = optuna.create_study(direction="minimize",
                                sampler=TPESampler(seed=42),
                                study_name="disvae_baseline")
    study.optimize(objective, n_trials=args.tune_trials, n_jobs=args.n_jobs,
                   show_progress_bar=True)
    print("Best params:", study.best_params)
    print("Best value:", round(study.best_value, 4))
    ua = study.best_trial.user_attrs
    print(f"  val_recon={ua['val_recon']:.4f}  val_sty_acc={ua['val_sty_acc']:.4f}  "
          f"val_adv_acc={ua['val_adv_acc']:.4f}")
    study.trials_dataframe(attrs=("number", "value", "params", "user_attrs")).to_csv(
        OUTPUTS_DIR / "disvae_optuna_trials.csv", index=False)
    return study


def load_or_init(cfg: DisVAEConfig, device, force_restart: bool):
    latest = CHECKPOINTS_DIR / "disvae_latest.pt"
    model = DisentangledVAE(cfg).to(device)
    main_ps, adv_ps = split_params(model)
    opt_main = torch.optim.AdamW(main_ps, lr=cfg.lr, weight_decay=1e-5)
    opt_adv  = torch.optim.AdamW(adv_ps,  lr=cfg.lr, weight_decay=1e-5)
    start_epoch, history, best_val = 1, [], float("inf")

    ckpt = None if force_restart else load_checkpoint(latest, map_location=device)
    if ckpt is not None and ckpt.get("cfg") == cfg.to_dict():
        model.load_state_dict(ckpt["model_state"])
        opt_main.load_state_dict(ckpt["optimizer_states"]["main"])
        opt_adv.load_state_dict(ckpt["optimizer_states"]["adv"])
        history = ckpt["history"]
        start_epoch = ckpt["epoch"] + 1
        best_val = ckpt.get("best_metric", float("inf"))
        print(f"Resumed from epoch {ckpt['epoch']} (best_val={best_val:.4f})")
    elif ckpt is not None:
        print("Config changed, discarding latest checkpoint.")
    return model, opt_main, opt_adv, start_epoch, history, best_val


def train_final(model, opt_main, opt_adv, train_loader, val_loader,
                start_epoch, history, best_val, cfg: DisVAEConfig,
                device) -> list[dict]:
    latest = CHECKPOINTS_DIR / "disvae_latest.pt"
    best   = CHECKPOINTS_DIR / "disvae_best.pt"
    for epoch in range(start_epoch, cfg.epochs + 1):
        model.train()
        sums = {"loss": 0.0, "recon": 0.0, "style_ce": 0.0, "adv_ce": 0.0, "n": 0}
        for x, y in tqdm(train_loader, desc=f"epoch {epoch}", leave=False):
            x, y = x.to(device), y.to(device)
            info = train_step(model, x, y, cfg, opt_main, opt_adv)
            bs = x.size(0)
            for k in ("loss", "recon", "style_ce", "adv_ce"):
                sums[k] += info[k] * bs
            sums["n"] += bs
        tr = {k: sums[k] / sums["n"] for k in ("loss", "recon", "style_ce", "adv_ce")}
        va = eval_epoch(model, val_loader, device)
        history.append({"epoch": epoch,
                        **{f"tr_{k}": v for k, v in tr.items()},
                        **{f"va_{k}": v for k, v in va.items()}})
        improved = va["recon"] < best_val
        if improved:
            best_val = va["recon"]
        print(f"ep {epoch:02d} | tr recon={tr['recon']:.4f}  "
              f"sty_ce={tr['style_ce']:.3f}  adv_ce={tr['adv_ce']:.3f} | "
              f"va recon={va['recon']:.4f}  sty_acc={va['sty_acc']:.2f}  "
              f"adv_acc={va['adv_acc']:.2f}" + ("  [best]" if improved else ""))

        save_checkpoint(
            latest, epoch=epoch, model_state=model.state_dict(),
            optimizer_states={"main": opt_main.state_dict(), "adv": opt_adv.state_dict()},
            history=history, best_metric=best_val, extra={"cfg": cfg.to_dict()},
        )
        if improved:
            save_checkpoint(
                best, epoch=epoch, model_state=model.state_dict(),
                history=history, best_metric=best_val,
                extra={"cfg": cfg.to_dict()},
            )
    print("Done. Best val_recon:", best_val)
    return history


def _load_eval_image(path: Path, image_size: int) -> torch.Tensor:
    tfm = build_transform(image_size, train=False)
    with Image.open(path) as im:
        return tfm(im.convert("RGB"))


def plot_and_transfer(history, cfg: DisVAEConfig, val_loader, n_styles: int,
                      device, test_image: Path | None,
                      style_image: Path | None) -> None:
    hdf = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 3, figsize=(12, 3.5))
    axes[0].plot(hdf["epoch"], hdf["tr_recon"], label="train")
    axes[0].plot(hdf["epoch"], hdf["va_recon"], label="val")
    axes[0].set_title("Reconstruction MSE"); axes[0].legend()
    axes[1].plot(hdf["epoch"], hdf["tr_style_ce"], label="style CE")
    axes[1].plot(hdf["epoch"], hdf["tr_adv_ce"],   label="adv CE")
    axes[1].set_title("Classifier losses"); axes[1].legend()
    axes[2].plot(hdf["epoch"], hdf["va_sty_acc"], label="z_s -> style")
    axes[2].plot(hdf["epoch"], hdf["va_adv_acc"], label="z_c -> style (adv)")
    axes[2].axhline(1 / n_styles, color="gray", linestyle="--", label="chance")
    axes[2].set_title("Val classifier accuracy"); axes[2].legend()
    axes[2].set_ylim(0, 1)
    for ax in axes: ax.set_xlabel("epoch")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "disvae_training_curves.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    best_ckpt = load_checkpoint(CHECKPOINTS_DIR / "disvae_best.pt", map_location=device)
    model = DisentangledVAE(cfg).to(device)
    model.load_state_dict(best_ckpt["model_state"]); model.eval()

    x, _ = next(iter(val_loader))
    content = x[:4].to(device); style = x[4:8].to(device)
    with torch.no_grad():
        rec = model(content)["x_hat"]
        transfer = model.transfer(content, style)
    fig = show_grid(torch.cat([content.cpu(), rec.cpu(), style.cpu(), transfer.cpu()], dim=0),
                    ncols=4,
                    suptitle="DisVAE rows: content | recon | style | content+style")
    fig.savefig(OUTPUTS_DIR / "disvae_transfer.png", dpi=120, bbox_inches="tight")
    plt.close(fig)

    if test_image is not None:
        style_source = style_image if style_image is not None else test_image
        test_x = _load_eval_image(test_image, cfg.image_size).unsqueeze(0).to(device)
        style_x = _load_eval_image(style_source, cfg.image_size).unsqueeze(0).to(device)
        with torch.no_grad():
            test_recon = model(test_x)["x_hat"]
            test_transfer = model.transfer(test_x, style_x)
        labeled = torch.cat([test_x.cpu(), style_x.cpu(), test_recon.cpu(),
                             test_transfer.cpu()], dim=0)
        titles = [
            "TEST_IMAGE (content)",
            "STYLE_REFERENCE",
            "TEST_RECONSTRUCTION",
            "STYLE_TRANSFER_OUTPUT",
        ]
        fig = show_grid(
            labeled, titles=titles, ncols=4,
            suptitle="DisVAE single-image style transfer (left to right labeled)"
        )
        fig.savefig(OUTPUTS_DIR / "disvae_test_image_transfer_labeled.png",
                    dpi=140, bbox_inches="tight")
        plt.close(fig)
        print("Saved labeled test style transfer ->",
              OUTPUTS_DIR / "disvae_test_image_transfer_labeled.png")


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--tune-trials",      type=int, default=10)
    p.add_argument("--tune-epochs",      type=int, default=3)
    p.add_argument("--tune-subset",      type=int, default=2000)
    p.add_argument("--final-epochs",     type=int, default=30)
    p.add_argument("--batch-size",       type=int, default=64)
    p.add_argument("--n-jobs",           type=int, default=1)
    p.add_argument("--n-loader-workers", type=int, default=4)
    p.add_argument("--force-restart",    action="store_true")
    p.add_argument("--test-image", type=str, default=None,
                   help="Path to a test content image for single-image style transfer.")
    p.add_argument("--style-image", type=str, default=None,
                   help="Optional path to style reference image. Defaults to --test-image.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(0)
    device = get_device()
    manifest = json.loads((DATA_PROCESSED / "manifest.json").read_text())
    n_styles = len(manifest["styles"])
    print("device:", device, " styles:", n_styles)

    splits = DATA_PROCESSED / "splits"
    train_ds = WikiArtDataset(splits / "train.csv", root_dir=PROJECT_ROOT,
                              transform=build_transform(128, train=True))
    val_ds   = WikiArtDataset(splits / "val.csv",   root_dir=PROJECT_ROOT,
                              transform=build_transform(128, train=False))

    study = tune(train_ds, val_ds, args, n_styles, device)

    best_cfg = DisVAEConfig(n_styles=n_styles, batch_size=args.batch_size,
                            epochs=args.final_epochs, **study.best_params)
    print("Final cfg:", best_cfg)

    model, opt_main, opt_adv, start_epoch, history, best_val = load_or_init(
        best_cfg, device, args.force_restart)
    print(f"Training epochs {start_epoch}..{best_cfg.epochs}  "
          f"({count_parameters(model)/1e6:.2f}M params)")

    train_loader = make_loader(train_ds, best_cfg.batch_size, True, args.n_loader_workers)
    val_loader   = make_loader(val_ds,   best_cfg.batch_size, False, args.n_loader_workers)
    history = train_final(model, opt_main, opt_adv, train_loader, val_loader,
                          start_epoch, history, best_val, best_cfg, device)
    test_image = (Path(args.test_image).expanduser()
                  if args.test_image else None)
    style_image = (Path(args.style_image).expanduser()
                   if args.style_image else None)
    if test_image is not None and not test_image.is_absolute():
        test_image = PROJECT_ROOT / test_image
    if style_image is not None and not style_image.is_absolute():
        style_image = PROJECT_ROOT / style_image
    plot_and_transfer(history, best_cfg, val_loader, n_styles, device,
                      test_image=test_image, style_image=style_image)


if __name__ == "__main__":
    main()
