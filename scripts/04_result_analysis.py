"""Step 4 - Result analysis.

Loads checkpoints/{vae,gan,disvae}_best.pt, trains a small ResNet18 style
judge, and compares the three models on the held-out test split:
reconstruction MSE, target-style classification accuracy, and LPIPS to the
content image.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import lpips
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models as tvm
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import (  # noqa: E402
    DisVAEConfig, DisentangledVAE,
    GANConfig, StarGANGenerator,
    VAEConfig, VanillaVAE,
)
from utils import (  # noqa: E402
    CHECKPOINTS_DIR, DATA_PROCESSED, OUTPUTS_DIR,
    WikiArtDataset, build_transform,
    get_device, load_checkpoint, set_seed, show_grid,
)


def load_all(device):
    def must(name):
        ck = load_checkpoint(CHECKPOINTS_DIR / f"{name}_best.pt", map_location=device)
        if ck is None:
            raise FileNotFoundError(
                f"Missing {name}_best.pt - run the corresponding training script first.")
        return ck

    vae_ck, gan_ck, dis_ck = must("vae"), must("gan"), must("disvae")

    vae = VanillaVAE(VAEConfig(**vae_ck["cfg"])).to(device).eval()
    vae.load_state_dict(vae_ck["model_state"])

    gan_cfg = GANConfig(**gan_ck["cfg"])
    G = StarGANGenerator(gan_cfg).to(device).eval()
    G.load_state_dict(gan_ck["model_state"]["G"])

    dis = DisentangledVAE(DisVAEConfig(**dis_ck["cfg"])).to(device).eval()
    dis.load_state_dict(dis_ck["model_state"])

    print("Loaded models:")
    print(f"  VAE    best recon={vae_ck['best_metric']:.4f} (ep {vae_ck['epoch']})")
    print(f"  GAN    best rec  ={gan_ck['best_metric']:.4f} (ep {gan_ck['epoch']})")
    print(f"  DisVAE best recon={dis_ck['best_metric']:.4f} (ep {dis_ck['epoch']})")
    return (vae, vae_ck), (G, gan_cfg, gan_ck), (dis, dis_ck)


def plot_training_curves(vae_ck, gan_ck, dis_ck, n_styles: int) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    vh = pd.DataFrame(vae_ck["history"])
    axes[0].plot(vh["epoch"], vh["tr_recon"], label="train")
    axes[0].plot(vh["epoch"], vh["va_recon"], label="val")
    axes[0].set_title("VAE - recon MSE"); axes[0].legend(); axes[0].set_xlabel("epoch")

    gh = pd.DataFrame(gan_ck["history"])
    axes[1].plot(gh["epoch"], gh["g_rec"], label="cycle L1")
    axes[1].plot(gh["epoch"], gh["g_loss"], label="G total", alpha=0.6)
    axes[1].set_title("GAN - losses"); axes[1].legend(); axes[1].set_xlabel("epoch")

    dh = pd.DataFrame(dis_ck["history"])
    axes[2].plot(dh["epoch"], dh["va_recon"], label="val recon")
    ax2 = axes[2].twinx()
    ax2.plot(dh["epoch"], dh["va_sty_acc"], color="tab:green",
             linestyle="--", label="z_s->style")
    ax2.plot(dh["epoch"], dh["va_adv_acc"], color="tab:red",
             linestyle="--", label="z_c->style")
    ax2.set_ylim(0, 1)
    axes[2].set_title("DisVAE"); axes[2].set_xlabel("epoch")
    axes[2].legend(loc="upper left"); ax2.legend(loc="upper right")
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "compare_training_curves.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)


def train_style_clf(train_ds, val_ds, n_styles: int, epochs: int,
                    batch_size: int, workers: int, device) -> torch.nn.Module:
    ckpt_path = CHECKPOINTS_DIR / "style_clf.pt"
    net = tvm.resnet18(weights=None)
    net.fc = torch.nn.Linear(net.fc.in_features, n_styles)
    net.to(device)
    if ckpt_path.exists():
        net.load_state_dict(torch.load(ckpt_path, map_location=device))
        print("Loaded pre-trained style classifier.")
        return net.eval()

    tl = DataLoader(train_ds, batch_size, shuffle=True,
                    num_workers=workers, drop_last=True)
    vl = DataLoader(val_ds,   batch_size, shuffle=False, num_workers=workers)
    opt = torch.optim.AdamW(net.parameters(), lr=1e-3, weight_decay=1e-4)
    for e in range(epochs):
        net.train()
        for x, y in tqdm(tl, desc=f"clf ep {e+1}", leave=False):
            x, y = x.to(device), y.to(device)
            loss = F.cross_entropy(net(x), y)
            opt.zero_grad(set_to_none=True); loss.backward(); opt.step()
        net.eval(); correct = n = 0
        with torch.no_grad():
            for x, y in vl:
                pred = net(x.to(device)).argmax(1).cpu()
                correct += (pred == y).sum().item(); n += y.numel()
        print(f"  epoch {e+1}: val_acc={correct/n:.3f}")
    torch.save(net.state_dict(), ckpt_path)
    return net.eval()


@torch.no_grad()
def recon_mse(forward_fn, loader, device) -> float:
    s = n = 0
    for x, _ in loader:
        x = x.to(device)
        xr = forward_fn(x)
        s += F.mse_loss(xr, x, reduction="sum").item()
        n += x.numel()
    return s / n


def eval_transfer(test_ds, pair_idx, n_styles: int, G, dis, clf, lpips_fn,
                  batch_size: int, device) -> dict:
    rng = np.random.default_rng(0)
    test_df = pd.read_csv(DATA_PROCESSED / "splits" / "test.csv")
    style_to_idx = {lbl: test_df.index[test_df["label"] == lbl].tolist()
                    for lbl in range(n_styles)}

    @torch.no_grad()
    def run_batch(xs, refs, tgts):
        x = torch.stack(xs).to(device)
        r = torch.stack(refs).to(device)
        t = torch.tensor(tgts, device=device)
        fake_gan = G(x, F.one_hot(t, n_styles).float())
        fake_dis = dis.transfer(x, r)
        gan_pred = clf(fake_gan).argmax(1)
        dis_pred = clf(fake_dis).argmax(1)
        lp_gan = lpips_fn(x, fake_gan).view(-1)
        lp_dis = lpips_fn(x, fake_dis).view(-1)
        return t, gan_pred, dis_pred, lp_gan, lp_dis

    overall = {"gan_acc": 0, "dis_acc": 0, "gan_lp": 0.0, "dis_lp": 0.0, "n": 0}
    per_style: dict[int, dict] = {s: {"gan_acc": [], "dis_acc": [],
                                      "gan_lp": [], "dis_lp": []}
                                  for s in range(n_styles)}
    for i in tqdm(range(0, len(pair_idx), batch_size), desc="transfer"):
        batch = pair_idx[i:i + batch_size]
        xs, refs, tgts = [], [], []
        for j in batch:
            x, y = test_ds[j]
            t = int(rng.choice([s for s in range(n_styles) if s != y]))
            ref_idx = int(rng.choice(style_to_idx[t]))
            r, _ = test_ds[ref_idx]
            xs.append(x); refs.append(r); tgts.append(t)
        t, gp, dp, lg, ld = run_batch(xs, refs, tgts)
        for k in range(len(tgts)):
            overall["gan_acc"] += int(gp[k] == t[k])
            overall["dis_acc"] += int(dp[k] == t[k])
            overall["gan_lp"] += lg[k].item()
            overall["dis_lp"] += ld[k].item()
            overall["n"] += 1
            s = tgts[k]
            per_style[s]["gan_acc"].append(int(gp[k] == t[k]))
            per_style[s]["dis_acc"].append(int(dp[k] == t[k]))
            per_style[s]["gan_lp"].append(lg[k].item())
            per_style[s]["dis_lp"].append(ld[k].item())
    return {"overall": overall, "per_style": per_style}


def per_style_table(res: dict, styles: list[str]) -> pd.DataFrame:
    rows = []
    for lbl, s in enumerate(styles):
        p = res["per_style"][lbl]
        rows.append({
            "style": s,
            "gan_acc":   float(np.mean(p["gan_acc"]))   if p["gan_acc"]   else np.nan,
            "dis_acc":   float(np.mean(p["dis_acc"]))   if p["dis_acc"]   else np.nan,
            "gan_lpips": float(np.mean(p["gan_lp"]))    if p["gan_lp"]    else np.nan,
            "dis_lpips": float(np.mean(p["dis_lp"]))    if p["dis_lp"]    else np.nan,
        })
    df = pd.DataFrame(rows).set_index("style")
    df.to_csv(OUTPUTS_DIR / "per_style_metrics.csv")
    return df


def plot_per_style(df: pd.DataFrame, n_styles: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    df[["gan_acc", "dis_acc"]].plot.bar(ax=axes[0])
    axes[0].set_title("Target-style classifier accuracy (higher = better)")
    axes[0].set_ylim(0, 1)
    axes[0].axhline(1 / n_styles, color="gray", linestyle="--")
    df[["gan_lpips", "dis_lpips"]].plot.bar(
        ax=axes[1], color=["tab:orange", "tab:green"])
    axes[1].set_title("LPIPS to content (lower = more content retained)")
    for ax in axes:
        ax.set_xlabel(""); ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "per_style_metrics.png", dpi=120, bbox_inches="tight")
    plt.close(fig)


def qualitative_grid(test_ds, n_styles: int, styles, vae, G, dis, device) -> None:
    rng = np.random.default_rng(1)
    show_idx = rng.choice(len(test_ds), 6, replace=False).tolist()
    test_df = pd.read_csv(DATA_PROCESSED / "splits" / "test.csv")
    style_to_idx = {lbl: test_df.index[test_df["label"] == lbl].tolist()
                    for lbl in range(n_styles)}

    content_list, refs_list, targets = [], [], []
    for i in show_idx:
        x, y = test_ds[i]
        t = int(rng.choice([s for s in range(n_styles) if s != y]))
        r, _ = test_ds[int(rng.choice(style_to_idx[t]))]
        content_list.append(x); refs_list.append(r); targets.append(t)
    content = torch.stack(content_list).to(device)
    refs    = torch.stack(refs_list).to(device)

    with torch.no_grad():
        gan_out = G(content, F.one_hot(torch.tensor(targets, device=device), n_styles).float())
        dis_out = dis.transfer(content, refs)
        vae_rec = vae(content)["x_hat"]
    rows = torch.cat([content.cpu(), vae_rec.cpu(), refs.cpu(),
                      gan_out.cpu(), dis_out.cpu()], dim=0)
    row_labels = ["content", "VAE recon", "style ref", "GAN transfer", "DisVAE transfer"]
    titles = []
    for r_label in row_labels:
        for t in targets:
            shows_target = ("transfer" in r_label) or (r_label == "style ref")
            titles.append(f"{r_label}\n-> {styles[t]}" if shows_target else r_label)
    fig = show_grid(rows, titles=titles, ncols=len(show_idx),
                    suptitle="Side-by-side qualitative comparison")
    fig.savefig(OUTPUTS_DIR / "qualitative_comparison.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--n-transfer-pairs", type=int, default=200)
    p.add_argument("--clf-epochs",       type=int, default=5)
    p.add_argument("--batch-size",       type=int, default=32)
    p.add_argument("--n-loader-workers", type=int, default=4)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    set_seed(0)
    device = get_device()
    manifest = json.loads((DATA_PROCESSED / "manifest.json").read_text())
    styles = manifest["styles"]
    n_styles = len(styles)
    print("device:", device, " styles:", styles)

    splits = DATA_PROCESSED / "splits"
    train_ds = WikiArtDataset(splits / "train.csv", root_dir=PROJECT_ROOT,
                              transform=build_transform(128, train=True))
    val_ds   = WikiArtDataset(splits / "val.csv",   root_dir=PROJECT_ROOT,
                              transform=build_transform(128, train=False))
    test_ds  = WikiArtDataset(splits / "test.csv",  root_dir=PROJECT_ROOT,
                              transform=build_transform(128, train=False))

    (vae, vae_ck), (G, gan_cfg, gan_ck), (dis, dis_ck) = load_all(device)
    plot_training_curves(vae_ck, gan_ck, dis_ck, n_styles)

    clf = train_style_clf(train_ds, val_ds, n_styles, args.clf_epochs,
                          args.batch_size, args.n_loader_workers, device)

    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False,
                             num_workers=args.n_loader_workers)
    mse_vae = recon_mse(lambda x: vae(x)["x_hat"], test_loader, device)
    mse_dis = recon_mse(lambda x: dis(x)["x_hat"], test_loader, device)
    print(f"Test recon MSE  VAE={mse_vae:.4f}  DisVAE={mse_dis:.4f}")

    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    rng = np.random.default_rng(0)
    pair_idx = rng.choice(len(test_ds),
                          min(args.n_transfer_pairs, len(test_ds)),
                          replace=False).tolist()
    res = eval_transfer(test_ds, pair_idx, n_styles, G, dis, clf, lpips_fn,
                        args.batch_size, device)
    n = res["overall"]["n"]
    print(f"Transfer pairs: {n}")
    print(f"  GAN    style_acc={res['overall']['gan_acc']/n:.3f}   "
          f"LPIPS={res['overall']['gan_lp']/n:.3f}")
    print(f"  DisVAE style_acc={res['overall']['dis_acc']/n:.3f}   "
          f"LPIPS={res['overall']['dis_lp']/n:.3f}")

    ps_df = per_style_table(res, styles)
    plot_per_style(ps_df, n_styles)
    qualitative_grid(test_ds, n_styles, styles, vae, G, dis, device)

    summary = pd.DataFrame({
        "Model": ["Vanilla VAE", "StarGAN", "Disentangled VAE"],
        "Test recon MSE":       [mse_vae,       float("nan"), mse_dis],
        "Style acc (transfer)": [float("nan"),  res["overall"]["gan_acc"]/n,
                                 res["overall"]["dis_acc"]/n],
        "LPIPS (content)":      [float("nan"),  res["overall"]["gan_lp"]/n,
                                 res["overall"]["dis_lp"]/n],
    }).set_index("Model").round(4)
    summary.to_csv(OUTPUTS_DIR / "summary.csv")
    print(summary)


if __name__ == "__main__":
    main()
