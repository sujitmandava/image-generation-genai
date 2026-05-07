"""Step 4 - Result analysis.

Loads checkpoints/{vae2,disvae}_best.pt, trains a small ResNet18 style judge,
and compares the two models on the held-out test split:

  * Reconstruction MSE (both models).
  * Art-style transfer accuracy + LPIPS-to-content (DisVAE; the only one of
    the two that supports a content/style swap via z_c, z_s).

The legacy vae_best.pt and the StarGAN run are intentionally excluded -
they predate the current model architecture / are out of scope here.
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
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import models as tvm
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from models import (  # noqa: E402
    DisVAEConfig, DisentangledVAE,
    VAEConfig, VanillaVAE,
)
from utils import (  # noqa: E402
    CHECKPOINTS_DIR, DATA_PROCESSED, OUTPUTS_DIR,
    WikiArtDataset, build_transform, denormalize,
    get_device, load_checkpoint, set_seed,
)


# Display names used throughout the figures and CSV summaries.
MODEL_NAMES: dict[str, str] = {
    "vae2":   "VAE-2 (extended)",
    "disvae": "Disentangled VAE",
}
MODEL_COLORS: dict[str, str] = {
    "vae2":   "tab:cyan",
    "disvae": "tab:green",
}


# ---------------------------------------------------------------------------
# Legacy DisVAE compatibility shim
#
# disvae_best.pt was trained against the pre-refactor models.py (no residual
# blocks or self-attention in encoder/decoder). The current DisentangledVAE
# class can't load that state_dict, so we rebuild the older module layout
# here exactly enough to load and call .forward / .transfer.
# ---------------------------------------------------------------------------


def _num_groups(ch: int) -> int:
    for g in (8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


def _legacy_conv(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 4, stride=2, padding=1),
        nn.GroupNorm(_num_groups(c_out), c_out),
        nn.SiLU(inplace=True),
    )


def _legacy_deconv(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(c_in, c_out, 4, stride=2, padding=1),
        nn.GroupNorm(_num_groups(c_out), c_out),
        nn.SiLU(inplace=True),
    )


class _LegacyHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(inplace=True))
        self.fc_mu = nn.Linear(hidden, out_dim)
        self.fc_lv = nn.Linear(hidden, out_dim)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.trunk(h)
        return self.fc_mu(t), self.fc_lv(t)


class _LegacyDisentangledVAE(nn.Module):
    """Pre-refactor DisVAE module layout (5 strided convs + 4 deconvs + final ConvT).

    Public API matches DisentangledVAE: encode / decode / forward / transfer.
    """

    def __init__(self, cfg: DisVAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        c, n_c, n_s = cfg.base_ch, cfg.latent_content, cfg.latent_style
        self.backbone = nn.Sequential(
            _legacy_conv(cfg.in_channels, c),
            _legacy_conv(c, c * 2),
            _legacy_conv(c * 2, c * 4),
            _legacy_conv(c * 4, c * 8),
            _legacy_conv(c * 8, c * 8),
        )
        flat = c * 8 * 4 * 4
        self.content_head = _LegacyHead(flat, n_c)
        self.style_head = _LegacyHead(flat, n_s)
        self.dec_fc = nn.Linear(n_c + n_s, flat)
        self.dec_start = (c * 8, 4, 4)
        self.decoder = nn.Sequential(
            _legacy_deconv(c * 8, c * 8),
            _legacy_deconv(c * 8, c * 4),
            _legacy_deconv(c * 4, c * 2),
            _legacy_deconv(c * 2, c),
            nn.ConvTranspose2d(c, cfg.in_channels, 4, stride=2, padding=1),
            nn.Tanh(),
        )
        self.style_clf = nn.Sequential(
            nn.Linear(n_s, 128), nn.SiLU(inplace=True),
            nn.Linear(128, cfg.n_styles),
        )
        self.adv_clf = nn.Sequential(
            nn.Linear(n_c, 128), nn.SiLU(inplace=True),
            nn.Linear(128, cfg.n_styles),
        )

    @staticmethod
    def reparam(mu: torch.Tensor, lv: torch.Tensor) -> torch.Tensor:
        return mu + (0.5 * lv).exp() * torch.randn_like(mu)

    def encode(self, x: torch.Tensor) -> dict:
        h = self.backbone(x).flatten(1)
        mu_c, lv_c = self.content_head(h)
        mu_s, lv_s = self.style_head(h)
        return {"mu_c": mu_c, "lv_c": lv_c, "mu_s": mu_s, "lv_s": lv_s}

    def decode(self, z_c: torch.Tensor, z_s: torch.Tensor) -> torch.Tensor:
        h = self.dec_fc(torch.cat([z_c, z_s], dim=1)).view(-1, *self.dec_start)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> dict:
        e = self.encode(x)
        z_c = self.reparam(e["mu_c"], e["lv_c"])
        z_s = self.reparam(e["mu_s"], e["lv_s"])
        return {**e, "z_c": z_c, "z_s": z_s,
                "x_hat": self.decode(z_c, z_s)}

    @torch.no_grad()
    def transfer(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        ec, es = self.encode(content), self.encode(style)
        return self.decode(ec["mu_c"], es["mu_s"])


def _build_disvae(ck: dict, device) -> nn.Module:
    """Pick the correct DisVAE arch based on the saved state_dict layout."""
    cfg = DisVAEConfig(**ck["cfg"])
    state = ck["model_state"]
    is_legacy = "backbone.5.body.0.weight" not in state
    Cls = _LegacyDisentangledVAE if is_legacy else DisentangledVAE
    model = Cls(cfg).to(device).eval()
    model.load_state_dict(state)
    if is_legacy:
        print("  (loaded DisVAE via legacy compatibility shim)")
    return model


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------


def load_all(device) -> dict:
    def must(name: str) -> dict:
        ck = load_checkpoint(CHECKPOINTS_DIR / f"{name}_best.pt", map_location=device)
        if ck is None:
            raise FileNotFoundError(
                f"Missing {name}_best.pt - run the corresponding training script first.")
        return ck

    vae2_ck = must("vae2")
    dis_ck  = must("disvae")

    vae2 = VanillaVAE(VAEConfig(**vae2_ck["cfg"])).to(device).eval()
    vae2.load_state_dict(vae2_ck["model_state"])

    dis = _build_disvae(dis_ck, device)

    print("Loaded models:")
    print(f"  {MODEL_NAMES['vae2']:20s} best={vae2_ck['best_metric']:.4f} "
          f"(epoch {vae2_ck['epoch']})")
    print(f"  {MODEL_NAMES['disvae']:20s} best={dis_ck['best_metric']:.4f} "
          f"(epoch {dis_ck['epoch']})")
    return {
        "vae2":   {"model": vae2, "ckpt": vae2_ck},
        "disvae": {"model": dis,  "ckpt": dis_ck},
    }


# ---------------------------------------------------------------------------
# Training-curve panel
# ---------------------------------------------------------------------------


def plot_training_curves(M: dict) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.0))

    v2h = pd.DataFrame(M["vae2"]["ckpt"]["history"])
    axes[0].plot(v2h["epoch"], v2h["tr_recon"], label="train")
    axes[0].plot(v2h["epoch"], v2h["va_recon"], label="val")
    axes[0].set_title(f"{MODEL_NAMES['vae2']} - reconstruction MSE\n(lower = better)")
    axes[0].set_xlabel("epoch"); axes[0].set_ylabel("MSE")
    axes[0].legend()

    dh = pd.DataFrame(M["disvae"]["ckpt"]["history"])
    axes[1].plot(dh["epoch"], dh["va_recon"], label="val recon MSE", color="tab:blue")
    ax2 = axes[1].twinx()
    ax2.plot(dh["epoch"], dh["va_sty_acc"], color="tab:green",
             linestyle="--", label="z_style -> style acc")
    ax2.plot(dh["epoch"], dh["va_adv_acc"], color="tab:red",
             linestyle="--", label="z_content -> style acc (adv)")
    ax2.set_ylim(0, 1); ax2.set_ylabel("classifier accuracy")
    axes[1].set_title(f"{MODEL_NAMES['disvae']} - recon + disentanglement")
    axes[1].set_xlabel("epoch"); axes[1].set_ylabel("MSE")
    axes[1].legend(loc="upper left"); ax2.legend(loc="upper right")

    fig.suptitle("Training curves per model (lower MSE = better reconstruction)",
                 fontsize=12)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "compare_training_curves.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Style judge (ResNet18) - reused across models
# ---------------------------------------------------------------------------


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


# ---------------------------------------------------------------------------
# Reconstruction evaluation (both models)
# ---------------------------------------------------------------------------


@torch.no_grad()
def recon_mse(forward_fn, loader, device) -> float:
    """Mean MSE between reconstruction and input over the given loader."""
    s = n = 0
    for x, _ in loader:
        x = x.to(device)
        xr = forward_fn(x)
        s += F.mse_loss(xr, x, reduction="sum").item()
        n += x.numel()
    return s / n


def plot_recon_bar(mses: dict[str, float]) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 3.8))
    keys  = list(mses.keys())
    names = [MODEL_NAMES[k] for k in keys]
    vals  = [mses[k] for k in keys]
    colors = [MODEL_COLORS[k] for k in keys]
    bars = ax.bar(names, vals, color=colors)
    ax.set_ylabel("Test reconstruction MSE")
    ax.set_title("Reconstruction error on the held-out test split "
                 "(lower = better)")
    ymax = max(vals) * 1.15 if max(vals) > 0 else 1.0
    ax.set_ylim(0, ymax)
    for b, v in zip(bars, vals):
        ax.text(b.get_x() + b.get_width() / 2, v + ymax * 0.01,
                f"{v:.4f}", ha="center", va="bottom", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "recon_mse_compare.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Style-transfer evaluation (DisVAE only)
# ---------------------------------------------------------------------------


def eval_transfer(test_ds, pair_idx, n_styles: int, dis, clf, lpips_fn,
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
        fake_dis = dis.transfer(x, r)
        dis_pred = clf(fake_dis).argmax(1)
        lp_dis = lpips_fn(x, fake_dis).view(-1)
        return t, dis_pred, lp_dis

    overall = {"dis_acc": 0, "dis_lp": 0.0, "n": 0}
    per_style: dict[int, dict] = {s: {"dis_acc": [], "dis_lp": []}
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
        t, dp, ld = run_batch(xs, refs, tgts)
        for k in range(len(tgts)):
            overall["dis_acc"] += int(dp[k] == t[k])
            overall["dis_lp"] += ld[k].item()
            overall["n"] += 1
            s = tgts[k]
            per_style[s]["dis_acc"].append(int(dp[k] == t[k]))
            per_style[s]["dis_lp"].append(ld[k].item())
    return {"overall": overall, "per_style": per_style}


def per_style_table(res: dict, styles: list[str]) -> pd.DataFrame:
    rows = []
    for lbl, s in enumerate(styles):
        p = res["per_style"][lbl]
        rows.append({
            "style": s,
            "DisVAE_acc":   float(np.mean(p["dis_acc"])) if p["dis_acc"] else np.nan,
            "DisVAE_lpips": float(np.mean(p["dis_lp"]))  if p["dis_lp"]  else np.nan,
        })
    df = pd.DataFrame(rows).set_index("style")
    df.to_csv(OUTPUTS_DIR / "per_style_metrics.csv")
    return df


def plot_per_style(df: pd.DataFrame, n_styles: int) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.4))

    df[["DisVAE_acc"]].plot.bar(
        ax=axes[0], color=[MODEL_COLORS["disvae"]], legend=False)
    axes[0].set_title(f"{MODEL_NAMES['disvae']} - "
                      "style-transfer success per target style\n"
                      "(higher = transferred image is classified as the target style)")
    axes[0].set_ylabel("Target-style classifier accuracy")
    axes[0].set_ylim(0, 1)
    axes[0].axhline(1 / n_styles, color="gray", linestyle="--",
                    label=f"random chance (1/{n_styles}={1/n_styles:.2f})")
    axes[0].legend()

    df[["DisVAE_lpips"]].plot.bar(
        ax=axes[1], color=[MODEL_COLORS["disvae"]], legend=False)
    axes[1].set_title(f"{MODEL_NAMES['disvae']} - "
                      "content preservation (LPIPS to source)\n"
                      "(lower = more original content retained)")
    axes[1].set_ylabel("LPIPS distance to content image")

    for ax in axes:
        ax.set_xlabel("Target style")
        ax.tick_params(axis="x", rotation=30)
    fig.tight_layout()
    fig.savefig(OUTPUTS_DIR / "style_transfer_per_style.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Qualitative grids (reconstruction & style transfer) with row/column labels
# ---------------------------------------------------------------------------


def _row_labelled_grid(grid: torch.Tensor, row_labels: list[str],
                       col_titles: list[str], suptitle: str | None = None,
                       cell: float = 1.7):
    """Plot an NxM image grid with row labels on the left and col titles on top."""
    nrows, ncols = len(row_labels), len(col_titles)
    assert grid.shape[0] == nrows * ncols, (
        f"grid has {grid.shape[0]} images but expected {nrows*ncols}")

    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(ncols * cell + 1.4, nrows * cell + 0.8))
    if nrows == 1: axes = np.expand_dims(axes, 0)
    if ncols == 1: axes = np.expand_dims(axes, 1)

    imgs = grid.detach().cpu()
    if imgs.min() < -0.01:
        imgs = denormalize(imgs)

    for r in range(nrows):
        for c in range(ncols):
            ax = axes[r, c]
            ax.imshow(imgs[r * ncols + c].permute(1, 2, 0).numpy())
            ax.set_xticks([]); ax.set_yticks([])
            for spine in ax.spines.values():
                spine.set_visible(False)
            if r == 0 and col_titles[c]:
                ax.set_title(col_titles[c], fontsize=8)
        axes[r, 0].set_ylabel(row_labels[r], fontsize=10, rotation=0,
                              ha="right", va="center", labelpad=12)

    if suptitle:
        fig.suptitle(suptitle, fontsize=12)
    fig.tight_layout()
    return fig


def qualitative_reconstruction(test_ds, M: dict, styles: list[str],
                               device, n: int = 6) -> None:
    """Per-example side-by-side: input vs each model's reconstruction."""
    rng = np.random.default_rng(2)
    idx = rng.choice(len(test_ds), n, replace=False).tolist()
    xs, ys = [], []
    for i in idx:
        x, y = test_ds[i]; xs.append(x); ys.append(y)
    x = torch.stack(xs).to(device)

    with torch.no_grad():
        xr_vae2 = M["vae2"]["model"](x)["x_hat"]
        xr_dis  = M["disvae"]["model"](x)["x_hat"]

    rows = torch.cat([x.cpu(), xr_vae2.cpu(), xr_dis.cpu()], dim=0)
    col_titles = [f"#{i+1}\n({styles[ys[i]]})" for i in range(n)]
    row_labels = [
        "Input",
        MODEL_NAMES["vae2"],
        MODEL_NAMES["disvae"],
    ]
    fig = _row_labelled_grid(
        rows, row_labels, col_titles,
        suptitle="Reconstruction comparison - top row is the original input; "
                 "each row below is that model's reconstruction")
    fig.savefig(OUTPUTS_DIR / "qualitative_reconstruction.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)


def qualitative_style_transfer(test_ds, M: dict, n_styles: int,
                               styles: list[str], device, n: int = 6) -> None:
    """Per-example: content + style ref + DisVAE transfer output."""
    rng = np.random.default_rng(3)
    test_df = pd.read_csv(DATA_PROCESSED / "splits" / "test.csv")
    style_to_idx = {lbl: test_df.index[test_df["label"] == lbl].tolist()
                    for lbl in range(n_styles)}

    show_idx = rng.choice(len(test_ds), n, replace=False).tolist()
    xs, refs, ys, tgts = [], [], [], []
    for i in show_idx:
        x, y = test_ds[i]
        t = int(rng.choice([s for s in range(n_styles) if s != y]))
        r, _ = test_ds[int(rng.choice(style_to_idx[t]))]
        xs.append(x); refs.append(r); ys.append(y); tgts.append(t)
    x = torch.stack(xs).to(device)
    r = torch.stack(refs).to(device)

    with torch.no_grad():
        dis_out = M["disvae"]["model"].transfer(x, r)

    rows = torch.cat([x.cpu(), r.cpu(), dis_out.cpu()], dim=0)
    col_titles = [f"#{i+1}\n{styles[ys[i]]} -> {styles[tgts[i]]}"
                  for i in range(n)]
    row_labels = [
        "Content",
        "Style reference",
        f"{MODEL_NAMES['disvae']}\ntransfer",
    ]
    fig = _row_labelled_grid(
        rows, row_labels, col_titles,
        suptitle="Art-style transfer - column header is "
                 "'source style -> target style'")
    fig.savefig(OUTPUTS_DIR / "qualitative_style_transfer.png",
                dpi=120, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


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

    M = load_all(device)
    plot_training_curves(M)

    clf = train_style_clf(train_ds, val_ds, n_styles, args.clf_epochs,
                          args.batch_size, args.n_loader_workers, device)

    test_loader = DataLoader(test_ds, args.batch_size, shuffle=False,
                             num_workers=args.n_loader_workers)
    mse: dict[str, float] = {
        "vae2":   recon_mse(lambda x: M["vae2"]["model"](x)["x_hat"],
                            test_loader, device),
        "disvae": recon_mse(lambda x: M["disvae"]["model"](x)["x_hat"],
                            test_loader, device),
    }
    print("\nTest reconstruction MSE (lower = better):")
    for k, v in mse.items():
        print(f"  {MODEL_NAMES[k]:20s} {v:.4f}")
    plot_recon_bar(mse)

    lpips_fn = lpips.LPIPS(net="vgg").to(device).eval()
    for p in lpips_fn.parameters():
        p.requires_grad_(False)

    rng = np.random.default_rng(0)
    pair_idx = rng.choice(len(test_ds),
                          min(args.n_transfer_pairs, len(test_ds)),
                          replace=False).tolist()
    res = eval_transfer(test_ds, pair_idx, n_styles,
                        M["disvae"]["model"],
                        clf, lpips_fn, args.batch_size, device)
    n = res["overall"]["n"]
    print(f"\nStyle-transfer evaluation ({n} pairs):")
    print(f"  {MODEL_NAMES['disvae']:20s} target-style acc={res['overall']['dis_acc']/n:.3f}"
          f"   LPIPS={res['overall']['dis_lp']/n:.3f}")

    ps_df = per_style_table(res, styles)
    plot_per_style(ps_df, n_styles)
    qualitative_reconstruction(test_ds, M, styles, device)
    qualitative_style_transfer(test_ds, M, n_styles, styles, device)

    summary = pd.DataFrame({
        "Model": [MODEL_NAMES["vae2"], MODEL_NAMES["disvae"]],
        "Test recon MSE (lower=better)": [
            mse["vae2"], mse["disvae"]],
        "Style-transfer acc (higher=better)": [
            float("nan"),
            res["overall"]["dis_acc"] / n,
        ],
        "Content LPIPS (lower=better)": [
            float("nan"),
            res["overall"]["dis_lp"] / n,
        ],
    }).set_index("Model").round(4)
    summary.to_csv(OUTPUTS_DIR / "summary.csv")
    print("\nSummary:")
    print(summary)


if __name__ == "__main__":
    main()
