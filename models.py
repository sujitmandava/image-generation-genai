"""Model definitions: vanilla VAE, disentangled (content + style) VAE, and a
compact StarGAN-style multi-domain GAN.

Shared building blocks keep the three architectures aligned at 128 x 128.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# Shared conv blocks (128 -> 4 in 5 strides)
# ---------------------------------------------------------------------------


def _num_groups(ch: int) -> int:
    for g in (8, 4, 2, 1):
        if ch % g == 0:
            return g
    return 1


def _conv(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(c_in, c_out, 4, stride=2, padding=1),
        nn.GroupNorm(_num_groups(c_out), c_out),
        nn.SiLU(inplace=True),
    )


def _deconv(c_in: int, c_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.ConvTranspose2d(c_in, c_out, 4, stride=2, padding=1),
        nn.GroupNorm(_num_groups(c_out), c_out),
        nn.SiLU(inplace=True),
    )


class _ResBlockGN(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.GroupNorm(_num_groups(ch), ch),
            nn.SiLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.GroupNorm(_num_groups(ch), ch),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.body(x)


class _SelfAttention2d(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.norm = nn.GroupNorm(_num_groups(ch), ch)
        self.q = nn.Conv2d(ch, ch, 1, bias=False)
        self.k = nn.Conv2d(ch, ch, 1, bias=False)
        self.v = nn.Conv2d(ch, ch, 1, bias=False)
        self.proj = nn.Conv2d(ch, ch, 1)
        self.scale = ch ** -0.5

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, c, h, w = x.shape
        y = self.norm(x)
        q = self.q(y).reshape(b, c, h * w).transpose(1, 2)
        k = self.k(y).reshape(b, c, h * w)
        v = self.v(y).reshape(b, c, h * w).transpose(1, 2)
        attn = torch.softmax((q @ k) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(b, c, h, w)
        return x + self.proj(out)


def _build_encoder(in_ch: int, base: int) -> tuple[nn.Sequential, int]:
    net = nn.Sequential(
        _conv(in_ch, base),       # 64
        _conv(base, base * 2),    # 32
        _conv(base * 2, base * 4),# 16
        _conv(base * 4, base * 8),# 8
        _conv(base * 8, base * 8),# 4
        _ResBlockGN(base * 8),
        _SelfAttention2d(base * 8),
        _ResBlockGN(base * 8),
    )
    return net, base * 8 * 4 * 4


def _build_decoder(out_ch: int, base: int) -> nn.Sequential:
    return nn.Sequential(
        _ResBlockGN(base * 8),
        _SelfAttention2d(base * 8),
        _ResBlockGN(base * 8),
        _deconv(base * 8, base * 8),  # 8
        _deconv(base * 8, base * 4),  # 16
        _deconv(base * 4, base * 2),  # 32
        _deconv(base * 2, base),      # 64
        nn.ConvTranspose2d(base, out_ch, 4, stride=2, padding=1),
        nn.Tanh(),
    )


# ---------------------------------------------------------------------------
# Vanilla beta-VAE
# ---------------------------------------------------------------------------


@dataclass
class VAEConfig:
    image_size: int = 128
    in_channels: int = 3
    base_ch: int = 96
    latent_dim: int = 384
    beta: float = 1.0
    lr: float = 2e-4
    batch_size: int = 64
    epochs: int = 25

    def to_dict(self) -> dict: return self.__dict__.copy()


class VanillaVAE(nn.Module):
    def __init__(self, cfg: VAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.encoder, flat_dim = _build_encoder(cfg.in_channels, cfg.base_ch)
        self.fc_mu = nn.Linear(flat_dim, cfg.latent_dim)
        self.fc_lv = nn.Linear(flat_dim, cfg.latent_dim)
        self.dec_fc = nn.Linear(cfg.latent_dim, flat_dim)
        self.dec_start = (cfg.base_ch * 8, 4, 4)
        self.decoder = _build_decoder(cfg.in_channels, cfg.base_ch)

    @staticmethod
    def reparam(mu: torch.Tensor, lv: torch.Tensor) -> torch.Tensor:
        return mu + (0.5 * lv).exp() * torch.randn_like(mu)

    def encode(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.encoder(x).flatten(1)
        return self.fc_mu(h), self.fc_lv(h)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        h = self.dec_fc(z).view(-1, *self.dec_start)
        return self.decoder(h)

    def forward(self, x: torch.Tensor) -> dict:
        mu, lv = self.encode(x)
        z = self.reparam(mu, lv)
        return {"x_hat": self.decode(z), "mu": mu, "lv": lv, "z": z}


def vae_loss(out: dict, x: torch.Tensor, beta: float) -> dict:
    recon = F.mse_loss(out["x_hat"], x, reduction="mean")
    kl_per_sample = -0.5 * (1 + out["lv"] - out["mu"].pow(2) - out["lv"].exp()).sum(1)
    kl = kl_per_sample.mean() / (x.shape[1] * x.shape[2] * x.shape[3])
    return {"loss": recon + beta * kl, "recon": recon, "kl": kl}


# ---------------------------------------------------------------------------
# Disentangled (content + style) VAE
# ---------------------------------------------------------------------------


@dataclass
class DisVAEConfig:
    image_size: int = 128
    in_channels: int = 3
    base_ch: int = 64
    latent_content: int = 128
    latent_style: int = 32
    n_styles: int = 8
    beta_content: float = 1.0
    beta_style: float = 0.5
    style_clf_w: float = 1.0
    adv_w: float = 0.1
    lr: float = 2e-4
    batch_size: int = 64
    epochs: int = 30

    def to_dict(self) -> dict: return self.__dict__.copy()


class _Head(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.trunk = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(inplace=True))
        self.fc_mu = nn.Linear(hidden, out_dim)
        self.fc_lv = nn.Linear(hidden, out_dim)

    def forward(self, h: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        t = self.trunk(h)
        return self.fc_mu(t), self.fc_lv(t)


class DisentangledVAE(nn.Module):
    def __init__(self, cfg: DisVAEConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.backbone, flat = _build_encoder(cfg.in_channels, cfg.base_ch)
        self.content_head = _Head(flat, cfg.latent_content)
        self.style_head = _Head(flat, cfg.latent_style)
        self.dec_fc = nn.Linear(cfg.latent_content + cfg.latent_style,
                                cfg.base_ch * 8 * 4 * 4)
        self.dec_start = (cfg.base_ch * 8, 4, 4)
        self.decoder = _build_decoder(cfg.in_channels, cfg.base_ch)
        self.style_clf = nn.Sequential(
            nn.Linear(cfg.latent_style, 128), nn.SiLU(inplace=True),
            nn.Linear(128, cfg.n_styles),
        )
        self.adv_clf = nn.Sequential(
            nn.Linear(cfg.latent_content, 128), nn.SiLU(inplace=True),
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
                "x_hat": self.decode(z_c, z_s),
                "style_logits": self.style_clf(z_s),
                "adv_logits": self.adv_clf(z_c)}

    @torch.no_grad()
    def transfer(self, content: torch.Tensor, style: torch.Tensor) -> torch.Tensor:
        ec, es = self.encode(content), self.encode(style)
        return self.decode(ec["mu_c"], es["mu_s"])


def disvae_loss(out: dict, x: torch.Tensor, y: torch.Tensor,
                cfg: DisVAEConfig) -> dict:
    recon = F.mse_loss(out["x_hat"], x, reduction="mean")
    n_pix = x.shape[1] * x.shape[2] * x.shape[3]
    kl_c = (-0.5 * (1 + out["lv_c"] - out["mu_c"].pow(2) - out["lv_c"].exp())).sum(1).mean() / n_pix
    kl_s = (-0.5 * (1 + out["lv_s"] - out["mu_s"].pow(2) - out["lv_s"].exp())).sum(1).mean() / n_pix
    style_ce = F.cross_entropy(out["style_logits"], y)
    adv_ce = F.cross_entropy(out["adv_logits"], y)
    total = (recon
             + cfg.beta_content * kl_c + cfg.beta_style * kl_s
             + cfg.style_clf_w * style_ce
             - cfg.adv_w * adv_ce)
    return {"loss": total, "recon": recon, "kl_c": kl_c, "kl_s": kl_s,
            "style_ce": style_ce, "adv_ce": adv_ce}


# ---------------------------------------------------------------------------
# Compact StarGAN-style multi-domain GAN
# ---------------------------------------------------------------------------


@dataclass
class GANConfig:
    image_size: int = 128
    in_channels: int = 3
    base_ch: int = 64
    n_styles: int = 8
    n_res_blocks: int = 4
    lr_g: float = 1e-4
    lr_d: float = 1e-4
    beta1: float = 0.5
    beta2: float = 0.999
    lambda_cls: float = 1.0
    lambda_rec: float = 10.0
    lambda_gp: float = 10.0
    n_critic: int = 5
    batch_size: int = 32
    epochs: int = 20

    def to_dict(self) -> dict: return self.__dict__.copy()


class _ResBlock(nn.Module):
    def __init__(self, ch: int) -> None:
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(ch, affine=True), nn.ReLU(inplace=True),
            nn.Conv2d(ch, ch, 3, 1, 1, bias=False),
            nn.InstanceNorm2d(ch, affine=True),
        )
    def forward(self, x): return x + self.body(x)


class StarGANGenerator(nn.Module):
    def __init__(self, cfg: GANConfig) -> None:
        super().__init__()
        c = cfg.base_ch
        in_ch = cfg.in_channels + cfg.n_styles
        layers: list[nn.Module] = [
            nn.Conv2d(in_ch, c, 7, 1, 3, bias=False),
            nn.InstanceNorm2d(c, affine=True), nn.ReLU(inplace=True),
            nn.Conv2d(c, c * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(c * 2, affine=True), nn.ReLU(inplace=True),
            nn.Conv2d(c * 2, c * 4, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(c * 4, affine=True), nn.ReLU(inplace=True),
        ]
        layers += [_ResBlock(c * 4) for _ in range(cfg.n_res_blocks)]
        layers += [
            nn.ConvTranspose2d(c * 4, c * 2, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(c * 2, affine=True), nn.ReLU(inplace=True),
            nn.ConvTranspose2d(c * 2, c, 4, 2, 1, bias=False),
            nn.InstanceNorm2d(c, affine=True), nn.ReLU(inplace=True),
            nn.Conv2d(c, cfg.in_channels, 7, 1, 3), nn.Tanh(),
        ]
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        c = c.view(c.size(0), c.size(1), 1, 1).expand(-1, -1, x.size(2), x.size(3))
        return self.net(torch.cat([x, c], dim=1))


class StarGANDiscriminator(nn.Module):
    def __init__(self, cfg: GANConfig) -> None:
        super().__init__()
        c = cfg.base_ch
        layers: list[nn.Module] = [
            nn.Conv2d(cfg.in_channels, c, 4, 2, 1),
            nn.LeakyReLU(0.01, inplace=True),
        ]
        cur = c
        for _ in range(4):
            nxt = min(cur * 2, c * 16)
            layers += [nn.Conv2d(cur, nxt, 4, 2, 1),
                       nn.LeakyReLU(0.01, inplace=True)]
            cur = nxt
        self.body = nn.Sequential(*layers)
        ks = cfg.image_size // (2 ** 5)
        self.src = nn.Conv2d(cur, 1, 3, 1, 1)
        self.cls = nn.Conv2d(cur, cfg.n_styles, ks, 1, 0)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.body(x)
        return self.src(h), self.cls(h).view(x.size(0), -1)
