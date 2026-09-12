"""Policy-side deterministic tactile VAE adapter for the three-module model."""

import json
import math
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from torch import nn


class TactileVAEAdapter(nn.Module):
    """Encode left/right tactile images as H deterministic tactile tokens.

    The spatial posterior-mean maps are normalized and flattened without global
    average pooling. The VAE is frozen by default and exposes only deterministic
    posterior means to the policy.
    """

    def __init__(self, vae, stats_file, expert_width, horizon, freeze_vae=True):
        super().__init__()
        self.vae = vae
        self.freeze_vae = freeze_vae
        stats = json.loads(Path(stats_file).read_text(encoding="utf-8"))
        if stats.get("feature") != "posterior_mean":
            raise ValueError("Latent statistics must be computed from deterministic posterior_mean")
        self.latent_shape = tuple(int(value) for value in stats["latent_shape"])
        mean = torch.tensor(stats["mean_per_channel"], dtype=torch.float32)[None, :, None, None]
        std = torch.tensor(stats["std_per_channel"], dtype=torch.float32)[None, :, None, None]
        if len(mean.flatten()) != self.latent_shape[0] or len(std.flatten()) != self.latent_shape[0]:
            raise ValueError("Latent statistics channels do not match latent_shape")
        self.register_buffer("latent_mean", mean, persistent=True)
        self.register_buffer("latent_std", std.clamp_min(1e-6), persistent=True)
        self.projection = nn.Linear(2 * math.prod(self.latent_shape), expert_width)
        self.horizon_embedding = nn.Parameter(torch.zeros(horizon, expert_width))
        nn.init.normal_(self.horizon_embedding, std=0.02)
        if freeze_vae:
            self.vae.requires_grad_(False)
            self.vae.eval()

    @classmethod
    def from_pretrained(cls, vae_path, stats_file, expert_width, horizon, freeze_vae=True):
        vae = AutoencoderKL.from_pretrained(vae_path)
        return cls(vae, stats_file, expert_width, horizon, freeze_vae=freeze_vae)

    def train(self, mode=True):
        super().train(mode)
        if self.freeze_vae:
            self.vae.eval()
        return self

    def encode_mean(self, images):
        if self.freeze_vae:
            with torch.no_grad():
                mean = self.vae.encode(images).latent_dist.mean
        else:
            mean = self.vae.encode(images).latent_dist.mean
        mean = mean.float()
        if tuple(mean.shape[1:]) != self.latent_shape:
            raise ValueError(
                f"Expected latent shape {self.latent_shape}, got {tuple(mean.shape[1:])}. "
                "Recompute latent statistics for this VAE and input resolution."
            )
        return mean

    def normalize(self, mean):
        return (mean - self.latent_mean) / self.latent_std

    def spatial_feature(self, left_images, right_images):
        left_mean = self.normalize(self.encode_mean(left_images))
        right_mean = self.normalize(self.encode_mean(right_images))
        joined = torch.cat([left_mean.flatten(1), right_mean.flatten(1)], dim=1)
        return self.projection(joined), left_mean, right_mean

    def forward(self, left_images, right_images):
        feature, _, _ = self.spatial_feature(left_images, right_images)
        return feature[:, None, :] + self.horizon_embedding[None, :, :]
