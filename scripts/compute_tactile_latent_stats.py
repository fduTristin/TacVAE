#!/usr/bin/env python3
"""Compute deterministic posterior-mean normalization for a trained tactile VAE."""

import argparse
import json
from pathlib import Path

import torch
from diffusers import AutoencoderKL
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF
from tqdm.auto import tqdm


class IndexedImages(Dataset):
    def __init__(self, image_dir, index_file, width, height, max_samples=None):
        self.image_dir = Path(image_dir)
        with Path(index_file).open(encoding="utf-8") as handle:
            self.paths = [Path(line.strip()).name for line in handle if line.strip()]
        if max_samples:
            self.paths = self.paths[:max_samples]
        self.width = width
        self.height = height

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.image_dir / self.paths[index]
        with Image.open(path) as image:
            image = image.convert("RGB")
            if image.size != (self.width, self.height):
                raise ValueError(f"Expected {(self.width, self.height)}, got {image.size}: {path}")
            return TF.to_tensor(image).sub(0.5).div(0.5)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--vae-path", type=Path, required=True)
    parser.add_argument("--image-dir", type=Path, required=True)
    parser.add_argument("--index-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--image-width", type=int, default=256)
    parser.add_argument("--image-height", type=int, default=144)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--max-samples", type=int)
    parser.add_argument("--device", default="cuda")
    return parser.parse_args()


def main():
    args = parse_args()
    dataset = IndexedImages(
        args.image_dir, args.index_file, args.image_width, args.image_height, args.max_samples
    )
    loader = DataLoader(
        dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True
    )
    device = torch.device(args.device)
    vae = AutoencoderKL.from_pretrained(args.vae_path).to(device).eval()

    channel_sum = None
    channel_square_sum = None
    values_per_channel = 0
    latent_shape = None
    with torch.no_grad():
        for images in tqdm(loader):
            mean = vae.encode(images.to(device, non_blocking=True)).latent_dist.mean.double()
            if latent_shape is None:
                latent_shape = list(mean.shape[1:])
            elif list(mean.shape[1:]) != latent_shape:
                raise ValueError(f"Latent shape changed from {latent_shape} to {list(mean.shape[1:])}")
            reduced = mean.sum(dim=(0, 2, 3)).cpu()
            square_reduced = mean.square().sum(dim=(0, 2, 3)).cpu()
            channel_sum = reduced if channel_sum is None else channel_sum + reduced
            channel_square_sum = square_reduced if channel_square_sum is None else channel_square_sum + square_reduced
            values_per_channel += mean.shape[0] * mean.shape[2] * mean.shape[3]

    if channel_sum is None:
        raise ValueError("The image index contains no samples")
    channel_mean = channel_sum / values_per_channel
    channel_var = channel_square_sum / values_per_channel - channel_mean.square()
    channel_std = channel_var.clamp_min(0).sqrt()
    global_mean = channel_sum.sum() / (values_per_channel * len(channel_sum))
    global_var = channel_square_sum.sum() / (values_per_channel * len(channel_sum)) - global_mean.square()
    global_std = global_var.clamp_min(0).sqrt()
    result = {
        "feature": "posterior_mean",
        "samples": len(dataset),
        "latent_shape": latent_shape,
        "mean_per_channel": channel_mean.tolist(),
        "std_per_channel": channel_std.tolist(),
        "global_mean": global_mean.item(),
        "global_std": global_std.item(),
        "recommended_scaling_factor": (1.0 / global_std.clamp_min(1e-12)).item(),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
