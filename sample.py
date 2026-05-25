"""
Ancestral DDPM sampler. Saves an 8x8 grid PNG and (optionally) raw tensors.
"""
import argparse
from pathlib import Path

import torch
from torchvision.utils import make_grid, save_image

from project2_ddpm_scratch.diffusion import GaussianDiffusion
from unet import UNet


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", required=True)
    p.add_argument("--n-samples", type=int, default=64)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--out", default="./samples/grid.png")
    p.add_argument("--save-tensor", default=None, help="optional .pt file with raw samples in [0, 1]")
    p.add_argument("--no-ema", action="store_true")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt, map_location=device)
    saved_args = ckpt.get("args", {})
    base_ch = saved_args.get("base_ch", 64)
    T = saved_args.get("T", 1000)
    schedule = saved_args.get("schedule", "cosine")

    model = UNet(in_ch=3, base_ch=base_ch).to(device)
    use_ema = (not args.no_ema) and ("ema" in ckpt)
    model.load_state_dict(ckpt["ema" if use_ema else "model"])
    model.eval()
    print(f"loaded {'EMA' if use_ema else 'raw'} weights from {args.ckpt}")

    diffusion = GaussianDiffusion(T=T, schedule=schedule, device=device)

    all_samples = []
    remaining = args.n_samples
    while remaining > 0:
        b = min(args.batch_size, remaining)
        samples = diffusion.sample(model, (b, 3, 32, 32), progress=True)
        samples = (samples.clamp(-1, 1) + 1) / 2
        all_samples.append(samples.cpu())
        remaining -= b
        print(f"  sampled {args.n_samples - remaining}/{args.n_samples}")
    samples = torch.cat(all_samples, dim=0)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    grid = make_grid(samples[:64], nrow=8)
    save_image(grid, args.out)
    print(f"saved grid: {args.out}")

    if args.save_tensor:
        Path(args.save_tensor).parent.mkdir(parents=True, exist_ok=True)
        torch.save(samples, args.save_tensor)
        print(f"saved tensors: {args.save_tensor}")


if __name__ == "__main__":
    main()
