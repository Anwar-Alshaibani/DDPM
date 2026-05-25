"""
FID against CIFAR-10 train, using torchmetrics' InceptionV3 (pool3, 2048-dim).
"""
import argparse

import torch
from torchmetrics.image.fid import FrechetInceptionDistance

from project2_ddpm_scratch.data import get_cifar10


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--samples", required=True, help="tensor file from sample.py (N, 3, 32, 32) in [0, 1]")
    p.add_argument("--n-real", type=int, default=10000)
    p.add_argument("--batch-size", type=int, default=64)
    p.add_argument("--feature", type=int, default=2048, choices=[64, 192, 768, 2048])
    p.add_argument("--data-root", default="./data")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")
    fid = FrechetInceptionDistance(feature=args.feature, normalize=True).to(device)

    real_ds = get_cifar10(args.data_root, train=True, augment=False)
    n = 0
    batch = []
    for img, _ in real_ds:
        # get_cifar10 normalizes to [-1, 1]; FID expects [0, 1] with normalize=True
        batch.append((img + 1) / 2)
        if len(batch) == args.batch_size:
            fid.update(torch.stack(batch).to(device), real=True)
            n += len(batch)
            batch = []
            if n >= args.n_real:
                break
    if batch and n < args.n_real:
        fid.update(torch.stack(batch).to(device), real=True)
        n += len(batch)
    print(f"loaded {n} real images")

    fake = torch.load(args.samples)
    print(f"loaded {len(fake)} fake samples")
    for i in range(0, len(fake), args.batch_size):
        chunk = fake[i:i + args.batch_size].to(device)
        fid.update(chunk, real=False)

    score = fid.compute().item()
    print(f"FID(real={n}, fake={len(fake)}, feature={args.feature}) = {score:.3f}")


if __name__ == "__main__":
    main()
