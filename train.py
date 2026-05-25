"""
Train DDPM on CIFAR-10. Tracks EMA weights separately (decay=0.9999) — these
produce noticeably better samples than the raw model weights.
"""
import argparse
import copy
import time
from pathlib import Path

import torch
from torch.optim import AdamW

from project2_ddpm_scratch.data import get_cifar10, make_loader
from project2_ddpm_scratch.diffusion import GaussianDiffusion
from unet import UNet


class EMA:
    def __init__(self, model, decay=0.9999):
        self.decay = decay
        self.model = copy.deepcopy(model).eval()
        for p in self.model.parameters():
            p.requires_grad = False

    @torch.no_grad()
    def update(self, model):
        for p_ema, p in zip(self.model.parameters(), model.parameters()):
            p_ema.data.mul_(self.decay).add_(p.data, alpha=1 - self.decay)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--epochs", type=int, default=200)
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--lr", type=float, default=2e-4)
    p.add_argument("--schedule", default="cosine", choices=["linear", "cosine"])
    p.add_argument("--T", type=int, default=1000)
    p.add_argument("--base-ch", type=int, default=64)
    p.add_argument("--data-root", default="./data")
    p.add_argument("--save", default="./checkpoints/ddpm.pt")
    p.add_argument("--ema-decay", type=float, default=0.9999)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--amp", action="store_true", default=True)
    p.add_argument("--log-every", type=int, default=200)
    p.add_argument("--max-steps", type=int, default=-1, help="cap total steps (for smoke tests)")
    args = p.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    ds = get_cifar10(args.data_root, train=True, augment=True)
    loader = make_loader(ds, args.batch_size, shuffle=True, num_workers=args.num_workers)

    model = UNet(in_ch=3, base_ch=args.base_ch).to(device)
    n_params = sum(prm.numel() for prm in model.parameters()) / 1e6
    print(f"U-Net params: {n_params:.2f}M")

    diffusion = GaussianDiffusion(T=args.T, schedule=args.schedule, device=device)
    opt = AdamW(model.parameters(), lr=args.lr)
    scaler = torch.amp.GradScaler("cuda", enabled=args.amp and device.type == "cuda")
    ema = EMA(model, decay=args.ema_decay)

    Path(args.save).parent.mkdir(parents=True, exist_ok=True)
    step = 0
    stop = False
    for epoch in range(args.epochs):
        if stop:
            break
        model.train()
        t0 = time.time()
        running = 0.0
        for imgs, _ in loader:
            imgs = imgs.to(device, non_blocking=True)
            opt.zero_grad(set_to_none=True)
            with torch.amp.autocast("cuda", enabled=args.amp and device.type == "cuda"):
                loss = diffusion.training_loss(model, imgs)
            scaler.scale(loss).backward()
            scaler.unscale_(opt)
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            scaler.step(opt)
            scaler.update()
            ema.update(model)
            running += loss.item()
            step += 1
            if step % args.log_every == 0:
                print(f"epoch {epoch+1} step {step} loss={loss.item():.4f}")
            if args.max_steps > 0 and step >= args.max_steps:
                stop = True
                break
        dt = time.time() - t0
        avg = running / max(1, step % max(1, len(loader)) or len(loader))
        print(f"== epoch {epoch+1}/{args.epochs}  avg_loss={avg:.4f}  ({dt:.1f}s)")
        torch.save({
            "model": model.state_dict(),
            "ema": ema.model.state_dict(),
            "epoch": epoch,
            "step": step,
            "args": vars(args),
        }, args.save)
    print(f"saved: {args.save}  steps: {step}")


if __name__ == "__main__":
    main()
