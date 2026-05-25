# DDPM from scratch on CIFAR-10

I wanted to actually understand diffusion models — not just import `diffusers`
and call it a day. So I implemented DDPM (Ho et al., 2020) end-to-end on
CIFAR-10: the U-Net, the forward/reverse processes, the training loop, the
ancestral sampler, and FID evaluation. No high-level diffusion libraries doing
the heavy lifting.

## What's in `src/`

| File | What it does |
|---|---|
| `unet.py` | DDPM U-Net — 4 levels, GroupNorm + SiLU, time-conditioned ResBlocks, self-attention at 16×16 and 8×8 |
| `diffusion.py` | Forward `q(x_t \| x_0)`, reverse posterior, ε-prediction loss, ancestral sampler, cosine + linear schedules |
| `data.py` | CIFAR-10 loader normalized to [-1, 1] |
| `train.py` | Training loop with EMA (decay 0.9999), AMP, gradient clipping |
| `sample.py` | Loads a checkpoint, runs the 1000-step sampler, saves a grid PNG |
| `fid.py` | Computes FID against CIFAR-10 train using Inception features |

## How to run

```bash
source ../.venv/bin/activate

# 1. Train
python src/train.py --epochs 200 --batch-size 128

# 2. Generate samples (uses EMA weights by default)
python src/sample.py --ckpt checkpoints/ddpm.pt --n-samples 5000 \
    --save-tensor samples/samples.pt --out samples/grid.png

# 3. Compute FID against 10k real CIFAR-10 images
python src/fid.py --samples samples/samples.pt --n-real 10000
```

## Results

After ~50 epochs (19,400 steps) on an RTX 3050:

| | Value |
|---|---|
| FID (5k samples vs 10k real) | **153.17** |
| Pure-noise FID baseline (reference) | ~445 |

A few things to take from this:

- The pipeline works end-to-end — sampler produces recognizable CIFAR-like
  structure, FID is well below the noise baseline.
- EMA decay (0.9999) really matters — samples from the raw model weights look
  noticeably worse than the EMA copy, which is why the sampler defaults to EMA.
