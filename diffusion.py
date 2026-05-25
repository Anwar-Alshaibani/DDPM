"""
DDPM forward / reverse processes.
"""
import math

import torch
import torch.nn.functional as F


def linear_beta_schedule(T, beta_start=1e-4, beta_end=0.02):
    return torch.linspace(beta_start, beta_end, T)


def cosine_beta_schedule(T, s=0.008):
    steps = T + 1
    t = torch.linspace(0, T, steps) / T
    alphas_cumprod = torch.cos((t + s) / (1 + s) * math.pi * 0.5) ** 2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clamp(betas, 1e-4, 0.999)


class GaussianDiffusion:
    """Discrete-time Gaussian diffusion. Predicts ε"""

    def __init__(self, T=1000, schedule="cosine", device="cuda"):
        self.T = T
        self.device = device
        betas = cosine_beta_schedule(T) if schedule == "cosine" else linear_beta_schedule(T)
        self.betas = betas.to(device)
        alphas = 1.0 - self.betas
        self.alphas_cumprod = torch.cumprod(alphas, dim=0)
        self.sqrt_alphas_cumprod = torch.sqrt(self.alphas_cumprod)
        self.sqrt_one_minus_alphas_cumprod = torch.sqrt(1 - self.alphas_cumprod)
        self.sqrt_recip_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod)
        self.sqrt_recipm1_alphas_cumprod = torch.sqrt(1.0 / self.alphas_cumprod - 1)

        alphas_cumprod_prev = F.pad(self.alphas_cumprod[:-1], (1, 0), value=1.0)
        self.posterior_variance = self.betas * (1 - alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.posterior_mean_coef1 = self.betas * torch.sqrt(alphas_cumprod_prev) / (1 - self.alphas_cumprod)
        self.posterior_mean_coef2 = (1 - alphas_cumprod_prev) * torch.sqrt(alphas) / (1 - self.alphas_cumprod)

    @staticmethod
    def _gather(coef, t, x_shape):
        out = coef.gather(0, t)
        while out.dim() < len(x_shape):
            out = out.unsqueeze(-1)
        return out

    def q_sample(self, x0, t, noise=None):
        if noise is None:
            noise = torch.randn_like(x0)
        sqrt_a = self._gather(self.sqrt_alphas_cumprod, t, x0.shape)
        sqrt_1ma = self._gather(self.sqrt_one_minus_alphas_cumprod, t, x0.shape)
        return sqrt_a * x0 + sqrt_1ma * noise, noise

    def predict_x0_from_eps(self, x_t, t, eps):
        return (self._gather(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t
                - self._gather(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * eps)

    def q_posterior(self, x0, x_t, t):
        mean = (self._gather(self.posterior_mean_coef1, t, x_t.shape) * x0
                + self._gather(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        var = self._gather(self.posterior_variance, t, x_t.shape)
        return mean, var

    @torch.no_grad()
    def p_sample(self, model, x_t, t):
        eps = model(x_t, t)
        x0 = self.predict_x0_from_eps(x_t, t, eps).clamp(-1, 1)
        mean, var = self.q_posterior(x0, x_t, t)
        noise = torch.randn_like(x_t) if t[0].item() > 0 else torch.zeros_like(x_t)
        return mean + var.sqrt() * noise

    @torch.no_grad()
    def sample(self, model, shape, progress=False):
        x = torch.randn(shape, device=self.device)
        rng = reversed(range(self.T))
        if progress:
            from tqdm import tqdm
            rng = tqdm(list(rng), desc="sampling")
        for i in rng:
            t = torch.full((shape[0],), i, device=self.device, dtype=torch.long)
            x = self.p_sample(model, x, t)
        return x

    def training_loss(self, model, x0):
        t = torch.randint(0, self.T, (x0.shape[0],), device=x0.device)
        x_t, noise = self.q_sample(x0, t)
        pred = model(x_t, t)
        return F.mse_loss(pred, noise)
