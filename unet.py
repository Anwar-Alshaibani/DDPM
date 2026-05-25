"""
Small DDPM U-Net for 32x32 CIFAR-10.
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t, dim, max_period=10000):
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, device=t.device).float() / half)
    args = t[:, None].float() * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


class ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, t_emb_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.GroupNorm(min(32, in_ch), in_ch)
        self.conv1 = nn.Conv2d(in_ch, out_ch, 3, padding=1)
        self.t_proj = nn.Linear(t_emb_dim, out_ch)
        self.norm2 = nn.GroupNorm(min(32, out_ch), out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = nn.Conv2d(out_ch, out_ch, 3, padding=1)
        self.skip = nn.Conv2d(in_ch, out_ch, 1) if in_ch != out_ch else nn.Identity()

    def forward(self, x, t_emb):
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.t_proj(F.silu(t_emb))[:, :, None, None]
        h = self.conv2(self.dropout(F.silu(self.norm2(h))))
        return h + self.skip(x)


class AttnBlock(nn.Module):
    def __init__(self, ch, num_heads=4):
        super().__init__()
        # Ensure heads divide channels; fall back to 1 head if not.
        if ch % num_heads != 0:
            num_heads = 1
        self.num_heads = num_heads
        self.head_dim = ch // num_heads
        self.norm = nn.GroupNorm(min(32, ch), ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = nn.Conv2d(ch, ch, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        h = self.norm(x)
        qkv = self.qkv(h).reshape(B, 3, self.num_heads, self.head_dim, H * W)
        qkv = qkv.permute(1, 0, 2, 4, 3)  # 3, B, heads, HW, head_dim
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)
        out = out.permute(0, 1, 3, 2).reshape(B, C, H, W)
        return x + self.proj(out)


class Downsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x):
        return self.op(x)


class Upsample(nn.Module):
    def __init__(self, ch):
        super().__init__()
        self.op = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        return self.op(x)


class UNet(nn.Module):

    def __init__(self, in_ch=3, base_ch=64, ch_mult=(1, 2, 2, 2),
                 num_res_blocks=2, attn_resolutions=(16, 8),
                 dropout=0.1, img_size=32):
        super().__init__()
        self.in_ch = in_ch
        self.base_ch = base_ch
        self.num_resolutions = len(ch_mult)
        self.num_res_blocks = num_res_blocks

        t_emb_dim = base_ch * 4
        self.t_mlp = nn.Sequential(
            nn.Linear(base_ch, t_emb_dim),
            nn.SiLU(),
            nn.Linear(t_emb_dim, t_emb_dim),
        )

        self.in_conv = nn.Conv2d(in_ch, base_ch, 3, padding=1)

        # Down path
        self.down_res = nn.ModuleList()
        self.down_attn = nn.ModuleList()
        self.downsample = nn.ModuleList()

        ch_in = base_ch
        skip_chs = [base_ch]
        cur_res = img_size
        for i, m in enumerate(ch_mult):
            ch_out = base_ch * m
            level_res = nn.ModuleList()
            level_attn = nn.ModuleList()
            for _ in range(num_res_blocks):
                level_res.append(ResBlock(ch_in, ch_out, t_emb_dim, dropout))
                ch_in = ch_out
                level_attn.append(AttnBlock(ch_in) if cur_res in attn_resolutions else nn.Identity())
                skip_chs.append(ch_in)
            self.down_res.append(level_res)
            self.down_attn.append(level_attn)
            if i != len(ch_mult) - 1:
                self.downsample.append(Downsample(ch_in))
                skip_chs.append(ch_in)
                cur_res //= 2
            else:
                self.downsample.append(nn.Identity())

        # Middle
        self.mid_res1 = ResBlock(ch_in, ch_in, t_emb_dim, dropout)
        self.mid_attn = AttnBlock(ch_in)
        self.mid_res2 = ResBlock(ch_in, ch_in, t_emb_dim, dropout)

        # Up path
        self.up_res = nn.ModuleList()
        self.up_attn = nn.ModuleList()
        self.upsample = nn.ModuleList()

        for i, m in reversed(list(enumerate(ch_mult))):
            ch_out = base_ch * m
            level_res = nn.ModuleList()
            level_attn = nn.ModuleList()
            for _ in range(num_res_blocks + 1):
                skip_ch = skip_chs.pop()
                level_res.append(ResBlock(ch_in + skip_ch, ch_out, t_emb_dim, dropout))
                ch_in = ch_out
                level_attn.append(AttnBlock(ch_in) if cur_res in attn_resolutions else nn.Identity())
            self.up_res.append(level_res)
            self.up_attn.append(level_attn)
            if i != 0:
                self.upsample.append(Upsample(ch_in))
                cur_res *= 2
            else:
                self.upsample.append(nn.Identity())

        self.out_norm = nn.GroupNorm(min(32, ch_in), ch_in)
        self.out_conv = nn.Conv2d(ch_in, in_ch, 3, padding=1)

    def forward(self, x, t):
        t_emb = self.t_mlp(timestep_embedding(t, self.base_ch))

        h = self.in_conv(x)
        skips = [h]

        for i in range(self.num_resolutions):
            for res, attn in zip(self.down_res[i], self.down_attn[i]):
                h = res(h, t_emb)
                h = attn(h)
                skips.append(h)
            if i != self.num_resolutions - 1:
                h = self.downsample[i](h)
                skips.append(h)

        h = self.mid_res1(h, t_emb)
        h = self.mid_attn(h)
        h = self.mid_res2(h, t_emb)

        for idx, i in enumerate(reversed(range(self.num_resolutions))):
            for res, attn in zip(self.up_res[idx], self.up_attn[idx]):
                h = torch.cat([h, skips.pop()], dim=1)
                h = res(h, t_emb)
                h = attn(h)
            if i != 0:
                h = self.upsample[idx](h)

        return self.out_conv(F.silu(self.out_norm(h)))
