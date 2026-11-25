import math

import torch
from torch import nn


def sinusoidal_embedding(t: torch.Tensor, dim: int) -> torch.Tensor:
    half = dim // 2
    freqs = torch.exp(-math.log(10_000) * torch.arange(half, device=t.device) / half)
    angles = t.float()[:, None] * freqs[None, :]
    return torch.cat([angles.sin(), angles.cos()], dim=-1)


class ResBlock(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, time_dim: int) -> None:
        super().__init__()
        self.conv1 = nn.Conv1d(in_ch, out_ch, kernel_size=3, padding=1)
        self.norm1 = nn.GroupNorm(8, out_ch)
        self.time_proj = nn.Linear(time_dim, out_ch)
        self.conv2 = nn.Conv1d(out_ch, out_ch, kernel_size=3, padding=1)
        self.norm2 = nn.GroupNorm(8, out_ch)
        self.act = nn.SiLU()
        self.skip = nn.Conv1d(in_ch, out_ch, kernel_size=1) if in_ch != out_ch else nn.Identity()

    def forward(self, x: torch.Tensor, t_emb: torch.Tensor) -> torch.Tensor:
        h = self.act(self.norm1(self.conv1(x)))
        h = h + self.time_proj(t_emb)[:, :, None]
        h = self.act(self.norm2(self.conv2(h)))
        return h + self.skip(x)


class UNet1D(nn.Module):
    def __init__(
        self,
        in_channels: int = 1,
        base_channels: int = 32,
        channel_mults: tuple[int, ...] = (1, 2, 4, 8, 8),
        time_dim: int = 128,
    ) -> None:
        super().__init__()
        self.n_downsamples = len(channel_mults) - 1
        self.time_mlp = nn.Sequential(
            nn.Linear(time_dim, time_dim * 4),
            nn.SiLU(),
            nn.Linear(time_dim * 4, time_dim * 4),
        )
        self.time_dim = time_dim
        emb_dim = time_dim * 4

        chs = [base_channels * m for m in channel_mults]
        self.stem = nn.Conv1d(in_channels, chs[0], kernel_size=3, padding=1)

        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for i, ch in enumerate(chs):
            self.down_blocks.append(ResBlock(chs[i - 1] if i > 0 else chs[0], ch, emb_dim))
            if i < self.n_downsamples:
                self.downsamples.append(nn.Conv1d(ch, ch, kernel_size=4, stride=2, padding=1))

        self.mid_block = ResBlock(chs[-1], chs[-1], emb_dim)

        self.upsamples = nn.ModuleList()
        self.up_blocks = nn.ModuleList()
        for i in reversed(range(self.n_downsamples)):
            self.upsamples.append(nn.ConvTranspose1d(chs[i + 1], chs[i + 1], kernel_size=4, stride=2, padding=1))
            self.up_blocks.append(ResBlock(chs[i + 1] + chs[i], chs[i], emb_dim))

        self.head = nn.Sequential(
            nn.GroupNorm(8, chs[0]),
            nn.SiLU(),
            nn.Conv1d(chs[0], in_channels, kernel_size=3, padding=1),
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        length = x.shape[-1]
        n_pad = -length % (2**self.n_downsamples)
        if n_pad > 0:
            x = nn.functional.pad(x, (0, n_pad), mode="replicate")
        t_emb = self.time_mlp(sinusoidal_embedding(t, self.time_dim))

        h = self.stem(x)
        skips = []
        for i, block in enumerate(self.down_blocks):
            h = block(h, t_emb)
            if i < self.n_downsamples:
                skips.append(h)
                h = self.downsamples[i](h)

        h = self.mid_block(h, t_emb)

        for upsample, block in zip(self.upsamples, self.up_blocks):
            h = upsample(h)
            h = block(torch.cat([h, skips.pop()], dim=1), t_emb)

        return self.head(h)[..., :length]
