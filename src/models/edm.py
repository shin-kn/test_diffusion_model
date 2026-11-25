import torch
from torch import nn


class EDMDenoiser(nn.Module):
    def __init__(self, net: nn.Module, sigma_data: float = 0.5) -> None:
        super().__init__()
        self.net = net
        self.sigma_data = sigma_data

    def forward(self, x: torch.Tensor, sigma: torch.Tensor) -> torch.Tensor:
        s = sigma[:, None, None]
        c_skip = self.sigma_data**2 / (s**2 + self.sigma_data**2)
        c_out = s * self.sigma_data / (s**2 + self.sigma_data**2).sqrt()
        c_in = 1 / (s**2 + self.sigma_data**2).sqrt()
        c_noise = sigma.log() / 4
        return c_skip * x + c_out * self.net(c_in * x, c_noise)

    def loss_weight(self, sigma: torch.Tensor) -> torch.Tensor:
        return (sigma**2 + self.sigma_data**2) / (sigma * self.sigma_data) ** 2


def sample_training_sigma(n: int, device: torch.device, p_mean: float = -1.2, p_std: float = 1.2) -> torch.Tensor:
    return (p_mean + p_std * torch.randn(n, device=device)).exp()
