import torch
from src.config import device

class NoiseSchedule:
    def __init__(self, num_timesteps: int):
        self.num_timesteps = num_timesteps
        self.betas = self.cosine_schedule(num_timesteps)
        self.alphas = 1 - self.betas
        self.alpha_bars = torch.cumprod(self.alphas, dim=0)
        self.sqrt_alpha_bars = torch.sqrt(self.alpha_bars)
        self.sqrt_one_minus_alpha_bars = torch.sqrt(1 - self.alpha_bars)

    def cosine_schedule(self, num_timesteps: int, s: float = 0.008) -> torch.Tensor:
        alpha_t = torch.linspace(0, num_timesteps, num_timesteps + 1)
        alpha_bars = torch.cos(((alpha_t / num_timesteps) + s) / (1 + s) * torch.pi * 0.5) ** 2
        alpha_bars = alpha_bars / alpha_bars[0]
        betas = 1 - (alpha_bars[1:] / alpha_bars[:-1])
        betas = betas.clamp(0.0001, 0.9999)  # Prevent numerical issues
        
        return betas.to(device)

    def add_noise(self, x: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        sqrt_alpha_bar_t = self.sqrt_alpha_bars[t].view(-1, 1, 1, 1)
        sqrt_one_minus_alpha_bar_t = self.sqrt_one_minus_alpha_bars[t].view(-1, 1, 1, 1)
        x_t = sqrt_alpha_bar_t * x + sqrt_one_minus_alpha_bar_t * noise
        return x_t

    def sample_timesteps(self, batch_size: int) -> torch.Tensor:
        return torch.randint(0, self.num_timesteps, (batch_size,), device=device)