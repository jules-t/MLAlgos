import torch


class NoiseSchedule:
    def __init__(self, num_timesteps: int, beta_start: float, beta_end: float):
        self.num_timesteps = num_timesteps
        self.betas = torch.linspace(beta_start, beta_end, num_timesteps)
        self.alphas = 1 - self.betas
        
    def get_alpha_bars(self, t: int) -> float:
        return torch.prod(self.alphas[:t])
        
    def add_noise(self, x: torch.tensor, t: int, noise: torch.tensor) -> torch.tensor:
        alpha_bar_t = self.get_alpha_bars(t).unsqueeze(1)
        
        x_t = torch.sqrt(alpha_bar_t) * x + torch.sqrt(1 - alpha_bar_t) * noise
        
        return x_t