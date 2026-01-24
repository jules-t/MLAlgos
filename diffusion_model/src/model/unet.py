# U-Net architecture for noise prediction
# TODO: Implement U-Net with time embedding
import torch
import torch.nn as nn


class TimeEmbeding(nn.Module):
    def __init__(self, time_dim: int):
        super().__init__()
        self.time_dim = time_dim
    
    def time_embedding(self, t: torch.Tensor):
        # Implement sinusoidal time embedding
        half_dim = self.time_dim // 2
        denominator = - torch.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=t.device) * denominator)
        emb = t[:, None] * emb[None, :]
        t_e = torch.cat((torch.sin(emb), torch.cos(emb)), dim=1)
        return t_e

    def time_mlp(self, t: torch.Tensor):
        t_e = nn.Sequential(
            nn.Linear(self.time_dim, self.time_dim * 4),  # Using this dimensions because of convention
            nn.SiLU(),  # SiLu to have a smooth activation function that does not cut at 0 like ReLu
            nn.Linear(self.time_dim * 4, self.time_dim),
        )
        return t_e
    
    def forward(self, t: torch.Tensor):
        t_e = self.time_embedding(t)
        t_e = self.time_mlp(t_e)
        return t_e


class UNet(nn.Module):
    def __init__(self, time_dim):
        super().__init__()
        self.time_dim = time_dim
        pass
    
    def encoder(self, x):
        pass
    
    def decoder(self, x):
        pass
    
    def bottleneck(self, x):
        pass
    
    def forward(self, x, t):
        t_e = TimeEmbeding(self.time_dim)(t)
        pass