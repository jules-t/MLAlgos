import torch
import torch.nn as nn
import math


class PositionalEncoding(nn.Module):
    def __init__(self, max_seq_len, dim_model):
        super().__init__()
        # Create encoding tensor of one row per token and one column per model dimension
        pe = torch.zeros(max_seq_len, dim_model)

        # Create positions
        position = torch.arange(0, max_seq_len, dtype=torch.float).unsqueeze(1)

        # Set up division term
        div_term = torch.exp(torch.arange(0, dim_model, 2).float() * (-math.log(10000.0) / dim_model))

        # Get final positional encoding
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # Add batch size dimension

        self.register_buffer('pe', pe, persistent=False)

    def forward(self, x):
        return self.pe[:, :x.size(1)]