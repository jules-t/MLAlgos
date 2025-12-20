import torch
import torch.nn as nn
from attentions import MultiHeadAttention


class Encoder(nn.Module):
    def __init__(self, input_dim, embed_dim, num_heads, dim_ffn):
        super().__init__()
         
        self.multi_head_attention = MultiHeadAttention(input_dim, embed_dim, num_heads, dropout=0.0)
        
        self.ffn = nn.Sequential(
            nn.Linear(input_dim, dim_ffn)
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_ffn, input_dim)
        )
        
        self.dropout = nn.Dropout(dropout)
        self.layernorm1 = nn.LayerNorm(input_dim)
        self.layernorm2 = nn.LayerNorm(input_dim)
    
    def forward(self, x, mask=None):
        attn_out = self.multi_head_attention(x, mask=None)
        x = x + self.Dropout(attn_out)
        x = self.layernorm1(x)
        
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = x + self.layernorm2(x)
        
        return x