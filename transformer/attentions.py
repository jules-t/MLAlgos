import torch
import torch.nn as nn
from utils import softmax


def self_attention(query, keys, values, mask=None):
        attn_logits = query @ keys.T
        attn_logits = attn_logits / query.size()[-1]
        
        # if batching of sentences
        if mask is not None:
            attn_logits = attn_logits.masked_fill(~mask, -9e15)
        
        attention = softmax(attn_logits)
        values = attention @ values

        return values, attention
    
    
# Support different mask size
def expand_mask(mask):
    assert mask.ndim >= 2, "Mask must be at least 2-dimensional with seq_length x seq_length"
    if mask.ndim == 3:
        mask = mask.unsqueeze(1)
    while mask.ndim < 4:
        mask = mask.unsqueeze(0)
    return mask


class MultiHeadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "The embedding dim must be modulo the number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Project input dims into q, k, and v
        self.q_proj = nn.Linear(embed_dim, embed_dim)
        self.k_proj = nn.Linear(embed_dim, embed_dim)
        self.v_proj = nn.Linear(embed_dim, embed_dim)
        
        # Project embedded dim into output dim (which is input)
        self.out_proj = nn.Linear(embed_dim, embed_dim)
        
        # Initialize weights and biases
        nn.init.xavier_uniform_(self.q_proj.weight)
        self.q_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.k_proj.weight)
        self.k_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.v_proj.weight)
        self.v_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.out_proj.weight)
        self.out_proj.bias.data.fill_(0)


    def forward(self, x, kv=None, mask=None, get_attention=False):
        batch_size, seq_len, _ = x.size()
        if mask is not None:
            mask = expand_mask(mask)
        
        # Get projection
        if kv is None:
            q = self.q_proj(x)
            k = self.k_proj(x)
            v = self.v_proj(x)
            
            # Reshape and permute to get correct tensor
            q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            q = q.permute(0, 2, 1, 3)
            
            k = k.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            k = k.permute(0, 2, 1, 3)
            
            v = v.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            v = v.permute(0, 2, 1, 3)

        else:            
            q = self.q_proj(x)
            k = self.k_proj(kv)
            v = self.v_proj(kv)
            
            kv_seq_len = kv.size(1)
            
            # Reshape and permute to get correct tensor
            q = q.reshape(batch_size, seq_len, self.num_heads, self.head_dim)
            q = q.permute(0, 2, 1, 3)
            
            k = k.reshape(batch_size, kv_seq_len, self.num_heads, self.head_dim)
            k = k.permute(0, 2, 1, 3)
            
            v = v.reshape(batch_size, kv_seq_len, self.num_heads, self.head_dim)
            v = v.permute(0, 2, 1, 3)
        
        # Compute attention
        values, attention = self_attention(q, k, v, mask=mask)
        
        # Inverse reshaping and permuting to get output
        values = values.permute(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_len, self.embed_dim)
        out = self.out_proj(values)
        
        if get_attention:
            return out, attention
        
        return out