import torch
import torch.nn as nn
    
def softmax(vector, index):
    exps = torch.exp(vector)
    numerator = exps[index]
    denominator = torch.sum(exps)
    
    return numerator / denominator


def self_attention(query, keys, values, mask=None):
        attn_logits = query @ keys.T
        attn_logits = attn_logits / query.size()[-1]
        
        # if batching of sentences
        if mask is not None:
            attn_logits = attn_logits.masked_fill(mask == 0, -9e15)
        
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
    def __init__(self, input_dim, embed_dim, num_heads):
        super().__init__()
        assert embed_dim % num_heads == 0, "The embedding dim must be modulo the number of heads"
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        
        # Project input dims into concatenated dims of q, k, and v
        self.qkv_proj = nn.Linear(input_dim, 3*embed_dim)
        # Project embedded dim into output dim (which is input)
        self.out_proj = nn.Linear(embed_dim, input_dim)
        
        # Initialize weights and biases
        nn.init.xavier_uniform_(self.qkv_proj.weight)
        self.qkv_proj.bias.data.fill_(0)
        nn.init.xavier_uniform_(self.o_proj.weight)
        self.o_proj.bias.data.fill_(0)


    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()
        if mask is not None:
            mask = expand_mask(mask)
        
        # Get projection (size: [batch_size, seq_len, 3*input_dim])
        qkv = self.qkv_proj(x)
        
        # Reshape to correct shape
        qkv = qkv.reshape(batch_size, seq_len, self.num_heads, 3*self.head_dim)  # since embed_dim = num_heads × head_dim
        
        # Permute heads for parallel processing
        qkv = qkv.permute(0, 2, 1, 3)  # [batch, num_heads, seq_len, 3*head_dim]
        
        # Get a, k, and v from last dimension of qkv
        q, k, v = qkv.chunk(3, dim=-1)
        
        # Compute attention
        values, attention = self_attention(q, k, v, mask=mask)
        
        # Inverse reshaping and permuting to get output
        values = values.permute(0, 2, 1, 3)
        values = values.reshape(batch_size, seq_len, self.embed_dim)
        out = self.out_proj(values)
        
        return out, attention