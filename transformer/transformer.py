import torch
import torch.nn as nn
from attentions import MultiHeadAttention
from positional_encoding import PositionalEncoding


class FFN(nn.Module):
    def __init__(self, embed_dim, dim_ffn, dropout=0.0):
        super().__init__()
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, dim_ffn),
            nn.Dropout(dropout),
            nn.ReLU(inplace=True),
            nn.Linear(dim_ffn, embed_dim)
        )
    def forward(self, x):
        return self.ffn(x)


class Encoder(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_ffn, dropout=0.0):
        super().__init__()
         
        self.multi_head_attention = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = FFN(embed_dim, dim_ffn)
        
        self.dropout = nn.Dropout(dropout)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
    
    def forward(self, x, mask=None):
        attn_out = self.multi_head_attention(x, mask)
        x = x + self.dropout(attn_out)
        x = self.layernorm1(x)
        
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.layernorm2(x)
        
        return x


class Decoder(nn.Module):
    def __init__(self, embed_dim, num_heads, dim_ffn, dropout=0.0):
        super().__init__()
        self.masked_attn = MultiHeadAttention(embed_dim, num_heads)
        self.cross_attn = MultiHeadAttention(embed_dim, num_heads)
        self.ffn = FFN(embed_dim, dim_ffn)
        
        self.dropout = nn.Dropout(dropout)
        self.layernorm1 = nn.LayerNorm(embed_dim)
        self.layernorm2 = nn.LayerNorm(embed_dim)
        self.layernorm3 = nn.LayerNorm(embed_dim)
    
    def forward(self, x, enc_mask, enc_output, dec_mask):
        self_attn = self.masked_attn(x, mask=dec_mask)
        x = x + self.dropout(self_attn)
        x = self.layernorm1(x)
        
        cross_attn = self.cross_attn(x, kv=enc_output, mask=enc_mask)
        x = x + self.dropout(cross_attn)
        x = self.layernorm2(x)
        
        ffn_out = self.ffn(x)
        x = x + self.dropout(ffn_out)
        x = self.layernorm3(x)
        
        return x


class Transformer(nn.Module):
    
    def __init__(self, src_vocab_size, tgt_vocab_size, src_max_seq_len, tgt_max_seq_len, embed_dim, num_heads, dim_ffn, num_layers=2, input_dropout=0.0):
        super().__init__()

        # Store hyperparameters
        self.hparams = type('HParams', (), {
            'src_vocab_size': src_vocab_size,
            'tgt_vocab_size': tgt_vocab_size,
            'src_max_seq_len': src_max_seq_len,
            'tgt_max_seq_len': tgt_max_seq_len,
            'embed_dim': embed_dim,
            'num_heads': num_heads,
            'dim_ffn': dim_ffn,
            'num_layers': num_layers,
            'input_dropout': input_dropout
        })()
        self.embed_scaler = torch.sqrt(torch.tensor(embed_dim, dtype=torch.float))

        self._create_model()
    

    def _create_model(self):
        # Input Embedding
        self.input_embedding = nn.Sequential(
            nn.Embedding(self.hparams.src_vocab_size, self.hparams.embed_dim),
            nn.Dropout(self.hparams.input_dropout)
        )

        # Output embedding
        self.output_embedding = nn.Sequential(
            nn.Embedding(self.hparams.tgt_vocab_size, self.hparams.embed_dim),
            nn.Dropout(self.hparams.input_dropout)
        )
        
        # Positional Encoding
        self.src_pe = PositionalEncoding(max_seq_len=self.hparams.src_max_seq_len, dim_model=self.hparams.embed_dim)
        self.tgt_pe = PositionalEncoding(max_seq_len=self.hparams.tgt_max_seq_len, dim_model=self.hparams.embed_dim)
        
        # Encoder and Decoder
        self.full_encoder = nn.ModuleList([Encoder(
                                                self.hparams.embed_dim,
                                                self.hparams.num_heads,
                                                self.hparams.dim_ffn
                                                )
                                            for _ in range(self.hparams.num_layers)])
        self.full_decoder = nn.ModuleList([Decoder(
                                                self.hparams.embed_dim,
                                                self.hparams.num_heads,
                                                self.hparams.dim_ffn
                                                )
                                            for _ in range(self.hparams.num_layers)])

        # Linear layer
        self.linear = nn.Linear(self.hparams.embed_dim, self.hparams.tgt_vocab_size)


    def generate_mask(self, src, tgt=None):
        # Where we have padding (value in tensor = 0) we do not compute mask
        src_mask  = (src != 0).unsqueeze(1).unsqueeze(2)  # To get acceptable shape for attention computation

        if tgt is not None:
            tgt_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)

            tg_seq_len = tgt.size(1)

            # Create no look ahead target mask
            nopeak_mask = (1 - torch.triu(torch.ones(1, tg_seq_len, tg_seq_len, device=tgt.device), diagonal=1)).bool()  # Create lower triangular matrix of bools
            tgt_mask = tgt_mask & nopeak_mask

            return src_mask, tgt_mask

        return src_mask


    def train(self, x, y):
        # Get masks first (needs token IDs, not embeddings)
        src_mask, tgt_mask = self.generate_mask(x, y)

        # Go through input and output embeddings
        x_embed = self.input_embedding(x) * self.embed_scaler
        y_embed = self.output_embedding(y) * self.embed_scaler

        # Add positional encoding
        x_embed = x_embed + self.src_pe(x_embed)
        y_embed = y_embed + self.tgt_pe(y_embed)

        # Encoder blocks - loop through each layer
        enc_out = x_embed
        for enc_layer in self.full_encoder:
            enc_out = enc_layer(enc_out, mask=src_mask)

        # Decoder blocks - loop through each layer
        dec_out = y_embed
        for dec_layer in self.full_decoder:
            dec_out = dec_layer(dec_out, enc_mask=src_mask, enc_output=enc_out, dec_mask=tgt_mask)

        # Apply linear layer (return logits, no softmax)
        out = self.linear(dec_out)

        return out
    
    
    def generate(self, x, max_new_tokens, temperature=1.0):
        # Encode source sequence once
        src_mask = self.generate_mask(x)
        x_embed = self.input_embedding(x) * self.embed_scaler
        x_embed = x_embed + self.src_pe(x_embed)

        enc_out = x_embed
        for enc_layer in self.full_encoder:
            enc_out = enc_layer(enc_out, mask=src_mask)

        # Initialize with start token (assuming token ID 1 is <BOS>)
        batch_size = x.size(0)
        generated = torch.ones((batch_size, 1), dtype=torch.long, device=x.device)

        # Autoregressive generation loop
        for _ in range(max_new_tokens):
            # Generate mask for current sequence
            tgt_mask = self.generate_mask(x, generated)[1]

            # Embed and add positional encoding
            y_embed = self.output_embedding(generated) * self.embed_scaler
            y_embed = y_embed + self.tgt_pe(y_embed)

            # Decoder blocks
            dec_out = y_embed
            for dec_layer in self.full_decoder:
                dec_out = dec_layer(dec_out, enc_mask=src_mask, enc_output=enc_out, dec_mask=tgt_mask)

            # Get logits for next token (only need last position)
            logits = self.linear(dec_out[:, -1, :])

            # Apply temperature and sample/select next token
            if temperature == 0.0:
                # Greedy decoding
                next_token = torch.argmax(logits, dim=-1, keepdim=True)
            else:
                # Temperature sampling
                probs = torch.softmax(logits / temperature, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)

            # Append to generated sequence
            generated = torch.cat([generated, next_token], dim=1)
 
        return generated