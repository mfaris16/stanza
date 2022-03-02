import torch.nn as nn

from stanza.models.constituency.positional_encoding import ConcatSinusoidalEncoding

class SimpleAttentionModule(nn.Module):
    def __init__(self,
                 n_layers,
                 n_heads,
                 d_input,
                 d_model,
                 d_timing):
        super().__init__()

        if d_model % n_heads != 0:
            d_model = d_model + n_heads - d_model % n_heads
            logger.warning("d_model % n_heads != 0.  changing d_model to %d", d_model)

        self.attn_proj = nn.Linear(d_input, d_model - d_timing)
        self.attn_timing = ConcatSinusoidalEncoding(d_model=d_timing)
        self.attn_layers = nn.ModuleList([nn.MultiheadAttention(d_model, n_heads, batch_first=True)
                                          for _ in range(n_layers)])

    def forward(self, x):
        x = self.attn_proj(x)
        x = self.attn_timing(x)

        for layer in self.attn_layers:
            # TODO: residual dropout if this is working at all
            x = layer(x, x, x)[0] + x
        return x

