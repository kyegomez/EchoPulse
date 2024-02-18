import torch 
from torch import nn, Tensor, einsum
from zeta.nn import MultiQueryAttention, FeedForward, Attention, threed_to_text
from einops import rearrange


class EchoBlock(nn.Module):
    def __init__(
        self,
        dim: int,
        heads: int,
        dim_head: int = 64,
        ff_mult: int = 4,
        dropout: float = 0.,
        pool_kernel: int = 2,
        stride: int = 2,
        padding: int = 0,
        *args,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.heads = heads
        self.dim_head = dim_head
        self.ff_mult = ff_mult
        self.dropout = dropout
        
        self.norm = nn.LayerNorm(dim)
        
        # multi-query attention
        self.attn = Attention(
            dim,
            dim_head,
            heads,
            causal=True,
            qk_norm=True,
            *args,
            **kwargs
        )
        
        # Feedforward
        self.ffn = FeedForward(
            dim,
            dim,
            ff_mult,
            *args,
            **kwargs
        )
        
        # Global pooling
        self.pool = nn.AvgPool1d(
            pool_kernel,
            stride =  stride,
            padding = padding,
        )
        
    def forward(self, x: Tensor):
        skip = x
        # x = self.norm(x) + x
        
        x, _ = self.attn(x)
        
        x = self.ffn(x) + x
        
        # Global pooling
        x = self.pool(x)
        
        return x
    
    
class EchoPulse(nn.Module):
    def __init__(
        self,
        dim: int,
        depth: int,
        heads: int,
        dim_head: int = 64,
        ff_mult: int = 4,
        dropout: float = 0.,
        pool_kernel: int = 2,
        stride: int = 2,
        padding: int = 0,
        *args,
        **kwargs
    ):
        super().__init__()
        self.dim = dim
        self.depth = depth
        self.heads = heads
        self.dim_head = dim_head
        self.ff_mult = ff_mult
        self.dropout = dropout
        self.pool_kernel = pool_kernel
        self.stride = stride
        self.padding = padding
        
        self.blocks = nn.ModuleList([])
        
        for _ in range(depth):
            self.blocks.append(
                EchoBlock(
                    dim,
                    heads,
                    dim_head,
                    ff_mult,
                    dropout,
                    pool_kernel,
                    stride,
                    padding,
                    *args,
                    **kwargs
                )
            )
        
    def forward(self, x: Tensor):
    # (Frequency, Signal Strength, SNR).
    # (Batch, Frequency, Signal Strength, SNR).
        b, f, s, r = x.shape
        x = rearrange(
            x, "b f s r -> b f (s r)",
        )
        print(x.shape)
        x = threed_to_text(
            x,
            s * r,
            self.dim,
        )
        
        
        for block in self.blocks:
            x = block(x)
        
        return x # (Batch, Frequency, Signal Strength, SNR).
    

    
x = torch.randn(1, 100, 100, 3)

# Model
model = EchoPulse(
    dim = 100,
    depth = 1,
    heads = 8,
    dim_head = 64,
    ff_mult = 4,
    dropout = 0.1,
    pool_kernel = 2,
    stride = 2,
    padding = 0,
)

# Forward
y = model(x)

print(y.shape)
