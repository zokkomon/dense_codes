import torch
import torch.nn as nn

class SelfAttention(nn.Module):
    def __init__(self, heads: int, embed_dim: int, plus_bias=True ):
        super().__init__()
        
        self.embed_dim = embed_dim
        self.heads = heads 
        self.head_dim = embed_dim // heads
        
        assert (self.head_dim * heads == embed_dim), "Embed size needs to div by heads"
        
        self.qkv = nn.Linear(embed_dim, 3* embed_dim, bias=plus_bias)
        self.dense = nn.Linear(heads*self.head_dim, embed_dim, bias=plus_bias)
        
    def forward(self, x, mask):
        # x.shape :(batch, Seq_Len, Dim)
        n_shape = x.shape
        n, seq_len, embed_dim = n_shape 
        
        # (Batch_Size, Seq_Len, H, emd_Dim / H)
        qkv = (n, seq_len, self.heads, self.embed_dim)
        
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Dim * 3) -> 3 tensor of shape (Batch_Size, Seq_Len, Dim)
        q, k, v = self.qkv(x).chunk(3, dim= -1)
        
        # (Batch_Size, Seq_Len, Dim) -> (Batch_Size, Seq_Len, Head, head_dim) -> (Batch_Size, H, Seq_Len, Dim / H)
        queries = q.view(qkv).transpose(1,2)
        keys = k.view(qkv).transpose(1,2)
        values = v.view(qkv).transpose(1,2)
        
        # (Batch_Size, H, Seq_Len, Dim) @ (Batch_Size, H, Dim, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = queries @ keys.transpose(-1, -2)
        
        if mask:
            # Mask where the upper triangle (above the principal diagonal) is 1
            mask = torch.ones_like(weight, dtype=torch.bool).triu(1)
            # Fill the upper triangle with -inf
            weight.masked_fill_(mask, -torch.inf)
        
        # (Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        attention = torch.softmax(weight / self.head_dim ** (1/2), dim=3)
        
        out = attention @ values
        
        # (Batch_Size, Seq_Len, H, Dim / H) -> (Batch_Size, Seq_Len, Dim)
        out = out.transpose(1, 2).reshape(n_shape)
        
        x = self.dense(out)
        return x
        
class CrossAttention(nn.Module):
    def __init__(self, heads, embed_dim, cross_dim, plus_bias=True):
        super().__init__()
        self.embed_dim = embed_dim
        self.heads = heads
        self.head_dim = embed_dim // heads
        
        assert (self.head_dim * heads == embed_dim), "Embed size needs to div by heads"
        
        self.values = nn.Linear(self.head_dim, self.head_dim, bias=plus_bias) 
        self.keys = nn.Linear(self.head_dim, self.head_dim, bias=plus_bias)
        self.queries = nn.Linear(self.head_dim, self.head_dim, bias=plus_bias)
        self.dense = nn.Linear(heads*self.head_dim, embed_dim, bias=plus_bias )
          
        
    def forward(self, queries, values):
        # x.shape :(batch, Seq_Len, Dim)
        n_shape = queries.shape
        n, seq_len, embed_dim = n_shape 
        
        # Divide each embedding of Q into multiple heads such that d_heads * n_heads = Dim_Q
        qkv = (n, -1, self.heads, self.embed_dim)      
        
        values = self.values(values)
        keys = self.keys(values)
        queries = self.queries(queries)
        
        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_Q, Dim_Q / H)
        queries = queries.view(qkv).transpose(1,2)
        # (Batch_Size, Seq_Len_KV, Dim_Q) -> (Batch_Size, Seq_Len_KV, H, Dim_Q / H) -> (Batch_Size, H, Seq_Len_KV, Dim_Q / H)
        keys = keys.view(qkv).transpose(1,2)
        values = values.view(qkv).transpose(1,2)
        
        # (Batch_Size, H, Seq_Len, Dim) @ (Batch_Size, H, Dim, Seq_Len) -> (Batch_Size, H, Seq_Len, Seq_Len)
        weight = queries @ keys.transpose(-1, -2)
        
        # (Batch_Size, H, Seq_Len, Seq_Len) @ (Batch_Size, H, Seq_Len, Dim / H) -> (Batch_Size, H, Seq_Len, Dim / H)
        attention = torch.softmax(weight / self.head_dim ** (1/2), dim=3)
        
        out = attention @ values
        
        # (Batch_Size, H, Seq_Len_Q, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, H, Dim_Q / H) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        out = out.transpose(1, 2).contiguous().view(n_shape)       
        
        # (Batch_Size, Seq_Len_Q, Dim_Q) -> (Batch_Size, Seq_Len_Q, Dim_Q)
        x = self.dense(out)
        
        # (Batch_Size, Seq_Len_Q, Dim_Q)
        return x
      
        
