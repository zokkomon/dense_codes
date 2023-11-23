import torch
import torch.nn as nn
from attention import SelfAttention

class CLIPEmbedding(nn.Module):
    def __init__(self, vocabs: int, embed_dim: int, tokens: int):
        super().__init__()
        
        self.token_embedding = nn.Embedding(vocabs, embed_dim)
        # A learnable weight matrix encodes the position information for each token
        self.position_embedding = nn.Parameter(torch.zeros((tokens, embed_dim)))
        
    def forward(self, tokens):
        x = self.token_embedding(tokens)
        x += self.position_embedding
        
        return x

class CLIPLayer(nn.Module):
    def __init(self, heads:int, embed_dim: int):
        super().__init__()
        
        # Attentionlayer
        self.layernorm_1 = nn.LayerNorm(embed_dim) 
        self.attention = SelfAttention(heads, embed_dim) 
        
        # FeedForward layer
        self.layernorm_2 = nn.LayerNorm(embed_dim) 
        self.dense = nn.Linear(embed_dim, 4* embed_dim)
        self.dense_final = nn.Linear(4* embed_dim, embed_dim )
        
    def forward(self, x):
        # (Batch_Size, Seq_Len, Dim)
       residue = x
       
       # Attentionlayer
       x = self.layernorm_1(x)
       
       x = self.attention(x, mask=True)
       
       x +=residue
       
       # FeedForward layer
       residue = x
       x = self.layernorm_2(x)
       
       x = self.dense(x)
       x = x * torch.sigmoid(1.702 * x)   # QuickGELU activation function
       
       x =self.dense_final(x)
       x +=residue
       
       return x
       
class CLIP(nn.Module):
    def __init__(self):
        super().__init__()
        
        self.embedding = CLIPEmbedding(49408, 768, 77)
        self.layers = nn.ModuleList([
            CLIPLayer(12, 768) for i in range(12)
            ])
        self.layernorm = nn.LayerNorm(768)
        
    def forward(self, tokens: torch.LongTensor) :
       tokens = tokens.type(torch.long)
       
       # (Batch_Size, Seq_Len) -> (Batch_Size, Seq_Len, Dim)
       state = self.embedding(tokens)

       # Apply encoder layers similar to the Transformer's encoder.
       for layer in self.layers:            
           state = layer(state)
       
       out = self.layernorm(state)
       
       return out