import torch
from torch import nn
from torch.nn import functional as F
from attention import SelfAttention, CrossAttention

class TimeEmbedding(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.dense_1 = nn.Linear(embed_dim, 4 * embed_dim)
        self.dense_2 = nn.Linear(4 * embed_dim, 4 * embed_dim)

    def forward(self, x):
        # x: (1, 320)
        # (1, 320) -> (1, 1280)
        x = self.dense_1(x)
        x = F.silu(x) 
        x = self.dense_2(x)

        return x

class UNET_ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, n_time=1280):
        super().__init__()
        self.groupnorm_latent = nn.GroupNorm(32, in_channels)
        self.conv_latent = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
        self.dense_time = nn.Linear(n_time, out_channels)

        self.groupnorm_merged = nn.GroupNorm(32, out_channels)
        self.conv_merged = nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1)

        if in_channels == out_channels:
            self.residual_layer = nn.Identity()
        else:
            self.residual_layer = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0)
    
    def forward(self, latent, time):
        # latent: (Batch_Size, In_Channels, Height, Width)
        # time: (1, 1280)
        residue = latent

        latent = self.groupnorm_latent(latent)
        latent = F.silu(latent)
        latent = self.conv_latent(latent)
        
        # (1, 1280) -> (1, 1280)
        time = F.silu(time)
        # (1, 1280) -> (1, Out_Channels)
        time = self.dense_time(time)
        
        # Add width and height dimension to time. 
        # (Batch_Size, Out_Channels, Height, Width) + (1, Out_Channels, 1, 1) -> (Batch_Size, Out_Channels, Height, Width)
        merged = latent + time.unsqueeze(-1).unsqueeze(-1)
        
        merged = self.groupnorm_merged(merged)
        merged = F.silu(merged)
        merged = self.conv_merged(merged)
        
        # (Batch_Size, Out_Channels, Height, Width) + (Batch_Size, Out_Channels, Height, Width) -> (Batch_Size, Out_Channels, Height, Width)
        return merged + self.residual_layer(residue)

class UNET_AttentionBlock(nn.Module):
    def __init__(self, n_head: int, embed_dim: int, context_dim=768):
        super().__init__()
        channels = n_head * embed_dim
        
        self.groupnorm = nn.GroupNorm(32, channels, eps=1e-6)
        self.conv_in = nn.Conv2d(channels, channels, kernel_size=1, padding=0)

        self.layernorm_1 = nn.LayerNorm(channels)
        self.attention_1 = SelfAttention(n_head, channels, bias=False)
        self.layernorm_2 = nn.LayerNorm(channels)
        self.attention_2 = CrossAttention(n_head, channels, context_dim, bias=False)
        self.layernorm_3 = nn.LayerNorm(channels)
        self.dense_geglu_1  = nn.Linear(channels, 4 * channels * 2)
        self.dense_geglu_2 = nn.Linear(4 * channels, channels)

        self.conv_out = nn.Conv2d(channels, channels, kernel_size=1, padding=0)
    
    def forward(self, x, context):
        # x: (Batch_Size, latents, Height, Width)
        # context: (Batch_Size, Seq_Len, Dim)
        residue_long = x

        x = self.groupnorm(x)
        x = self.conv_in(x)
        
        n, c, h, w = x.shape
        x = x.view((n, c, h * w))
        x = x.transpose(-1, -2)
        
        # Normalization + Self-Attention with skip connection
        residue_short = x
        
        x = self.layernorm_1(x)
        x = self.attention_1(x)

        x += residue_short

        # Normalization + Cross-Attention with skip connection
        residue_short = x
        
        x = self.layernorm_2(x)
        x = self.attention_2(x, context)

        x += residue_short

        # Normalization + FFN with GeGLU and skip connection
        residue_short = x
        
        x = self.layernorm_3(x)
        x, gate = self.dense_geglu_1(x).chunk(2, dim=-1) 
        # Element-wise product: (Batch_Size, Height * Width, latents * 4) * (Batch_Size, Height * Width, latents * 4) -> (Batch_Size, Height * Width, latents * 4)
        x = x * F.gelu(gate)
        x = self.dense_geglu_2(x)

        x += residue_short

        x = x.transpose(-1, -2)
        x = x.view((n, c, h, w))

        # Final skip connection between initial input and out of the block
        # (Batch_Size, latents, Height, Width) + (Batch_Size, latents, Height, Width) -> (Batch_Size, latents, Height, Width)
        return self.conv_out(x) + residue_long

class Upsample(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # (Batch_Size, latents, Height, Width) -> (Batch_Size, latents, Height * 2, Width * 2)
        x = F.interpolate(x, scale_factor=2, mode='nearest') 
        return self.conv(x)

class SwitchSequential(nn.Sequential):
    def forward(self, latent, context, time):
        for layer in self:
            if isinstance(layer, UNET_AttentionBlock):
                x = layer(latent, context)
            elif isinstance(layer, UNET_ResidualBlock):
                x = layer(latent, time)
            else:
                x = layer(latent)
        return x

class UNET(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoders = nn.ModuleList([
            # (Batch_Size, 4, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 8, Width / 8)
            SwitchSequential(nn.Conv2d(4, 320, kernel_size=3, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(320, 320), UNET_AttentionBlock(8, 40)),
            
            # (Batch_Size, 320, Height / 8, Width / 8) -> (Batch_Size, 320, Height / 16, Width / 16)
            SwitchSequential(nn.Conv2d(320, 320, kernel_size=3, stride=2, padding=1)),
            SwitchSequential(UNET_ResidualBlock(320, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(640, 640), UNET_AttentionBlock(8, 80)),
            
            # (Batch_Size, 640, Height / 16, Width / 16) -> (Batch_Size, 640, Height / 32, Width / 32)
            SwitchSequential(nn.Conv2d(640, 640, kernel_size=3, stride=2, padding=1)),            
            SwitchSequential(UNET_ResidualBlock(640, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280), UNET_AttentionBlock(8, 160)),
            
            # (Batch_Size, 1280, Height / 32, Width / 32) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(nn.Conv2d(1280, 1280, kernel_size=3, stride=2, padding=1)),            
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
            SwitchSequential(UNET_ResidualBlock(1280, 1280)),
        ])

        self.bottleneck = SwitchSequential(
            # (Batch_Size, 1280, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            UNET_ResidualBlock(1280, 1280), 

            UNET_AttentionBlock(8, 160), 
  
            UNET_ResidualBlock(1280, 1280), 
        )
        
        self.decoders = nn.ModuleList([
            # (Batch_Size, 2560, Height / 64, Width / 64) -> (Batch_Size, 1280, Height / 64, Width / 64)
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280)),
            
            SwitchSequential(UNET_ResidualBlock(2560, 1280), Upsample(1280)),
            
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),
            SwitchSequential(UNET_ResidualBlock(2560, 1280), UNET_AttentionBlock(8, 160)),

            SwitchSequential(UNET_ResidualBlock(1920, 1280), UNET_AttentionBlock(8, 160), Upsample(1280)),
 
            SwitchSequential(UNET_ResidualBlock(1920, 640), UNET_AttentionBlock(8, 80)),
            SwitchSequential(UNET_ResidualBlock(1280, 640), UNET_AttentionBlock(8, 80)),
            
            SwitchSequential(UNET_ResidualBlock(960, 640), UNET_AttentionBlock(8, 80), Upsample(640)),

            SwitchSequential(UNET_ResidualBlock(960, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
            SwitchSequential(UNET_ResidualBlock(640, 320), UNET_AttentionBlock(8, 40)),
        ])

    def forward(self, latent, context, time):
        # x: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim) 
        # time: (1, 1280)

        skip_connections = []
        for layers in self.encoders:
            latent = layers(latent, context, time)
            skip_connections.append(latent)

        latent = self.bottleneck(latent, context, time)

        for layers in self.decoders:
            # Since we always concat with the skip connection of the encoder, the number of latents increases before being sent to the decoder's layer
            latent = torch.cat((latent, skip_connections.pop()), dim=1) 
            x = layers(latent, context, time)
        
        return x


class UNET_FInalLayer(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groupnorm = nn.GroupNorm(32, in_channels)
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1)
    
    def forward(self, x):
        # x: (Batch_Size, 320, Height / 8, Width / 8)
        x = self.groupnorm(x)
        
        x = F.silu(x)
        
        x = self.conv(x)
        
        return x

class Diffusion(nn.Module):
    def __init__(self):
        super().__init__()
        self.time_embedding = TimeEmbedding(320)
        self.unet = UNET()
        self.final = UNET_FInalLayer(320, 4)
    
    def forward(self, latent, context, time):
        # latent: (Batch_Size, 4, Height / 8, Width / 8)
        # context: (Batch_Size, Seq_Len, Dim)
        # time: (1, 320)

        # (1, 320) -> (1, 1280)
        time = self.time_embedding(time)
        
        out = self.unet(latent, context, time)

        out = self.final(out)

        return out