import torch
from torch import nn
from torch.nn import functional as F
from decoder import VAE_AttentionBlock, VAE_ResidualBlock

class VAE_Encoder(nn.Sequential):
    def __init__(self):
        super(VAE_Encoder, self).__init__()
        self.encoder = nn.Sequential(
            # input: N x img_channels x h x w
            nn.Conv2d(3, 128, kernel_size=3, padding=1),
            
            VAE_ResidualBlock(128, 128),
            VAE_ResidualBlock(128, 128),
            
            nn.Conv2d(128, 128, kernel_size=3, stride=2, padding=0),

            VAE_ResidualBlock(128, 256), 
            VAE_ResidualBlock(256, 256), 

            nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=0), 

            VAE_ResidualBlock(256, 512), 
            VAE_ResidualBlock(512, 512), 
 
            nn.Conv2d(512, 512, kernel_size=3, stride=2, padding=0), 

            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            VAE_ResidualBlock(512, 512), 
            
            VAE_AttentionBlock(512), 

            VAE_ResidualBlock(512, 512), 

            nn.GroupNorm(32, 512), 
 
            nn.SiLU(), 

            nn.Conv2d(512, 8, kernel_size=3, padding=1), 
            nn.Conv2d(8, 8, kernel_size=1, padding=0), 
        )

def forward(self, x, noise):
    # x: (Batch_Size, Channel, Height, Width)
    # noise: (Batch_Size, 4, Height / 8, Width / 8)
    
    x = self.disc(x)

    for module in self:

        if getattr(module, 'stride', None) == (2, 2):  # Padding at downsampling should be asymmetric (see #8)
            # Pad: (Padding_Left, Padding_Right, Padding_Top, Padding_Bottom).
            # Pad with zeros on the right and bottom.
            # (Batch_Size, Channel, Height, Width) -> (Batch_Size, Channel, Height + Padding_Top + Padding_Bottom, Width + Padding_Left + Padding_Right) = (Batch_Size, Channel, Height + 1, Width + 1)
            x = F.pad(x, (0, 1, 0, 1))
        
        x = module(x)
    # (Batch_Size, 8, Height / 8, Width / 8) -> two tensors of shape (Batch_Size, 4, Height / 8, Width / 8)
    mean, log_variance = torch.chunk(x, 2, dim=1)
    log_variance = torch.clamp(log_variance, -30, 20)
    variance = log_variance.exp()
    stdev = variance.sqrt()
    
    # Transform N(0, 1) -> N(mean, stdev) 
    x = mean + stdev * noise

    x *= 0.18215
    
    return x
 
        