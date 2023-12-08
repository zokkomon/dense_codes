#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 23:03:25 2023

@author: zok
"""
import torch.nn as nn
import config
from patcher import PatchEmbedding

class ViT(nn.Module):
    def __init__(self, num_patches, img_size, num_classes, patch_size, embed_dim, num_encoders, num_heads, hidden_dim, dropout, activation, in_channels):
        super().__init__()
        self.embeddings_block = PatchEmbedding(embed_dim, patch_size, num_patches, dropout, in_channels)
        
        encoder_layer = nn.TransformerEncoderLayer(d_model=embed_dim, nhead=num_heads, dropout=dropout, activation=activation, batch_first=True, norm_first=True)
        self.encoder_blocks = nn.TransformerEncoder(encoder_layer, num_layers=num_encoders)

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(normalized_shape=embed_dim),
            nn.Linear(in_features=embed_dim, out_features=num_classes)
        )

    def forward(self, x):
        x = self.embeddings_block(x)
        x = self.encoder_blocks(x)
        x = self.mlp_head(x[:, 0, :])  # Apply MLP on the CLS token only
        return x

# model = ViT(config.NUM_PATCHES, config.IMG_SIZE, config.NUM_CLASSES, config.PATCH_SIZE, config.EMBED_DIM, config.NUM_ENCODERS, config.NUM_HEADS, config.HIDDEN_DIM, config.DROPOUT, config.ACTIVATION, config.IN_CHANNELS).to(config.device)
# x = torch.randn(512, 1, 28, 28).to(config.device)
# print(model(x).shape) # BATCH_SIZE X NUM_CLASSES