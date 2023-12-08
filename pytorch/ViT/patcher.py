#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 20:28:23 2023

@author: zok
"""
import torch
import config
from torch import nn

class PatchEmbedding(nn.Module):
    def __init__(self, embed_dim, patch_size, num_patches, dropout, in_):
        super().__init__()
        self.patcher = nn.Sequential(
            nn.Conv2d(
                in_,
                embed_dim,
                patch_size,
                patch_size,
            ),                  
            nn.Flatten(2))

        self.cls_token = nn.Parameter(torch.randn(size=(1, 1, embed_dim)), requires_grad=True)
        self.position_embeddings = nn.Parameter(torch.randn(size=(1, num_patches+1, embed_dim)), requires_grad=True)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        cls_token = self.cls_token.expand(x.shape[0], -1, -1)

        x = self.patcher(x).permute(0, 2, 1)
        x = torch.cat([cls_token, x], dim=1)
        x = self.position_embeddings + x 
        x = self.dropout(x)
        return x
    
# model = PatchEmbedding(config.EMBED_DIM, config.PATCH_SIZE, config.NUM_PATCHES, config.DROPOUT, config.IN_CHANNELS).to(config.device)
# x = torch.randn(512, 1, 28, 28).to(config.device)
# print(model(x).shape)
