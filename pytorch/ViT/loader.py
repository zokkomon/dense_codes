#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 23:36:40 2023

@author: zok
"""
import torchvision.datasets as datasets
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import config

transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
dataset = datasets.MNIST(root="dataset/",transform=transform,download=True)
loader = DataLoader(dataset, batch_size=config.BATCH_SIZE, shuffle=True)