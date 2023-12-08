#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 23:27:01 2023

@author: zok
"""


import torch
from torch import nn,save,load
from torch import optim
import timeit
from tqdm import tqdm
import config
from vit import ViT
from loader import loader

model = ViT(config.NUM_PATCHES, config.IMG_SIZE, config.NUM_CLASSES, config.PATCH_SIZE, config.EMBED_DIM, config.NUM_ENCODERS, config.NUM_HEADS, config.HIDDEN_DIM, config.DROPOUT, config.ACTIVATION, config.IN_CHANNELS).to(config.device)

criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), betas=config.ADAM_BETAS, lr=config.LEARNING_RATE, weight_decay=config.ADAM_WEIGHT_DECAY)

start = timeit.default_timer()
for epoch in tqdm(range(config.EPOCHS), position=0, leave=True):
    # model.train()
    train_labels = []
    train_preds = []
    train_running_loss = 0
    for idx, (img, tag) in enumerate(tqdm(loader, position=0, leave=True)):
        img = img.float().to(config.device)
        label = tag.type(torch.uint8).to(config.device)
        y_pred = model(img)
        y_pred_label = torch.argmax(y_pred, dim=1)

        train_labels.extend(label.cpu().detach())
        train_preds.extend(y_pred_label.cpu().detach())
        
        loss = criterion(y_pred, label)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_running_loss += loss.item()
    train_loss = train_running_loss / (idx + 1)

    # model.eval()
    val_labels = []
    val_preds = []
    val_running_loss = 0
    with torch.no_grad():
        for idx, (img, tag) in enumerate(tqdm(loader, position=0, leave=True)):
            img = img.float().to(config.device)
            label = tag.type(torch.uint8).to(config.device)         
            y_pred = model(img)
            y_pred_label = torch.argmax(y_pred, dim=1)
            
            val_labels.extend(label.cpu().detach())
            val_preds.extend(y_pred_label.cpu().detach())
            
            loss = criterion(y_pred, label)
            val_running_loss += loss.item()
    val_loss = val_running_loss / (idx + 1)

    print("-"*30)
    print(f"Train Loss EPOCH {epoch+1}: {train_loss:.4f}")
    print(f"Valid Loss EPOCH {epoch+1}: {val_loss:.4f}")
    print(f"Train Accuracy EPOCH {epoch+1}: {sum(1 for x,y in zip(train_preds, train_labels) if x == y) / len(train_labels):.4f}")
    print(f"Valid Accuracy EPOCH {epoch+1}: {sum(1 for x,y in zip(val_preds, val_labels) if x == y) / len(val_labels):.4f}")
    print("-"*30)

stop = timeit.default_timer()
print(f"Training Time: {stop-start:.2f}s")

with open('image_recognition.pt', 'wb') as f: 
    save(model.state_dict(), f) 
    
with open('image_recognition.pt', 'rb') as f: 
    model.load_state_dict(load(f))
