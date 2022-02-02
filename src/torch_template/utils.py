import logging

import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm

from .data import CustomDataset


def train(dataloader, device, optimizer, model, loss_fn):
    for batch, (X, y) in tqdm(enumerate(dataloader)):
        X, y = X.to(device), y.to(device)

        # Clear gradients w.r.t. parameters
        optimizer.zero_grad()

        # Forward to get output
        pred = model(X)

        # Calculate Loss
        loss = loss_fn(pred, y)

        # Getting gradients w.r.t. parameters
        loss.backward()

        # Updating parameters
        optimizer.step()

        return loss


def test(dataloader, device, model, loss_fn):
    model.eval()
    with torch.no_grad():
        for batch, (X, y) in tqdm(enumerate(dataloader)):
            X, y = X.to(device), y.to(device)

            # Forward to get output
            pred = model(X)

            # Calculate Loss
            loss = loss_fn(pred, y)
            return loss


def generate_data(n=256, m=3, c=2, plot=False, test=True):
    if test:
        np.random.seed(24)

    x = np.random.rand(n)
    noise = np.random.randn(n) / 4

    y = m * x + c + noise

    x = x.astype("float32").reshape(-1, 1)
    y = y.astype("float32").reshape(-1, 1)

    if plot:
        plt.scatter(x, y)
        plt.show()

    return x, y


def get_dataloaders(batch_size):
    x, y = generate_data(n=320)
    full_dataset = CustomDataset(x, y)
    train_dataset, test_dataset = random_split(full_dataset, [256, 64])
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=64, shuffle=True)
    return train_dataloader, test_dataloader
