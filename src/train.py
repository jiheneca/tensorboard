"""Entraîne le détecteur, évalue (IoU moyen, % IoU > 0.5) et sauvegarde une figure."""

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader
from src.detector import Detector, ShapeDataset, compute_loss, iou, predict, predict_class
from src.detector import Detector, ShapeDataset, compute_loss, iou, predict


def train(model: Detector, loader: DataLoader, optimizer: torch.optim.Optimizer) -> float:
    """Entraîne le modèle sur une époque et retourne la perte moyenne."""
    model.train()
    total_loss = 0.0
    for images, targets in loader:
        optimizer.zero_grad()  
        outputs = model(images)  
        loss = compute_loss(outputs, targets)  
        loss.backward()  
        optimizer.step()  
        total_loss += loss.item() 
    
    
    return total_loss / len(loader)

# jai modifier ca:
def evaluate(model: Detector, loader: DataLoader, size: int) -> tuple[float, float, float]:
    model.eval()
    ious, correct = [], []
    with torch.no_grad():
        for images, targets in loader:
            logits = model(images)
            grid = targets.shape[-1]
            for logit, target in zip(logits, targets):
                box, _, pred_class = predict_class(logit, size)
                row, col = divmod(int(target[0].argmax()), grid)
                _, dx, dy, w, h = target[:5, row, col].tolist()
                true_class = int(target[5:, row, col].argmax())
                cx, cy = (col + dx) * size / grid, (row + dy) * size / grid
                truth = torch.tensor([cx - w * size / 2, cy - h * size / 2,
                                      cx + w * size / 2, cy + h * size / 2])
                ious.append(float(iou(box, truth[None])[0]))
                correct.append(float(pred_class == true_class))
    t = torch.tensor(ious)
    return t.mean().item(), (t > 0.5).float().mean().item(), torch.tensor(correct).mean().item()