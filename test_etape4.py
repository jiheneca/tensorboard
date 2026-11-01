import torch
from src.detector import code, predict_class

target = code(30, 20, 16, 16, 2)                  # un triangle

# On fabrique de faux logits "parfaits" à partir de la cible
logits = torch.full((8, 8, 8), -10.0)             # partout : "rien ici"
logits[0, 2, 3] = 10.0                            # pobj très fort dans la case du centre
logits[1:5, 2, 3] = torch.logit(target[1:5, 2, 3])
logits[5:, 2, 3] = torch.tensor([0.0, 0.0, 5.0])  # le réseau pense "triangle"

box, score, classe = predict_class(logits)
print(box)
print(round(score, 3))
print(classe)