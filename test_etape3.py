import torch
from src.detector import code, compute_loss

target = code(30, 20, 16, 16, 2).unsqueeze(0)    # un triangle, forme (1, 8, 8, 8)

torch.manual_seed(0)
logits = torch.randn(1, 8, 8, 8)
print(compute_loss(logits, target))

bon = logits.clone()
bon[0, 7, 2, 3] = 10.0       # fort score "triangle" dans la case du centre
mauvais = logits.clone()
mauvais[0, 5, 2, 3] = 10.0   # fort score "cercle"
mauvais[0, 7, 2, 3] = -10.0  # faible score "triangle"

print(compute_loss(bon, target))
print(compute_loss(mauvais, target))
print(compute_loss(bon, target) < compute_loss(mauvais, target))