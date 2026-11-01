import torch
from src.detector import Detector, code

model = Detector()
images = torch.rand(2, 3, 64, 64)       # 2 fausses images au hasard
out = model(images)
print(out.shape)

target = code(30, 20, 16, 16, 2)
print(target.shape)
print(out.shape[1:] == target.shape)

print(sum(p.numel() for p in model.parameters()))