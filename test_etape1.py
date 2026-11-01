from src.detector import code

t = code(30, 20, 16, 16, 2)       # un triangle (class_id = 2)
print(t.shape)
print(t[:, 2, 3])
print(t.sum())