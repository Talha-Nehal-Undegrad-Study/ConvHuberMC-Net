import numpy as np
import torch

c = 1.345

def get_nonzeros(matrix):
    
    return np.array(np.nonzero(matrix))

def rho(x):
    x = torch.as_tensor(x)
    subset = torch.abs(x) <= c
    return 0.5 * x**2 * subset + (1 - subset) * (c * torch.abs(x) - 0.5 * c**2)

def psi(x):
    x = torch.as_tensor(x)
    subset = torch.abs(x) <= c
    return subset * x + (1 - subset) * c * torch.sign(x)

def alpha():
    return 0.7102
