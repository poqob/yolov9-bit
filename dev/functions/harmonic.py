import torch
import torch.nn as nn
import torch.nn.functional as F


def harmonic(x, step=2, maximum_degree=9):
    """Harmonic activation function implementation.
    
    Computes the sum of 1/i * x^i for i in range(1, maximum_degree+1, step)
    """
    result = 0
    for i in range(1, maximum_degree + 1, step):
        result += 1 / i * torch.pow(x, i)
    return result


class Harmonic(nn.Module):
    """Harmonic activation module.
    
    Applies the harmonic function element-wise:
    Harmonic(x) = Σ(1/i * x^i) for i in range(1, maximum_degree+1, step)
    """
    def __init__(self, step=2, maximum_degree=9, inplace=False):
        super(Harmonic, self).__init__()
        self.step = step
        self.maximum_degree = maximum_degree
        self.inplace = inplace  # Standard parameter for activation functions

    def forward(self, x):
        return harmonic(x, self.step, self.maximum_degree)
        
    def extra_repr(self):
        return f'step={self.step}, maximum_degree={self.maximum_degree}, inplace={self.inplace}'

