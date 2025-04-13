from torch import Tensor
import torch.nn as nn
from dev.functions.harmonic import harmonic

class Strategy(nn.Module):

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.silu = nn.SiLU(inplace=inplace)
        self.relu = nn.ReLU(inplace=inplace)
        self.leaky_relu = nn.LeakyReLU(inplace=inplace)
        self.mish = nn.Mish(inplace=inplace)
        self.gelu = nn.GELU()
        self.h_sigmoid = nn.Hardsigmoid(inplace=inplace)
        self.h_swish = nn.Hardswish(inplace=inplace)

    def forward(self, input: Tensor) -> Tensor:
        # return self.relu(input)
        # return self.silu(input)
        # return harmonic(input, step=2, maximum_degree=9) # no enough vram
        # return self.leaky_relu(input)
        # return self.mish(input)
        # return self.h_sigmoid(input)
        return self.h_swish(input)

        pass
    
