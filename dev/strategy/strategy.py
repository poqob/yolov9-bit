from torch import Tensor
import torch.nn as nn
from dev.functions.harmonic import harmonic, harmonic_activation

class Strategy(nn.Module):

    def __init__(self, inplace: bool = False):
        super().__init__()
        self.inplace = inplace
        self.silu = nn.SiLU(inplace=inplace)

    def forward(self, input: Tensor) -> Tensor:
        return self.silu(input)
        # return harmonic(input, step=2, maximum_degree=9)
