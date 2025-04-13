import torch
import torch.nn as nn
import torch.nn.functional as F

def kcoth(x):
    return 1 / torch.tanh(x)

def tanhkcothx(x):
    return torch.tanh(kcoth(x) * x)



class TanhCothx(nn.Module):
    def __init__(self):
        super(TanhCothx, self).__init__()
        self.alpha = nn.Parameter(torch.tensor(1.0))

    def forward(self, x):
        return tanhkcothx(x)