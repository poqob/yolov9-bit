import torch
import torch.nn as nn
#imlementation from https://github.com/ashis0013/SinLU
class SinLU(nn.Module):
    def __init__(self,k=1.0):
        super(SinLU,self).__init__()
        self.a = nn.Parameter(torch.ones(1)*k)
        self.b = nn.Parameter(torch.ones(1))
    def forward(self,x):
        return torch.sigmoid(x)*(x+self.a*torch.sin(self.b*x))


class SinLUPositive(nn.Module):
    def __init__(self, k=1.0):
        super(SinLUPositive, self).__init__()
        self.a = nn.Parameter(torch.ones(1)*k)
        self.b = nn.Parameter(torch.ones(1))
    
    def forward(self, x):
        # Apply original SinLU activation only to positive values
        # For negative values, return 0 (like ReLU)
        positive_mask = (x > 0).float()
        activation = torch.sigmoid(x) * (x + self.a * torch.sin(self.b * x))
        return activation * positive_mask