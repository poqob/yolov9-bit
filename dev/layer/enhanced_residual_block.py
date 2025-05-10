import math
import torch.nn as nn
from dev.layer.conv import Conv

class ResBlockEnhanced(nn.Module):
    """Enhanced ResNet style residual block with bottleneck design"""
    def __init__(self, c1, c2=None, shortcut=True, g=1, e=0.5):
        super(ResBlockEnhanced, self).__init__()
        # If only one parameter is provided, use it for both input and output channels
        c2 = c2 if c2 is not None else c1
        c_ = int(c2 * e)  # bottleneck channels
        
        # Use the Conv class properly, not the activation function
        self.cv1 = Conv(c1, c_, 1, 1)  # point-wise
        self.cv2 = Conv(c_, c_, 3, 1, g=g)  # depth-wise conv
        self.cv3 = Conv(c_, c2, 1, 1)  # point-wise, no activation
        self.shortcut = shortcut and c1 == c2
    
    def forward(self, x):
        return x + self.cv3(self.cv2(self.cv1(x))) if self.shortcut else self.cv3(self.cv2(self.cv1(x)))
