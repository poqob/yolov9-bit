import math
import torch.nn as nn
from dev.dev import Dev
from dev.layer.conv import Conv

class ResBlock(nn.Module):
    """ResNet style residual block with BN-ReLU-Conv pattern"""
    def __init__(self, c1, c2=None, shortcut=True, g=1, e=0.5):
        super(ResBlock, self).__init__()
        # If only one parameter is provided, use it for both input and output channels
        c2 = c2 if c2 is not None else c1
        c_ = int(c2 * e)  # hidden channels
        
        # Fix: Get the Conv class that already handles activation properly instead of 
        # trying to get the activation directly
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
