import torch.nn as nn

class GlobalAvgPool2d(nn.Module):
    def __init__(self, keep_dim=True):
        super(GlobalAvgPool2d, self).__init__()
        self.keep_dim = keep_dim
        
    def forward(self, x):
        if self.keep_dim:
            # [batch_size, channels, height, width] -> [batch_size, channels, 1, 1]
            return nn.functional.adaptive_avg_pool2d(x, (1, 1))
        else:
            # [batch_size, channels, height, width] -> [batch_size, channels]
            return nn.functional.adaptive_avg_pool2d(x, (1, 1)).view(x.size(0), -1)