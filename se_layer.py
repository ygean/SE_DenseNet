"""SE module"""
from torch import nn


class SELayer(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(SELayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
                nn.Linear(in_channels, in_channels // reduction),
                nn.ReLU(inplace=True),
                nn.Linear(in_channels // reduction, in_channels),
                nn.Sigmoid()
        )


    def forward(self, x):
        batch_size, channel, _, _ = x.size()
        y = self.avg_pool(x).view(batch_size, channel)
        y = self.fc(y).view(batch_size, channel, 1, 1)
        return x * y
