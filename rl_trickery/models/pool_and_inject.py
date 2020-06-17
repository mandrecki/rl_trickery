from torch import nn
import torch


class PoolAndInject(nn.Module):
    def __init__(self, spatial_shape, state_channels):
        super(PoolAndInject, self).__init__()
        self.spatial_shape = spatial_shape

        self.max_pool = nn.MaxPool2d(kernel_size=spatial_shape)
        self.avg_pool = nn.AvgPool2d(kernel_size=spatial_shape)
        self.conv = nn.Conv2d(
            in_channels=3*state_channels,
            out_channels=state_channels,
            kernel_size=(3, 3),
            padding=1,
        )

    def forward(self, x):
        max = self.max_pool(x)
        max = max.expand((x.size(0), -1) + self.spatial_shape)
        avg = self.avg_pool(x)
        avg = avg.expand((x.size(0), -1) + self.spatial_shape)
        x = torch.cat([x, max, avg], 1)
        x = self.conv(x)
        return x