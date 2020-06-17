import torch
from torch import nn


class CoordConv(nn.Module):
    """
    Appends coordinates to an image tensor (4D). Creates tensor once for a predefined shape. Adjusts batch size so that
    it minimal changes are required at run time.
    """
    def __init__(self, shape, append_radius=True):
        super(CoordConv, self).__init__()
        x = torch.arange(0, 1, 1/shape[0])
        x = x.view(1, 1, 1, -1)
        x = x.expand(-1, -1, shape[1], shape[0])
        y = torch.arange(0, 1, 1/shape[1])
        y = y.view(1, 1, -1, 1)
        y = y.expand(-1, -1, shape[1], shape[0])
        r = ((x - 0.5) ** 2 + (y - 0.5) ** 2).sqrt()

        self.coords = torch.cat((x,y), dim=1)
        if append_radius:
            self.coords = torch.cat((self.coords, r), dim=1)

    def forward(self, x):
        batch_size = x.size(0)
        if batch_size > self.coords.size(0):
            self.coords = self.coords[:1].expand(batch_size, -1, -1, -1)

        x = torch.cat((x, self.coords[:batch_size, ...].detach()), dim=1)
        return x
