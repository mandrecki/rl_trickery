import torch


def conv3x3(in_channels, out_channels, stride=1):
    return torch.nn.Conv2d(
        in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False
    )


# Residual block
class ResidualBlock(torch.nn.Module):
    def __init__(self, num_channels, stride=1):
        super().__init__()
        self.conv1 = conv3x3(num_channels, num_channels, stride)
        self.bn1 = torch.nn.BatchNorm2d(num_channels)
        self.relu = torch.nn.ReLU()
        self.conv2 = conv3x3(num_channels, num_channels)
        self.bn2 = torch.nn.BatchNorm2d(num_channels)

    def forward(self, x):
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        out += x
        out = self.relu(out)
        return out


class FullyConnectedNetwork(torch.nn.Module):
    def __init__(self, input_size, layer_sizes, output_size, activation=None):
        super().__init__()
        size_list = [input_size] + layer_sizes
        layers = []
        if 1 < len(size_list):
            for i in range(len(size_list) - 1):
                layers.extend(
                    [
                        torch.nn.Linear(size_list[i], size_list[i + 1]),
                        torch.nn.LeakyReLU(),
                    ]
                )
        layers.append(torch.nn.Linear(size_list[-1], output_size))
        if activation:
            layers.append(activation)
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x):
        for layer in self.layers:
            x = layer(x)
        return x


class DynamicsNetwork(torch.nn.Module):
    def __init__(
        self,
        obs_space,
        num_blocks=16,
        num_channels=33,
        reduced_channels=32,
    ):
        super().__init__()
        observation_shape = obs_space.shape
        self.observation_shape = observation_shape
        self.conv = conv3x3(num_channels -1, num_channels - 1)
        self.bn = torch.nn.BatchNorm2d(num_channels - 1)
        self.relu = torch.nn.ReLU()
        self.resblocks = torch.nn.ModuleList(
            [ResidualBlock(num_channels - 1) for _ in range(num_blocks)]
        )

        self.conv1x1 = torch.nn.Conv2d(num_channels - 1, reduced_channels, 1)

    def forward(self, x):
        out = self.conv(x)
        out = self.bn(out)
        out = self.relu(out)
        for block in self.resblocks:
            out = block(out)
        state = out
        return state