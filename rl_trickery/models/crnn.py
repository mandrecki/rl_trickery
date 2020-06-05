import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):
    def __init__(self,
                 input_size,
                 input_dim,
                 hidden_dim,
                 kernel_size,
                 stride,
                 bias):
        super().__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.bias = bias

        padding = kernel_size // 2

        self.Wxi = nn.Conv2d(input_dim, hidden_dim, kernel_size, stride, padding, bias)
        self.Wxf = nn.Conv2d(input_dim, hidden_dim, kernel_size, stride, padding, bias)
        self.Wxo = nn.Conv2d(input_dim, hidden_dim, kernel_size, stride, padding, bias)
        self.Wxg = nn.Conv2d(input_dim, hidden_dim, kernel_size, stride, padding, bias)

        self.Whi = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding, bias)
        self.Whf = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding, bias)
        self.Who = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding, bias)
        self.Whg = nn.Conv2d(hidden_dim, hidden_dim, kernel_size, stride, padding, bias)

    def forward(self, x, h, c):
        # x = [batch, input dim, height, width]
        # h = [batch, hidden dim, height, width]
        # c = [batch, hidden dim, height, width]

        batch_size = x.shape[0]

        assert x.shape == (batch_size, self.input_dim, self.height, self.width)
        assert h.shape == (batch_size, self.hidden_dim, self.height, self.width)
        assert c.shape == (batch_size, self.hidden_dim, self.height, self.width)

        i = torch.sigmoid(self.Wxi(x) + self.Whi(h))
        f = torch.sigmoid(self.Wxf(x) + self.Whf(h))
        o = torch.sigmoid(self.Wxo(x) + self.Who(h))
        g = torch.tanh(self.Wxg(x) + self.Whg(h))

        # i/f/o/g = [batch, hidden dim, height, width]

        assert i.shape == (batch_size, self.hidden_dim, self.height, self.width)
        assert f.shape == (batch_size, self.hidden_dim, self.height, self.width)
        assert o.shape == (batch_size, self.hidden_dim, self.height, self.width)
        assert g.shape == (batch_size, self.hidden_dim, self.height, self.width)

        c = f * c + i * g
        h = o * torch.tanh(c)

        # h = [batch, hidden dim, height, width]
        # c = [batch, hidden dim, height, width]

        assert h.shape == (batch_size, self.hidden_dim, self.height, self.width)
        assert c.shape == (batch_size, self.hidden_dim, self.height, self.width)

        return h, c


class ConvLSTM(nn.Module):
    def __init__(self,
                 input_size,
                 input_dim,
                 hidden_dim,
                 kernel_size,
                 stride,
                 n_layers,
                 bias):
        super().__init__()

        self.height, self.width = input_size
        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.stride = stride
        self.n_layers = n_layers
        self.bias = bias

        convs = []

        for i in range(n_layers):
            input_dim = input_dim if i == 0 else hidden_dim

            convs.append(ConvLSTMCell(input_size=(self.height, self.width),
                                      input_dim=input_dim,
                                      hidden_dim=hidden_dim,
                                      kernel_size=kernel_size,
                                      stride=stride,
                                      bias=bias))

        self.convs = nn.ModuleList(convs)

    def forward(self, x):

        # x = [batch, length, input dim, height, width]

        batch_size = x.shape[0]
        seq_length = x.shape[1]

        assert x.shape == (batch_size, seq_length, self.input_dim, self.height, self.width)

        for _, conv in enumerate(self.convs):

            h = torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(x.device)
            c = torch.zeros(batch_size, self.hidden_dim, self.height, self.width).to(x.device)
            H = torch.zeros(batch_size, seq_length, self.hidden_dim, self.height, self.width).to(x.device)

            for t in range(seq_length):
                h, c = conv(x[:, t, :, :, :], h, c)

                # h = [batch, hidden dim, height, width]
                # c = [batch, hidden dim, height, width]

                assert h.shape == (batch_size, self.hidden_dim, self.height, self.width)
                assert c.shape == (batch_size, self.hidden_dim, self.height, self.width)

                H[:, t, :, :, :] = h

                # H = [batch, seq length, hidden dim, height, width]

            x = H

        assert H.shape == (batch_size, seq_length, self.hidden_dim, self.height, self.width)
        assert h.shape == (batch_size, self.hidden_dim, self.height, self.width)

        return H, h


class TinyEncoder(nn.Module):
    """
    Used for the "tiny" 3x10x10 image input
    """

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3)
        self.conv2 = nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)
        x = self.conv2(x)

        return x


class TinyDRC(nn.Module):
    def __init__(self,
                 image_encoder,
                 encoded_image_dim,
                 hidden_dim,
                 kernel_size,
                 stride,
                 n_layers,
                 n_ticks,
                 bias,
                 policy_layer_dim,
                 output_dim
                 ):
        super().__init__()

        self.input_dim = (10, 10, 3)
        self.n_ticks = n_ticks
        self.output_dim = output_dim

        self.image_encoder = image_encoder

        encoded_image_size = (encoded_image_dim[1], encoded_image_dim[2])
        encoded_image_channels = encoded_image_dim[0]

        policy_layer_input_dim = hidden_dim * encoded_image_dim[1] * encoded_image_dim[2]

        self.convlstm = ConvLSTM(encoded_image_size, encoded_image_channels, hidden_dim, kernel_size, stride, n_layers,
                                 bias)
        self.policy_layer = nn.Linear(policy_layer_input_dim, policy_layer_dim)
        self.action_head = nn.Linear(policy_layer_dim, output_dim)
        self.value_head = nn.Linear(policy_layer_dim, 1)

    def forward(self, x):
        # x = [batch, height, width, channel] = [n_envs, 10, 10, 3]

        batch_size = x.shape[0]

        x = x.permute(0, 3, 1, 2)

        # x = [batch, channel, height, width] = [n_envs, 3, 10, 10]

        x = self.image_encoder(x)

        # x = [batch, channel, height, width] = [n_envs, 32, 6, 6]

        x = x.unsqueeze(1).repeat(1, self.n_ticks, 1, 1, 1)

        # x = [batch, n_ticks, channel, height, width] = [n_envs, n_ticks, 32, 6, 6]

        _, x = self.convlstm(x)

        # x = [batch, hidden_dim, height, width]

        x = x.view(batch_size, -1)

        # x = [batch, hidden_dim * height * width] = [n_envs, 32 * 6 * 6]

        x = self.policy_layer(x)

        # x = [batch, policy_layer_dim]

        return self.action_head(x), self.value_head(x)