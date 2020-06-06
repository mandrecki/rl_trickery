import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init

from rl_trickery.models.conv_lstm import ConvLSTMCell

class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Policy(nn.Module):
    def __init__(self, obs_shape, action_space, base=None, base_kwargs=None):
        super(Policy, self).__init__()
        if base_kwargs is None:
            base_kwargs = {}
        if base is None:
            if len(obs_shape) == 3:
                if obs_shape[-1] == 84:
                    base = CNNBase
                elif obs_shape[-1] == 64:
                    base = CNNBase64
                else:
                    raise NotImplementedError
            elif len(obs_shape) == 1:
                base = MLPBase
            else:
                raise NotImplementedError

        self.base = base(obs_shape[0], **base_kwargs)

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.base.output_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.base.output_size, num_outputs)
        else:
            raise NotImplementedError

    @property
    def is_recurrent(self):
        return self.base.is_recurrent

    @property
    def recurrent_hidden_state_size(self):
        """Size of rnn_hx."""
        return self.base.recurrent_hidden_state_size

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        value, _, _ = self.base(inputs, rnn_hxs, masks)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        value, actor_features, rnn_hxs = self.base(inputs, rnn_hxs, masks)
        dist = self.dist(actor_features)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs


class NNBase(nn.Module):
    def __init__(self, recurrent, recurrent_input_size, hidden_size):
        super(NNBase, self).__init__()

        self._hidden_size = hidden_size
        self._recurrent = recurrent

        if recurrent:
            self.gru = nn.GRU(recurrent_input_size, hidden_size)
            for name, param in self.gru.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

    @property
    def is_recurrent(self):
        return self._recurrent

    @property
    def recurrent_hidden_state_size(self):
        if self._recurrent:
            return self._hidden_size
        return 1

    @property
    def output_size(self):
        return self._hidden_size

    def _forward_gru(self, x, hxs, masks):
        if x.size(0) == hxs.size(0):
            x, hxs = self.gru(x.unsqueeze(0), (hxs * masks).unsqueeze(0))
            x = x.squeeze(0)
            hxs = hxs.squeeze(0)
        else:
            # x is a (T, N, -1) tensor that has been flatten to (T * N, -1)
            N = hxs.size(0)
            T = int(x.size(0) / N)

            # unflatten
            x = x.view(T, N, x.size(1))

            # Same deal with masks
            masks = masks.view(T, N)

            # Let's figure out which steps in the sequence have a zero for any agent
            # We will always assume t=0 has a zero in it as that makes the logic cleaner
            has_zeros = ((masks[1:] == 0.0) \
                            .any(dim=-1)
                            .nonzero()
                            .squeeze()
                            .cpu())

            # +1 to correct the masks[1:]
            if has_zeros.dim() == 0:
                # Deal with scalar
                has_zeros = [has_zeros.item() + 1]
            else:
                has_zeros = (has_zeros + 1).numpy().tolist()

            # add t=0 and t=T to the list
            has_zeros = [0] + has_zeros + [T]

            hxs = hxs.unsqueeze(0)
            outputs = []
            for i in range(len(has_zeros) - 1):
                # We can now process steps that don't have any zeros in masks together!
                # This is much faster
                start_idx = has_zeros[i]
                end_idx = has_zeros[i + 1]

                rnn_scores, hxs = self.gru(
                    x[start_idx:end_idx],
                    hxs * masks[start_idx].view(1, -1, 1))

                outputs.append(rnn_scores)

            # assert len(outputs) == T
            # x is a (T, N, -1) tensor
            x = torch.cat(outputs, dim=0)
            # flatten
            x = x.view(T * N, -1)
            hxs = hxs.squeeze(0)

        return x, hxs


class CNNBase(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        self.main = nn.Sequential(
            init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
            init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
            init_(nn.Conv2d(64, 32, 3, stride=1)), nn.ReLU(), Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)), nn.ReLU())

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class CNNBase64(NNBase):
    def __init__(self, num_inputs, recurrent=False, hidden_size=512):
        super(CNNBase64, self).__init__(recurrent, hidden_size, hidden_size)

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0), nn.init.calculate_gain('relu'))

        # CNN for 64x64
        self.main = nn.Sequential(
            # input (3, 64, 64)
            init_(nn.Conv2d(num_inputs, 32, 6, stride=4, padding=1)),
            nn.ReLU(),
            # input (3, 16, 16)
            init_(nn.Conv2d(32, 64, 4, stride=2, padding=2)),
            nn.ReLU(),
            # input (3, 9, 9)
            init_(nn.Conv2d(64, 32, 3, stride=1)),
            nn.ReLU(),
            Flatten(),
            init_(nn.Linear(32 * 7 * 7, hidden_size)),
            nn.ReLU()
        )

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                               constant_(x, 0))

        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        self.train()

    def forward(self, inputs, rnn_hxs, masks):
        x = self.main(inputs / 255.0)

        if self.is_recurrent:
            x, rnn_hxs = self._forward_gru(x, rnn_hxs, masks)

        return self.critic_linear(x), x, rnn_hxs


class NoTransition(nn.Module):
    def __init__(self, hidden_size, state_channels):
        super(NoTransition, self).__init__()
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))
        self.recurrent_hidden_state_size = 1
        self.flat_lin = nn.Sequential(
                Flatten(),
                init_(nn.Linear(state_channels * 7 * 7, hidden_size)),
                nn.ReLU()
        )

    def forward(self, x, rnn_state, masks):
        x = self.flat_lin(x)
        return x, rnn_state


class RNNTransition(NoTransition):
    def __init__(self, hidden_size, state_channels, recurse_depth):
        super(RNNTransition, self).__init__(hidden_size, state_channels)
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0),
                               nn.init.calculate_gain('relu'))
        self.recurse_depth = recurse_depth
        self.recurrent_hidden_state_size = hidden_size
        # self.gru = nn.GRU(hidden_size, hidden_size)
        self.gru = nn.GRUCell(hidden_size, hidden_size, bias=True)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

    def forward_step(self, x, hxs, masks):
        x = self.flat_lin(x)

        hxs = hxs * masks.view(-1, 1)
        for i in range(self.recurse_depth):
            hxs = self.gru(x, hxs)

        return hxs, hxs

    def forward_multistep(self, x, hxs, masks):
        # num steps as first dimension
        outs = []
        for i in range(x.size(0)):
            out, hxs = self.forward_step(x[i], hxs, masks[i])
            outs.append(out)

        outs = torch.cat(outs, dim=0)
        return outs,  hxs

    def forward(self, x, hxs, masks):
        b = hxs.size(0)
        t = int(x.size(0) / b)
        masks = masks.view(t, b)
        x_dims = x.size()[1:]

        if t == 1:
            x, hxs = self.forward_step(x, hxs, masks)
        else:
            x = x.view(t, b, *x_dims)
            x, hxs = self.forward_multistep(x, hxs, masks)

        return x, hxs


class CRNNTransition(NoTransition):
    def __init__(self, hidden_size, state_channels, recurse_depth):
        super(CRNNTransition, self).__init__(hidden_size, state_channels)
        self.state_channels = state_channels
        self.recurse_depth = recurse_depth
        self.recurrent_hidden_state_size = (2*state_channels, 7, 7)
        self.conv_lstm = ConvLSTMCell(input_dim=state_channels, hidden_dim=state_channels, kernel_size=(3,3), bias=True)

    def forward_step(self, x, hxs, masks):
        expansion_dims = ((hxs.dim() - 1) * (1,))
        hxs = hxs * masks.view(-1, *expansion_dims)
        hxs = hxs.split(self.state_channels, dim=1)

        for i in range(self.recurse_depth):
            h, c = self.conv_lstm(x, hxs)
            hxs = (h, c)

        x = self.flat_lin(c)
        hxs = torch.cat(hxs, dim=1)
        return x, hxs

    def forward_multistep(self, x, hxs, masks):
        # num steps as first dimension
        outs = []
        for i in range(x.size(0)):
            out, hxs = self.forward_step(x[i], hxs, masks[i])
            outs.append(out)

        outs = torch.cat(outs, dim=0)
        return outs,  hxs

    def forward(self, x, hxs, masks):
        b = hxs.size(0)
        t = int(x.size(0) / b)
        masks = masks.view(t, b)
        x_dims = x.size()[1:]

        if t == 1:
            x, hxs = self.forward_step(x, hxs, masks)
        else:
            x = x.view(t, b, *x_dims)
            x, hxs = self.forward_multistep(x, hxs, masks)

        return x, hxs


class RecurrentPolicy(nn.Module):
    def __init__(self, obs_shape, action_space, architecture, state_channels, hidden_size, recurse_depth=1):
        super(RecurrentPolicy, self).__init__()

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0), nn.init.calculate_gain('relu'))

        num_inputs = obs_shape[0]
        im_size = obs_shape[1]
        assert im_size == obs_shape[2]
        self.spatial_latent_size = (7, 7)
        self.hidden_size = hidden_size
        self.architecture = architecture
        self.is_recurrent = architecture in ["rnn", "crnn"]

        if im_size == 84:
            self.encoder = nn.Sequential(
                init_(nn.Conv2d(num_inputs, 32, 8, stride=4)), nn.ReLU(),
                init_(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
                init_(nn.Conv2d(64, state_channels, 3, stride=1)), nn.ReLU(),
            )
        elif im_size == 64:
            self.encoder = nn.Sequential(
                # input (x, 64, 64)
                init_(nn.Conv2d(num_inputs, 32, 6, stride=4, padding=1)), nn.ReLU(),
                # input (3, 16, 16)
                init_(nn.Conv2d(32, 64, 4, stride=2, padding=2)), nn.ReLU(),
                # input (3, 9, 9)
                init_(nn.Conv2d(64, state_channels, 3, stride=1)), nn.ReLU(),
                # input (3, 7, 7)
            )
        else:
            raise NotImplementedError

        if architecture == "ff":
            self.transition = NoTransition(hidden_size, state_channels=state_channels)
        elif architecture == "rnn":
            self.transition = RNNTransition(hidden_size, state_channels=state_channels, recurse_depth=recurse_depth)
        elif architecture == "crnn":
            self.transition = CRNNTransition(hidden_size, state_channels=state_channels, recurse_depth=recurse_depth)
        else:
            raise NotImplementedError

        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.critic_linear = init_(nn.Linear(self.hidden_size, 1))

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.dist = Categorical(self.hidden_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.dist = DiagGaussian(self.hidden_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.dist = Bernoulli(self.hidden_size, num_outputs)
        else:
            raise NotImplementedError

        self.train()

    @property
    def recurrent_hidden_state_size(self):
        if self.is_recurrent:
            return self.transition.recurrent_hidden_state_size
        return 1

    def forward(self, inputs, rnn_hxs, masks):
        raise NotImplementedError

    def act(self, inputs, rnn_hxs, masks, deterministic=False):
        x = self.encoder(inputs / 255.0)
        x, rnn_hxs = self.transition(x, rnn_hxs, masks)
        value = self.critic_linear(x)
        dist = self.dist(x)

        if deterministic:
            action = dist.mode()
        else:
            action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action, action_log_probs, rnn_hxs

    def get_value(self, inputs, rnn_hxs, masks):
        x = self.encoder(inputs / 255.0)
        x, rnn_hxs = self.transition(x, rnn_hxs, masks)
        value = self.critic_linear(x)
        return value

    def evaluate_actions(self, inputs, rnn_hxs, masks, action):
        x = self.encoder(inputs / 255.0)
        x, rnn_hxs = self.transition(x, rnn_hxs, masks)
        value = self.critic_linear(x)
        dist = self.dist(x)

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().mean()

        return value, action_log_probs, dist_entropy, rnn_hxs
