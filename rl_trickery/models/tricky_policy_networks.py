import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from a2c_ppo_acktr.distributions import Bernoulli, Categorical, DiagGaussian
from a2c_ppo_acktr.utils import init

from rl_trickery.models.conv_lstm import ConvLSTMCell
import collections
import gym


class Discrete2OneHot(nn.Module):
    def __init__(self, n):
        super(Discrete2OneHot, self).__init__()
        self.n = n

    def forward(self, x):
        oh = torch.zeros((x.size(0), self.n), device=x.device)
        oh[range(x.size(0)), x.long()] = 1
        return oh


class Flatten(nn.Module):
    def forward(self, x):
        return x.view(x.size(0), -1)


class Unflatten(nn.Module):
    def __init__(self, target_shape):
        super(Unflatten, self).__init__()
        self.out_shape = target_shape

    def forward(self, x):
        return x.view(x.size(0), *self.out_shape)


class NoTransition(nn.Module):
    def __init__(self, hidden_size):
        super(NoTransition, self).__init__()
        self.recurrent_hidden_state_size = 0
        self.in_shape = (hidden_size,)
        self.out_shape = (hidden_size,)

    def forward(self, x, rnn_state, done, action_cog=None):
        return x, rnn_state


class RNNTransition(nn.Module):
    def __init__(self, hidden_size, action_cog=False):
        super(RNNTransition, self).__init__()
        self.in_shape = (hidden_size,)
        self.out_shape = (hidden_size,)
        # self.in_shape += int(action_cog)
        self.recurrent_hidden_state_size = (hidden_size,)
        self.gru = nn.GRUCell(hidden_size, hidden_size, bias=True)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

    def forward(self, x, hxs, done, action_cog=None):
        hxs = hxs * (1 - done.view(-1, 1))
        hxs = self.gru(x, hxs)
        x = hxs

        return x, hxs


class CRNNTransition(nn.Module):
    def __init__(self, state_channels, action_cog=False):
        super(CRNNTransition, self).__init__()
        self.in_shape = (state_channels, 7, 7)
        self.out_shape = (state_channels, 7, 7)
        self.state_channels = state_channels
        self.recurrent_hidden_state_size = (2*state_channels, 7, 7)
        self.conv_lstm = ConvLSTMCell(
            input_dim=state_channels, hidden_dim=state_channels, kernel_size=(3,3), bias=True,
        )

    def forward(self, x, hxs, done):
        expansion_dims = ((hxs.dim() - 1) * (1,))
        hxs = hxs * (1 - done.view(-1, *expansion_dims))
        hxs = hxs.split(self.state_channels, dim=1)


        h, c = self.conv_lstm(x, hxs)

        hxs = torch.cat((h, c), dim=1)
        return h, hxs


class DimensionalityAdjuster(nn.Module):
    def __init__(self, in_shape, out_shape):
        super(DimensionalityAdjuster, self).__init__()
        init_relu = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0),
                                   nn.init.calculate_gain('relu'))
        if len(in_shape) > len(out_shape):
            # flatten
            self.net = nn.Sequential(
                Flatten(),
                init_relu(nn.Linear(int(np.prod(in_shape)), out_shape[0])),
                nn.ReLU()
            )
        elif len(in_shape) < len(out_shape):
            # unflatten
            self.net = nn.Sequential(
                init_relu(nn.Linear(in_shape[0], int(np.prod(out_shape)))),
                Unflatten(out_shape),
                nn.ReLU()
            )
            pass
        else:
            if all(in_shape == out_shape):
                self.net = nn.Sequential(
                    nn.Identity()
                )
            elif len(in_shape) == 1:
                raise NotImplementedError
                # self.net = nn.Sequential(
                #     init_relu(nn.Linear(in_shape[0], out_shape[0])),
                #     nn.ReLU()
                # )
            else:
                raise NotImplementedError

    def forward(self, x):
        x = self.net(x)
        return x


class Encoder(nn.Module):
    def __init__(self, obs_space, hidden_size=512, state_channels=32):
        super(Encoder, self).__init__()
        if obs_space.__class__.__name__ == "Discrete":
            self.out_shape = np.atleast_1d(hidden_size)
            self.in_shape = (obs_space.n,)
            init_linear = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                   constant_(x, 0), np.sqrt(2))
            self.net = nn.Sequential(
                Discrete2OneHot(obs_space.n),
                init_linear(nn.Linear(self.in_shape[0], hidden_size)), nn.Tanh(),
                init_linear(nn.Linear(hidden_size, hidden_size)), nn.Tanh())

        elif len(obs_space.shape) == 1:
            self.in_shape = obs_space.shape
            self.out_shape = np.atleast_1d(hidden_size)
            init_linear = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.
                                   constant_(x, 0), np.sqrt(2))
            self.net = nn.Sequential(
                init_linear(nn.Linear(obs_space.shape[0], hidden_size)), nn.Tanh(),
                init_linear(nn.Linear(hidden_size, hidden_size)), nn.Tanh())
        elif len(obs_space.shape) == 3:
            self.in_shape = obs_space.shape
            n_channels = obs_space.shape[0]
            im_size = obs_space.shape[1]
            self.out_shape = np.array([state_channels, 7, 7])
            init_relu = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0),
                                       nn.init.calculate_gain('relu'))
            if im_size == 84:
                self.net = nn.Sequential(
                    init_relu(nn.Conv2d(n_channels, 32, 8, stride=4)), nn.ReLU(),
                    init_relu(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
                    init_relu(nn.Conv2d(64, state_channels, 3, stride=1)), nn.ReLU(),
                )
            elif im_size == 64:
                self.net = nn.Sequential(
                    # input (x, 64, 64)
                    init_relu(nn.Conv2d(n_channels, 32, 6, stride=4, padding=1)), nn.ReLU(),
                    # input (3, 16, 16)
                    init_relu(nn.Conv2d(32, 64, 4, stride=2, padding=2)), nn.ReLU(),
                    # input (3, 9, 9)
                    init_relu(nn.Conv2d(64, state_channels, 3, stride=1)), nn.ReLU(),
                    # input (3, 7, 7)
                )
            else:
                raise NotImplementedError

    def forward(self, x):
        if len(self.in_shape) == 3:
            y = self.net(x.float() / 255.0)
        else:
            y = self.net(x.float())
        return y


class ActorCritic(nn.Module):
    def __init__(self, action_space, hidden_size):
        super(ActorCritic, self).__init__()
        self.in_shape = np.atleast_1d(hidden_size)
        self.out_shape = None
        init_ = lambda m: init(m, nn.init.orthogonal_, lambda x: nn.init.constant_(x, 0))
        self.critic_linear = init_(nn.Linear(hidden_size, 1))

        if action_space.__class__.__name__ == "Discrete":
            num_outputs = action_space.n
            self.actor_linear = Categorical(hidden_size, num_outputs)
        elif action_space.__class__.__name__ == "Box":
            num_outputs = action_space.shape[0]
            self.actor_linear = DiagGaussian(hidden_size, num_outputs)
        elif action_space.__class__.__name__ == "MultiBinary":
            num_outputs = action_space.shape[0]
            self.actor_linear = Bernoulli(hidden_size, num_outputs)
        else:
            raise NotImplementedError

    def forward_actor(self, x):
        x = self.actor_linear(x)
        return x

    def forward_critic(self, x):
        x = self.critic_linear(x)
        return x


PolicyOutput = collections.namedtuple('PolicyOutput', 'value, action, action_log_probs, dist_entropy')


class RecursivePolicy(nn.Module):
    def __init__(
            self,
            obs_space, action_space,
            architecture,
            state_channels, hidden_size,
            action_cog=False,
            random_cog_fraction=0.0,
            **kwargs):
        super(RecursivePolicy, self).__init__()

        assert 0.0 <= random_cog_fraction <= 1.0
        self.twoAM = action_cog
        self.random_cog_fraction = random_cog_fraction
        self.hidden_size = hidden_size
        self.architecture = architecture
        self.is_recurrent = architecture in ["rnn", "crnn"]

        self.encoder = Encoder(obs_space, hidden_size, state_channels)

        if architecture == "ff":
            self.transition = NoTransition(hidden_size)
        elif architecture == "rnn":
            self.transition = RNNTransition(
                hidden_size=hidden_size,
                action_cog=False,
            )
        elif architecture == "crnn":
            self.transition = CRNNTransition(
                state_channels=state_channels,
                action_cog=False,
            )
        else:
            raise NotImplementedError

        self.ac_env = ActorCritic(
            action_space=action_space,
            hidden_size=hidden_size
        )

        self.enc2trans = DimensionalityAdjuster(
            in_shape=self.encoder.out_shape,
            out_shape=self.transition.in_shape
        )

        self.trans2ac = DimensionalityAdjuster(
            in_shape=self.transition.out_shape,
            out_shape=self.ac_env.in_shape
        )

        if self.twoAM:
            self.ac_cog = ActorCritic(
                action_space=gym.spaces.Discrete(2),
                hidden_size=hidden_size
            )
            self.trans2ac_cog = DimensionalityAdjuster(
                in_shape=self.transition.out_shape,
                out_shape=self.ac_cog.in_shape
            )

        self.train()

    def recurrent_hidden_state_size(self):
        if self.is_recurrent:
            return self.transition.recurrent_hidden_state_size
        else:
            return (0,)

    def forward(self, obs, rnn_h, done):
        x = self.encoder(obs.float())
        x = self.enc2trans(x)
        x, rnn_h = self.transition(x, rnn_h, done)
        x = self.trans2ac(x)
        value = self.ac_env.forward_critic(x)
        dist = self.ac_env.forward_actor(x)
        action = dist.sample()

        action_log_probs = dist.log_probs(action)
        dist_entropy = dist.entropy().unsqueeze(-1)

        env_policy = PolicyOutput(
            value=value,
            action=action,
            action_log_probs=action_log_probs,
            dist_entropy=dist_entropy
        )

        if not self.twoAM:
            if self.random_cog_fraction:
                action_cog = torch.rand((action.size(0), 1), device=action.device) > self.random_cog_fraction
            else:
                action_cog = torch.ones((action.size(0), 1), device=action.device)
            action_cog = action_cog.long()
            cog_policy = PolicyOutput(
                value=None,
                action=action_cog,
                action_log_probs=None,
                dist_entropy=None
            )
        else:
            value_cog = self.ac_cog.forward_critic(x)
            dist_cog = self.ac_cog.forward_actor(x)
            action_cog = dist.sample()

            action_cog_log_probs = dist_cog.log_probs(action)
            dist_cog_entropy = dist_cog.entropy().unsqueeze(-1)

            cog_policy = PolicyOutput(
                value=value_cog,
                action=action_cog,
                action_log_probs=action_cog_log_probs,
                dist_entropy=dist_cog_entropy
            )

        return env_policy, cog_policy, rnn_h

    def get_value(self, obs, rnn_h, done):
        x = self.encoder(obs.float())
        x = self.enc2trans(x)
        x, rnn_h = self.transition(x, rnn_h, done)
        x = self.trans2ac(x)
        value = self.ac_env.forward_critic(x)
        return value
