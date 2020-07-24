import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import collections
import gym

from rl_trickery.models.distributions import Bernoulli, Categorical, DiagGaussian, init
from rl_trickery.models.conv_lstm import ConvLSTMCell
from rl_trickery.models.coord_conv import CoordConv
from rl_trickery.models.pool_and_inject import PoolAndInject


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

    def forward(self, x, rnn_state, done, action_cog):
        return x, rnn_state


class RNNTransition(nn.Module):
    def __init__(self, hidden_size, recurse_depth=1, append_a_cog=False, skip_connection=False, two_transitions=False):
        super(RNNTransition, self).__init__()
        gru_in_size = hidden_size+int(append_a_cog)
        self.two_transitions = two_transitions
        self.append_a_cog = append_a_cog
        self.recurse_depth = recurse_depth
        self.skip_connection = skip_connection
        self.in_shape = (hidden_size,)
        if self.skip_connection:
            self.out_shape = (hidden_size+gru_in_size,)
        else:
            self.out_shape = (hidden_size,)
        self.recurrent_hidden_state_size = (hidden_size,)
        self.gru = nn.GRUCell(gru_in_size, hidden_size, bias=True)
        for name, param in self.gru.named_parameters():
            if 'bias' in name:
                nn.init.constant_(param, 0)
            elif 'weight' in name:
                nn.init.orthogonal_(param)

        if self.two_transitions:
            assert not self.skip_connection
            assert not self.append_a_cog
            assert self.recurse_depth == 1

            self.recurse_rnn = nn.GRUCell(gru_in_size, hidden_size, bias=True)
            for name, param in self.recurse_rnn.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

        assert recurse_depth >= 1

    def time_transition(self, x_in, hxs, done, action_cog):
        if self.append_a_cog:
            x_in = torch.cat((x_in, action_cog.float().detach()), dim=1)

        hxs = hxs * (1 - done.view(-1, 1))
        x = x_in
        for i in range(self.recurse_depth):
            hxs = self.gru(x, hxs)
        x = hxs

        if self.skip_connection:
            x = torch.cat([x_in, x], dim=1)

        return x, hxs

    def recurse(self, x_in, hxs, done, action_cog):
        hxs = self.recurse_rnn(x_in, hxs)
        return hxs, hxs

    def forward(self, x_in, hxs, done, action_cog):
        x, hxs = self.time_transition(x_in, hxs, done, action_cog)

        if self.two_transitions:
            deep_x, deep_hxs = self.recurse(x_in, hxs, done, action_cog)
            x = action_cog * x + (1 - action_cog) * deep_x
            hxs = action_cog * hxs + (1 - action_cog) * deep_hxs

        return x, hxs


class CRNNTransition(nn.Module):
    def __init__(self, state_channels, spatial_shape, recurse_depth=1, append_a_cog=False, append_coords=False, pool_inject=False, skip_connection=False, two_transitions=False):
        super(CRNNTransition, self).__init__()
        self.append_a_cog = append_a_cog
        self.append_coords = append_coords
        self.two_transitions = two_transitions
        self.pool_inject = pool_inject
        self.recurse_depth = recurse_depth
        self.skip_connection = skip_connection

        n_channels = state_channels + int(append_a_cog) + 3*int(append_coords)
        self.in_shape = (state_channels,) + spatial_shape

        if self.skip_connection:
            self.out_shape = (state_channels + n_channels,) + spatial_shape
        else:
            self.out_shape = (state_channels,) + spatial_shape
        self.state_channels = state_channels
        self.recurrent_hidden_state_size = (2*state_channels,) + spatial_shape

        self.conv_lstm = ConvLSTMCell(
            input_dim=n_channels, hidden_dim=state_channels, kernel_size=(3,3), bias=True,
        )
        if self.pool_inject:
            self.pj_net = PoolAndInject(spatial_shape, state_channels)

        if self.append_coords:
            self.coord_conv = CoordConv(spatial_shape)

        assert recurse_depth >= 1

        if self.two_transitions:
            assert not self.skip_connection
            assert not self.append_a_cog
            assert not self.pool_inject
            assert self.recurse_depth == 1

            self.recurse_rnn = self.conv_lstm = ConvLSTMCell(
                input_dim=n_channels, hidden_dim=state_channels, kernel_size=(3,3), bias=True,
            )

            for name, param in self.recurse_rnn.named_parameters():
                if 'bias' in name:
                    nn.init.constant_(param, 0)
                elif 'weight' in name:
                    nn.init.orthogonal_(param)

        assert recurse_depth >= 1

    def time_transition(self, x, hxs, done, action_cog):
        if self.pool_inject:
            x = self.pj_net(x)

        if self.append_a_cog:
            # cover spatial dimension and cat to channel
            action_cog = action_cog.detach().float().view((-1, 1, 1, 1)).expand(-1, -1, *self.in_shape[-2:])
            x = torch.cat((x, action_cog), dim=1)

        if self.append_coords:
            x = self.coord_conv(x)

        expansion_dims = ((hxs.dim() - 1) * (1,))
        hxs = hxs * (1 - done.view(-1, *expansion_dims))
        hxs = hxs.split(self.state_channels, dim=1)

        for i in range(self.recurse_depth):
            hxs = self.conv_lstm(x, hxs)

        h, c = hxs
        hxs = torch.cat(hxs, dim=1)

        if self.skip_connection:
            h = torch.cat([x, h], dim=1)

        return h, hxs

    def recurse(self, x_in, hxs, done, action_cog):
        hxs = hxs.split(self.state_channels, dim=1)
        hxs = self.recurse_rnn(x_in, hxs)
        h, c = hxs
        hxs = torch.cat(hxs, dim=1)

        return h, hxs

    def forward(self, x_in, hxs, done, action_cog):
        x, hxs = self.time_transition(x_in, hxs, done, action_cog)

        if self.two_transitions:
            deep_x, deep_hxs = self.recurse(x_in, hxs, done, action_cog)

            expansion_dims = ((hxs.dim() - 1) * (1,))
            ac_viewed = action_cog.view(-1, *expansion_dims)
            x = ac_viewed * x + (1 - ac_viewed) * deep_x
            hxs = ac_viewed * hxs + (1 - ac_viewed) * deep_hxs

        return x, hxs


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
                self.net = nn.Sequential(
                    init_relu(nn.Linear(in_shape[0], out_shape[0])),
                    nn.ReLU()
                )
            else:
                raise NotImplementedError

    def forward(self, x):
        x = self.net(x)
        return x


class Encoder(nn.Module):
    def __init__(self, obs_space, hidden_size=512, state_channels=32, append_coords=False):
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
                    CoordConv(obs_space.shape[-2:]) if append_coords else nn.Identity(),
                    init_relu(nn.Conv2d(n_channels+3*int(append_coords), 32, 8, stride=4)), nn.ReLU(),
                    init_relu(nn.Conv2d(32, 64, 4, stride=2)), nn.ReLU(),
                    init_relu(nn.Conv2d(64, state_channels, 3, stride=1)), nn.ReLU(),
                )
            elif im_size == 64:
                self.net = nn.Sequential(
                    CoordConv(obs_space.shape[-2:]) if append_coords else nn.Identity(),
                    # input (x, 64, 64)
                    init_relu(nn.Conv2d(n_channels+3*int(append_coords), 32, 6, stride=4, padding=1)), nn.ReLU(),
                    # input (3, 16, 16)
                    init_relu(nn.Conv2d(32, 64, 4, stride=2, padding=2)), nn.ReLU(),
                    # input (3, 9, 9)
                    init_relu(nn.Conv2d(64, state_channels, 3, stride=1)), nn.ReLU(),
                    # input (3, 7, 7)
                )
            # minipacman 15x19
            elif im_size == 15:
                self.net = nn.Sequential(
                    CoordConv(obs_space.shape[-2:]) if append_coords else nn.Identity(),
                    init_relu(nn.Conv2d(n_channels+3*int(append_coords), 32, (5, 5), stride=1)), nn.ReLU(),
                    init_relu(nn.Conv2d(32, 64, (3, 5), stride=1)), nn.ReLU(),
                    init_relu(nn.Conv2d(64, state_channels, (3, 5), stride=1)), nn.ReLU(),
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
            state_channels=32,
            hidden_size=128,
            twoAM=False,
            random_cog_fraction=0.0,
            fixed_recursive_depth=1,
            append_a_cog=False,
            append_coords=False,
            pool_and_inject=False,
            detach_cognition=False,
            skip_connection=False,
            two_transitions=False,
            amnesia=False,
    ):
        super(RecursivePolicy, self).__init__()
        self.amnesia = amnesia
        if amnesia:
            self.amnesia_multiplier = 0
        else:
            self.amnesia_multiplier = 1

        self.twoAM = twoAM
        self.detach_cog = detach_cognition
        self.random_cog_fraction = random_cog_fraction
        self.hidden_size = hidden_size
        self.architecture = architecture
        self.is_recurrent = architecture in ["rnn", "crnn"]
        self.spatial_state_shape = (7, 7) if architecture == "crnn" else ()  # dictated by layer choice in encoder

        self.encoder = Encoder(obs_space, hidden_size, state_channels, append_coords=append_coords)

        if architecture == "ff":
            self.transition = NoTransition(hidden_size)
        elif architecture == "rnn":
            self.transition = RNNTransition(
                hidden_size=hidden_size,
                recurse_depth=fixed_recursive_depth,
                append_a_cog=append_a_cog,
                skip_connection=skip_connection,
                two_transitions=two_transitions,
            )
        elif architecture == "crnn":
            self.transition = CRNNTransition(
                state_channels=state_channels,
                spatial_shape=self.spatial_state_shape,
                recurse_depth=fixed_recursive_depth,
                append_a_cog=append_a_cog,
                append_coords=append_coords,
                pool_inject=pool_and_inject,
                skip_connection=skip_connection,
                two_transitions=two_transitions,
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

        assert 0.0 <= random_cog_fraction <= 1.0
        if fixed_recursive_depth > 1:
            assert random_cog_fraction == 0.0
            assert self.is_recurrent

    def recurrent_hidden_state_size(self):
        if self.is_recurrent:
            return self.transition.recurrent_hidden_state_size
        else:
            return (0,)

    def forward(self, obs, rnn_h, done, a_cog=None):
        x = self.encoder(obs)

        x_enc = self.enc2trans(x)
        x_trans, rnn_h = self.transition(x_enc, rnn_h * self.amnesia_multiplier, done, a_cog)
        in_ac_env = self.trans2ac(x_trans)
        value = self.ac_env.forward_critic(in_ac_env)
        dist = self.ac_env.forward_actor(in_ac_env)
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
            # in_cog = torch.cat((in_ac_env.detach(), obs), dim=1).detach()
            # in_cog = torch.cat((x_trans, x_enc), dim=1).detach()
            if self.detach_cog:
                in_cog  = self.trans2ac(x_trans.detach())
            else:
                in_cog = self.trans2ac(x_trans)
            value_cog = self.ac_cog.forward_critic(in_cog)
            dist_cog = self.ac_cog.forward_actor(in_cog)
            # value_cog = self.ac_cog.forward_critic(in_ac_env.detach())
            # dist_cog = self.ac_cog.forward_actor(in_ac_env.detach())
            # value_cog = self.ac_cog.forward_critic(x)
            # dist_cog = self.ac_cog.forward_actor(x)
            action_cog = dist_cog.sample()

            action_cog_log_probs = dist_cog.log_probs(action_cog)
            dist_cog_entropy = dist_cog.entropy().unsqueeze(-1)

            cog_policy = PolicyOutput(
                value=value_cog,
                action=action_cog,
                action_log_probs=action_cog_log_probs,
                dist_entropy=dist_cog_entropy
            )

        return env_policy, cog_policy, rnn_h

