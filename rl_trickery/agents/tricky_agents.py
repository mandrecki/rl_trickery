import torch
import numpy as np

from torch.nn import functional as F


# from R2D2 repo - reward scaling
def value_rescale(x, eps=1.0e-3):
    return x.sign() * ((x.abs() + 1.0).sqrt() - 1.0) + eps * x


def inv_value_rescale(x, eps=1.0e-3):
    if eps == 0:
        return x.sign() * (x * x + 2.0 * x.abs())
    else:
        return x.sign() * ((((1.0 + 4.0 * eps * (x.abs() + 1.0 + eps)).sqrt() - 1.0) / (2.0 * eps)).pow(2) - 1.0)


def compute_returns_with_cognition(
        v_pred, r, a_c,
        done, timeout,
        gamma,
        use_timeout=False,
        rescale=False,
):
    v = []
    v_next = v_pred[-1]
    for t in reversed(range(len(r))):
        # if timeout then copy over best estimate for this timestep as last value
        # otherwise usual discounting of future values, and setting to zero if done
        if rescale:
            v_next = inv_value_rescale(v_next)

        if use_timeout:
            v_t = ((1 - done[t]) * gamma * v_next + r[t]) * (1 - timeout[t]) + v_pred[t] * timeout[t]
        else:
            v_t = (1 - done[t]) * gamma * v_next + r[t]

        v_t = a_c[t] * v_t + (1 - a_c[t]) * v_next

        if rescale:
            v_t = value_rescale(v_t)

        v.append(v_t)
        v_next = v_t
    v = list(reversed(v))

    return v

# non-cognitive working returns
def compute_returns(
        v_pred, r,
        done, timeout,
        gamma=0.99,
        use_timeout=False,
        rescale=False,
):
    v = []
    v_next = v_pred[-1]
    for t in reversed(range(len(r))):
        # if timeout then copy over best estimate for this timestep as last value
        # otherwise usual discounting of future values, and setting to zero if done
        if rescale:
            v_next = inv_value_rescale(v_next)

        if use_timeout:
            v_t = ((1 - done[t]) * gamma * v_next + r[t]) * (1 - timeout[t]) + v_pred[t] * timeout[t]
        else:
            v_t = (1 - done[t]) * gamma * v_next + r[t]

        if rescale:
            v_t = value_rescale(v_t)

        v.append(v_t)
        v_next = v_t
    v = list(reversed(v))

    return v


class TrickyRollout(object):
    def __init__(self):
        self.s = []
        self.a = []
        self.r = []
        self.done = []
        self.timeout = []
        self.h = []
        self.v = []
        self.a_logp = []
        self.a_ent = []

        self.v_target = []

        self.buffers = [self.s, self.a, self.r, self.done, self.timeout,  self.h, self.v, self.a_logp, self.a_ent, self.v_target]

        self.a_c = []
        self.a_c_logp = []
        self.a_c_ent = []
        self.r_c = []
        self.v_c = []
        self.v_c_target = []

        self.buffers_cog = [self.v_c, self.a_c, self.a_c_logp, self.a_c_ent, self.r_c, self.v_c_target]

    def append(self, *args):
        for b, val in zip(self.buffers, args):
            b.append(val)

    def append_cog(self, *args):
        for b, val in zip(self.buffers_cog, args):
            b.append(val)

    def after_update(self):
        for i in range(len(self.buffers)):
            self.buffers[i].clear()

        for i in range(len(self.buffers_cog)):
            self.buffers_cog[i].clear()

    def compute_returns(self, gamma, use_timeout, rescale):
        with torch.no_grad():
            self.v_target = compute_returns_with_cognition(
                self.v, self.r, self.a_c,
                self.done, self.timeout,
                gamma=gamma,
                use_timeout=use_timeout,
                rescale=rescale,
            )

import collections
A2CLoss = collections.namedtuple('A2CLoss', 'value, action, entropy')


class A2C(object):
    def __init__(
            self,
            net,
            buffer: TrickyRollout,
            gamma=0.99,
            value_loss_coef=.5,
            entropy_coef=.01,
            lr=None,
            eps=None,
            alpha=None,
            max_grad_norm=None,
            use_timeout=True,
            gamma_cog=0.9,
            cognition_cost=0.2,
            cognition_coef=0.5,
            optimizer_type="RMSprop",
            smooth_value_loss=False,
            reward_rescale=False,
            update_cognitive_values=False,
            twoAM=False,
    ):
        self.net = net
        self.buf = buffer

        self.twoAM = twoAM
        self.gamma_cog = gamma_cog
        self.cognition_coef = cognition_coef
        self.cognition_cost = cognition_cost
        self.update_cognitive_values = update_cognitive_values
        # self.long_horizon = long_horizon
        # self.only_action_values = only_action_values

        self.gamma = gamma
        self.use_timeout = use_timeout

        self.value_loss_coef = value_loss_coef
        self.entropy_coef = entropy_coef

        self.smooth_value_loss = smooth_value_loss
        self.max_grad_norm = max_grad_norm
        self.optimizer_type = optimizer_type.lower()
        self.reward_rescale = reward_rescale

        if self.optimizer_type == "rmsprop":
            self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr, eps=eps, alpha=alpha)
        elif self.optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr, eps=eps)
        elif self.optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr)
        else:
            raise NotImplementedError

    # vanilla a2c update
    def update(self):
        self.buf.compute_returns(self.gamma, self.use_timeout, self.reward_rescale)

        self.optimizer.zero_grad()

        v = torch.stack(self.buf.v[:-1])
        v_target = torch.stack(self.buf.v_target)
        a_logp = torch.stack(self.buf.a_logp)
        ent = torch.stack(self.buf.a_ent)

        if self.smooth_value_loss:
            adv = F.smooth_l1_loss(v, v_target.detach(), reduction="none")
        else:
            adv = F.mse_loss(v, v_target.detach(), reduction="none")
        v_loss = adv.mean()

        full_adv = (v_target - v)
        a_loss = -(full_adv.detach() * a_logp).mean()

        ent_loss = ent.mean()

        env_loss = (v_loss * self.value_loss_coef + a_loss - ent_loss * self.entropy_coef)

        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return v_loss, a_loss, ent_loss

    def cognitive_update(self):
        self.buf.compute_returns(self.gamma, self.use_timeout, self.reward_rescale)

        self.optimizer.zero_grad()

        v = torch.stack(self.buf.v[:-1])
        v_target = torch.stack(self.buf.v_target)
        a_logp = torch.stack(self.buf.a_logp)
        ent = torch.stack(self.buf.a_ent)
        a_c = torch.stack(self.buf.a_c).detach()

        if self.smooth_value_loss:
            adv = F.smooth_l1_loss(v, v_target.detach(), reduction="none")
        else:
            adv = F.mse_loss(v, v_target.detach(), reduction="none")

        if self.update_cognitive_values:
            v_loss = adv.mean()
        else:
            v_loss = adv[a_c == 1].mean()

        full_adv = (v_target - v).detach()
        a_loss_full = -(full_adv * a_logp)
        a_loss = a_loss_full[a_c == 1].mean()

        env_loss = A2CLoss(
            value=v_loss,
            action=a_loss,
            entropy=ent[a_c == 1].mean()
        )
        total_loss = env_loss.value * self.value_loss_coef + env_loss.action - env_loss.entropy * self.entropy_coef

        if self.twoAM:
            cog_loss = self.compute_cognitive_loss(full_adv.pow(2).detach(), a_loss_full.detach(), a_c)
            total_loss += self.cognition_coef \
                          * (cog_loss.value * self.value_loss_coef
                             + cog_loss.action
                             - cog_loss.entropy * self.entropy_coef)
        else:
            cog_loss = A2CLoss(value=0, action=0, entropy=0)

        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return env_loss, cog_loss

    def compute_cognitive_loss(self, env_value_loss, env_action_loss, a_c):
        # REWARD A:
        # if time flows r=0
        # otherwise reward improvement in value estimate
        # always punish thinking with constant
        value_accuracy = (torch.log2(env_value_loss[:-1, ...] + 1e-5) - torch.log2(env_value_loss[1:, ...] + 1e-5))
        action_improvement = -(env_action_loss[:-1, ...] - env_action_loss[1:, ...])

        # punish cognition
        reward_cog = -self.cognition_cost * (1 - a_c[:-1])

        # reward value accuracy improvement only in non-cognitive states
        # reward_cog += value_accuracy * a_c[:-1]

        # reward value accuracy improvement only in cognitive states
        # reward_cog += value_accuracy * (1 - a_c[:-1])

        # reward value accuracy improvement only in cognitive states followed by action state
        reward_cog += value_accuracy * (1 - a_c[:-1]) * a_c[1:]

        # reward action selection improvement
        # reward_cog += action_improvement * a_c[:-1]


        # value_accuracy = env_advantages_squared[:-1, ...] - env_advantages_squared[1:, ...]
        # reward_cog = (1 - a_c[:-1]) * (value_accuracy - self.cognition_cost)

        # REWARD B
        # if time flows, pay for error
        # otherwise, pay constant cost

        # REWARD C

        # returns
        with torch.no_grad():
            cog_returns = compute_returns(
                self.buf.v_c, reward_cog,
                self.buf.done, self.buf.timeout,
                gamma=self.gamma_cog,
                use_timeout=self.use_timeout,
                rescale=self.reward_rescale,
            )

        v = torch.stack(self.buf.v_c[:-2])
        v_target = torch.stack(cog_returns)
        a_logp = torch.stack(self.buf.a_c_logp[:-1])
        ent = torch.stack(self.buf.a_c_ent[:-1])

        adv = (v_target.detach() - v)

        cog_loss = A2CLoss(
            value=adv.pow(2).mean(),
            action=-(-adv.detach() * a_logp).mean(),
            entropy=ent.mean()
        )

        return cog_loss

if __name__ == "__main__":
    pass

