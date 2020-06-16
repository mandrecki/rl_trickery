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


def compute_returns(
        v_pred, r,
        done, timeout,
        gamma,
        use_timeout=False,
        rescale=False,
):
    """
    >>> np.array(compute_returns([0.5]*7, [0.1,0.,1.,0.1,0.1,1.], [0.,0.,1.,0.,1.,0.], [0.,0.,0.,0.,1.,0.], 0.9))
    array([0.91, 0.9 , 1.  , 0.55, 0.5 , 1.45])
    """
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

        self.buffers = [self.s, self.a, self.r, self.done, self.timeout,  self.h, self.v, self.a_logp, self.a_ent]

        self.a_c = []
        self.r_c = []
        self.v_c = []
        self.v_c_target = []

        self.buffers_cog = [self.a_c, self.r_c, self.v_c]

    def append(self, *args):
        for b, val in zip(self.buffers, args):
            b.append(val)

    def append_cog(self, *args):
        for b, val in zip(self.buffers_cog, args):
            b.append(val)

    def after_update(self):
        for i in range(len(self.buffers)):
            # last = self.buffers[i][-1].detach()
            self.buffers[i].clear()
            # self.buffers[i].append(last)
        self.v_target = []

        if not self.a_c:
            for i in range(len(self.buffers_cog)):
                # last = self.buffers[i][-1].detach()
                self.buffers[i].clear()
                # self.buffers[i].append(last)

    def compute_returns(self, gamma, use_timeout, rescale):
        with torch.no_grad():
            if not self.a_c:
                self.v_target = compute_returns(
                    self.v, self.r,
                    self.done, self.timeout,
                    gamma=gamma,
                    use_timeout=use_timeout,
                    rescale=rescale,
                )
            else:
                raise NotImplementedError


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
            long_horizon=False,
            cognition_cost=0.1,
            cognitive_coef=0.5,
            only_action_values=True,
            optimizer_type="RMSprop",
            smooth_value_loss=False,
            reward_rescale=False,
    ):
        self.net = net
        self.buf = buffer

        self.gamma = gamma
        self.use_timeout = use_timeout
        self.cognitive_coef = cognitive_coef
        self.long_horizon = long_horizon
        self.cognition_cost = cognition_cost
        self.only_action_values = only_action_values
        self.gamma_cog = 0.9
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

        env_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return v_loss, a_loss, ent_loss


if __name__ == "__main__":
    import doctest
    doctest.testmod()


