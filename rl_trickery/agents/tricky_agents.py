import torch
import numpy as np


def compute_returns(
        v_pred, r,
        done, timeout,
        gamma,
        use_timeout=True
):
    """
    >>> np.array(compute_returns([0.5]*7, [0.1,0.,1.,0.1,0.1,1.], [0.,0.,1.,0.,1.,0.], [0.,0.,0.,0.,1.,0.], 0.9))
    array([0.91, 0.9 , 1.  , 0.55, 0.5 , 1.45])
    """

    v = []
    v_discounted = gamma * v_pred[-1]
    for t in reversed(range(len(r))):
        # timeout than copy over best estimate for this timestep as last value
        # otherwise usual discounting of future values, and setting to zero if done
        if use_timeout:
            v_t = ((1 - done[t]) * v_discounted + r[t]) * (1 - timeout[t]) + v_pred[t] * timeout[t]
        else:
            v_t = (1 - done[t]) * v_discounted + r[t]
        v.append(v_t)
        v_discounted = gamma * v_t
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

    def compute_returns(self, gamma, use_timeout):
        # usual returns
        if not self.a_c:
            self.v_target = compute_returns(
                self.v, self.r,
                self.done, self.timeout,
                gamma,
                use_timeout=use_timeout
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

        self.max_grad_norm = max_grad_norm
        self.optimizer_type = optimizer_type.lower()

        if self.optimizer_type == "rmsprop":
            self.optimizer = torch.optim.RMSprop(self.net.parameters(), lr, eps=eps, alpha=alpha)
        elif self.optimizer_type == "adam":
            self.optimizer = torch.optim.Adam(self.net.parameters(), lr, eps=eps)
        elif self.optimizer_type == "sgd":
            self.optimizer = torch.optim.SGD(self.net.parameters(), lr)
        else:
            raise NotImplementedError

    def update(self):
        self.buf.compute_returns(self.gamma, self.use_timeout)

        self.optimizer.zero_grad()

        v = torch.stack(self.buf.v[:-1])
        v_target = torch.stack(self.buf.v_target)
        a_logp = torch.stack(self.buf.a_logp)
        ent = torch.stack(self.buf.a_ent)

        adv = v_target - v
        v_loss = adv.pow(2).mean()
        a_loss = -(adv.detach() * a_logp).mean()
        ent_loss = ent.mean()

        env_loss = (v_loss * self.value_loss_coef + a_loss - ent_loss * self.entropy_coef)

        env_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return v_loss, a_loss, ent_loss



if __name__ == "__main__":
    import doctest
    doctest.testmod()


