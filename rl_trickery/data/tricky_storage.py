import numpy as np


def compute_returns(
        v_pred, r,
        done, timeout,
        gamma
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
        v_t = (1 - done[t]) * v_discounted + r[t] * (1 - timeout[t]) + v_pred[t] * timeout[t]
        v.append(v_t)
        v_discounted = gamma * v_t
    v = list(reversed(v))

    return v


class TrickyRollout(object):
    def __init__(self, gamma=0.99):
        self.gamma = gamma

        self.s = []
        self.a = []
        self.r = []
        self.done = []
        self.timeout = []
        self.h = []
        self.v = []
        self.v_target = []

        self.buffers = [self.s, self.a, self.r, self.done, self.timeout,  self.h, self.v, self.v_target]

        self.a_c = []
        self.r_c = []
        self.v_c = []
        self.v_c_target = []

        self.buffers_cog = [self.a_c, self.r_c, self.v_c, self.v_c_target]

    def append(self, *args):
        for b, val in zip(self.buffers, args):
            b.append(val)

    def append_cog(self, *args):
        for b, val in zip(self.buffers_cog, args):
            b.append(val)

    def after_update(self):
        for i in range(len(self.buffers)):
            self.buffers[i] = self.buffers[i][-1:]

        if not self.a_c:
            for i in range(len(self.buffers_cog)):
                self.buffers_cog[i] = self.buffers_cog[i][-1:]

    def compute_raturns(self):
        # usual returns
        if not self.a_c:
            self.v_target = compute_returns(
                self.v, self.r,
                self.done, self.timeout,
                self.gamma
            )
        else:
            raise NotImplementedError


if __name__ == "__main__":
    import doctest
    doctest.testmod()


