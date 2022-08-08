from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

import torch
import time
import math
import torch.nn.functional as F

from autoattack.autopgd_base import L1_projection


class GradientFreeAttack:
    """
    :param predict:       forward pass function
    :param seed:          random seed for the starting point
    """

    def __init__(self, predict, eps=None, seed=0, verbose=False, device=None):
        """ """

        self.predict = predict
        self.eps = eps
        self.seed = seed
        self.verbose = verbose
        self.device = device
        self.return_all = False

    def init_hyperparam(self, x):
        assert not self.eps is None
        assert self.loss in ["ce", "margin"]

        if self.device is None:
            self.device = x.device
        self.orig_dim = list(x.shape[1:])
        self.ndims = len(self.orig_dim)
        if self.seed is None:
            self.seed = time.time()

    def l2_norm(self, x):
        t = (x ** 2).view(x.shape[0], -1).sum(-1).sqrt()
        return t.view(-1, *([1] * self.ndims))

    def random_target_classes(self, y_pred, n_classes):
        y = torch.zeros_like(y_pred)
        for counter in range(y_pred.shape[0]):
            l = list(range(n_classes))
            l.remove(y_pred[counter])
            t = self.random_int(0, len(l))
            y[counter] = l[t]

        return y.long().to(self.device)

    def solve(self, x, l):
        ## find Q(x)
        return 0

    def perturb(self, x, delta_ws, L, n_1, n_2, n_3, n_4, alpha):
        with torch.no_grad():
            delta_0 = self.solve(x, L)

            if self.l2_norm(delta_0) < self.l2_norm(delta_ws):
                delta = delta_0
                u = self.l2_norm(delta_0)
            else:
                delta = delta_ws
                u = self.l2_norm(delta_ws)

            S = torch.full(n_1, delta)

            for a_1 in range(1, n_4 + 1):
                for a_2 in range(1, n_3 + 1):
                    R = torch.empty(0)

                    for i in range(1, n_1 + 1):
                        for j in range(1, n_2 + 1):
                            epsilon = torch.distributions.binomial.Binomial(
                                0, u / a_1
                            ).sample()
                            R = torch.cat((R, x + S[i] + epsilon), 0)

                for y in R:
                    delta_temp = self.solve(y, L)

                    if self.l2_norm(delta_temp) < u:
                        delta = delta_0
                        u = self.l2_norm(delta_0)

                    m = (S == torch.max(S)).nonzero(as_tuple=True)

                    if self.l2_norm(delta_temp) < alpha * u and self.l2_norm(
                        delta_temp
                    ) < self.l2_norm(S[m]):
                        S[m] = delta_temp

            for j in range(1, n_2 + 1):
                epsilon = torch.distributions.binomial.Binomial(
                    0, u / a_1
                ).sample()
                y = x + delta + epsilon
                delta_temp = self.solve(y, L)

                if self.l2_norm(delta_temp) < u:
                    delta = delta_0
                    u = self.l2_norm(delta_0)

                m = (S == torch.max(S)).nonzero(as_tuple=True)

                if self.l2_norm(delta_temp) < alpha * u and self.l2_norm(
                    delta_temp
                ) < self.l2_norm(S[m]):
                    S[m] = delta_temp

            return delta
