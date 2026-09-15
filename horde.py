"""The Horde: GQ(lambda) run for every demon at once.

theta, w and e are (n_demons x n_features) matrices, so one call to update()
advances the entire bank. Per-demon quantities (gamma, lambda, rho, r, z) are
length-D vectors. This is the only structural decision that is painful to
retrofit, which is why it is here from the start.
"""
import numpy as np
from env import N_ACTIONS
from features import phi, N_FEATURES, N_ACTIVE


class Horde:
    def __init__(self, demons):
        self.demons = demons
        D = len(demons)
        self.theta = np.zeros((D, N_FEATURES))
        self.w = np.zeros((D, N_FEATURES))
        self.e = np.zeros((D, N_FEATURES))
        self.lam = np.array([d["lam"] for d in demons])
        # scale step sizes by the number of active features, so alpha is comparable
        self.a_th = np.array([d["alpha"] for d in demons]) / N_ACTIVE
        self.a_w = np.array([d["beta"] for d in demons]) / N_ACTIVE

    def q(self, obs):
        """(D, A) matrix: every demon's action-values at obs."""
        return np.stack([self.theta[:, phi(obs, a)].sum(1) for a in range(N_ACTIONS)], 1)

    def pi(self, obs, Q):
        """(D, A) target policy probabilities. Control demons read their own Q."""
        return np.stack([d["pi"](obs, Q[i]) for i, d in enumerate(self.demons)])

    def update(self, obs, a, obs2, mu):
        idx = phi(obs, a)
        idx2 = [phi(obs2, b) for b in range(N_ACTIONS)]
        Q2 = np.stack([self.theta[:, j].sum(1) for j in idx2], 1)
        pi_t = self.pi(obs, self.q(obs))
        pi_2 = self.pi(obs2, Q2)

        g = np.array([d["gamma"](obs) for d in self.demons])
        g2 = np.array([d["gamma"](obs2) for d in self.demons])
        r = np.array([d["r"](obs2) for d in self.demons])
        z = np.array([d["z"](obs2) for d in self.demons])

        # delta_t = r + (1-gamma')z + gamma' * theta^T phi_bar - theta^T phi
        q_bar = (pi_2 * Q2).sum(1)
        delta = r + (1 - g2) * z + g2 * q_bar - self.theta[:, idx].sum(1)

        # e_t = phi + gamma*lambda*rho*e_{t-1}; rho=0 zeroes the trace off-policy
        rho = pi_t[:, a] / mu
        self.e *= (g * self.lam * rho)[:, None]
        self.e[:, idx] += 1.0

        wte = (self.w * self.e).sum(1)
        wphi = self.w[:, idx].sum(1)

        self.theta += (self.a_th * delta)[:, None] * self.e
        c = self.a_th * g2 * (1 - self.lam) * wte           # the gradient correction
        for b in range(N_ACTIONS):
            self.theta[:, idx2[b]] -= (c * pi_2[:, b])[:, None]

        self.w += (self.a_w * delta)[:, None] * self.e
        self.w[:, idx] -= (self.a_w * wphi)[:, None]
        return delta
