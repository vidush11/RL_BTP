"""The Horde: GQ(lambda) run for every demon at once.

theta, w and e are (n_demons x n_features) matrices, so one call to update()
advances the entire bank. Per-demon quantities (gamma, lambda, rho, r, z) are
length-D vectors. Works fast because we have D~8 a realtively small number of demons

Author: Vidush Jindal(jindalv)
"""

import numpy as np

from env import N_ACTIONS, Action
from features import N_ACTIVE, N_FEATURES, phi


class Horde:
    """
    A general class to perform updates on demons according to horde architecture.
    It contains the parameter theta which can be used to compute state action values
    """

    def __init__(self, demons):
        self.demons = demons
        no_D = len(demons)
        self.theta = np.zeros((no_D, N_FEATURES))
        self.w = np.zeros((no_D, N_FEATURES))
        self.e = np.zeros((no_D, N_FEATURES))
        self.lam = np.array([demon["lam"] for demon in demons])
        # scale step sizes by the number of active features, as we would take sum
        # in computing q(s,a)= theta * phi(s,a) and theta computation uses a_th, a_w
        self.a_th = np.array([demon["alpha"] for demon in demons]) / N_ACTIVE
        self.a_w = np.array([demon["beta"] for demon in demons]) / N_ACTIVE

    def q(self, obs: np.ndarray) -> np.ndarray:
        """Returns the state action value i.e q^(s,a,theta)

        :param obs: observation from sensors that is s
        :type obs: np.ndarray
        :return: State action values per demon, the shape is (D X actions)
        :rtype: np.ndarray
        """

        # demons share phi(obs,a)
        return np.array([self.theta[:, phi(obs, a)].sum(1) for a in Action]).T

    def pi(self, obs: np.ndarray, q_sa: np.ndarray) -> np.ndarray:
        """Returns the stochastic distribution of target policy actions for each demon

        :param obs: current observation from the sensors
        :type obs: np.ndarray
        :param Q_sa: the state actions values for all demons as a matrix (Demons X Actions)
        :type Q_sa: np.ndarray
        :return: Target policy probabilities. Control demons derive it from Q[demon]
        :rtype: np.ndarray
        """

        return np.stack(
            [demon["pi"](obs, q_sa[ind]) for ind, demon in enumerate(self.demons)]
        )

    def update(self, obs: np.ndarray, a: Action, obs2: np.ndarray, mu: float):
        """
        We take an action a in obs and get obs2, rewards.
        This updates the state varaibles of horde architecture w, theta, e.

        :param obs: The current sensor readings
        :type obs: np.ndarray
        :param a: Target policy action
        :type a: Action
        :param obs2: Next state sensor readings
        :type obs2: np.ndarray
        :param mu: the probability of take action a in behavious policy
        :type mu: float
        :return: del_t a temporary quantity that can be used to detect divergence
        :rtype: _type_
        """
        idx = phi(obs, a)
        idx2 = [phi(obs2, action) for action in Action]
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
        c = self.a_th * g2 * (1 - self.lam) * wte  # the gradient correction
        for b in range(N_ACTIONS):
            self.theta[:, idx2[b]] -= (c * pi_2[:, b])[:, None]

        self.w += (self.a_w * delta)[:, None] * self.e
        self.w[:, idx] -= (self.a_w * wphi)[:, None]
        return delta
