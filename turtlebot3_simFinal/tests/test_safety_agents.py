"""Part B/C unit tests: TD targets, policies, Algorithm 1 storage, controllers, DQN."""
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from agents.dqn.dqn import DQNAgent, DQNConfig, dqn_targets  # noqa: E402
from agents.gvf.controller import ControllerConfig, PaperController  # noqa: E402
from agents.gvf.policies import BehaviourPolicy, PolicyConfig, TargetPolicy  # noqa: E402
from agents.gvf.predictor import GVFBank, GVFReplay, PredictorConfig, td_targets  # noqa: E402

NAMES = ["FORWARD", "LEFT", "RIGHT", "STOP"]


@pytest.mark.parametrize("ordered", [False, True])
def test_policy_probabilities_sum_to_one(ordered):
    for P in (TargetPolicy(4 if not ordered else 5, ordered, PolicyConfig()),
              BehaviourPolicy(4 if not ordered else 5, ordered, PolicyConfig())):
        assert np.allclose(P.P.sum(1), 1.0) and (P.P >= 0).all()


def test_target_and_behaviour_structure():
    pi, mu = TargetPolicy(4, False, PolicyConfig(eps_pi=0.1)), BehaviourPolicy(4, False, PolicyConfig())
    assert pi.probs(2)[2] == pytest.approx(0.9) and pi.probs(2)[0] == pytest.approx(0.1 / 3)
    assert mu.probs(1)[1] == pytest.approx(0.70 + 0.05 / 4)
    assert mu.probs(1)[0] == pytest.approx(0.25 / 3 + 0.05 / 4)
    assert not np.allclose(pi.P, mu.P)                                  # off-policy, as in the paper
    g = TargetPolicy(5, True, PolicyConfig(sigma_pi=0.5)).probs(2)
    assert np.argmax(g) == 2 and g[1] == pytest.approx(g[3])           # discretised Gaussian


def test_td_target_terminal_and_truncation():
    cf, cv = np.array([1.0, 0.0, 1.0]), np.array([0.5, 0.0, 0.5])
    qf, qv = np.array([0.8, 0.8, 0.8]), np.array([0.4, 0.4, 0.4])
    cont = np.array([1.0, 0.0, 1.0])        # row 1: collision (terminal); row 2: truncation -> bootstraps
    yf, yv = td_targets(cf, cv, cont, qf, qv, 0.95)
    assert yf[0] == pytest.approx(0.05 + 0.95 * 0.8) and yf[1] == 0.0
    assert yf[2] == pytest.approx(yf[0]) and yv[2] == pytest.approx(0.05 * 0.5 + 0.95 * 0.4)


def test_algorithm1_replay_stores_target_action_and_mask():
    rb = GVFReplay(10, 3, seed=0)
    rb.add(np.ones(3), 2, 0.0, 0.5, True, np.zeros(3), 1)       # terminated
    rb.add(np.ones(3), 0, 1.0, 0.5, False, np.zeros(3), 3)      # truncated or ordinary step
    assert rb.cont.tolist()[:2] == [0.0, 1.0] and rb.ap.tolist()[:2] == [1, 3]


def test_gvf_converges_on_toy_chain():
    """Self-loop with c = 1 and no termination: q -> 1 (the (1-gamma) scaling
    normalises the infinite sum); with termination every step: q -> (1-gamma)."""
    for cont, expect in ((1.0, 1.0), (0.0, 0.05)):
        bank = GVFBank(4, 2, PredictorConfig(gammas=(0.95,), lr=3e-3, batch=32, learning_starts=1,
                                             train_every=1, hidden=(16,)), seed=0)
        s = np.full(4, 0.5, np.float32)
        for _ in range(64):
            bank.replay.add(s, 0, 1.0, 1.0, cont == 0.0, s, 0)
        for t in range(1, 6000):
            bank.maybe_train(t)
        qf, qv = bank.nets[0.95].predict(s)
        assert qf[0, 0] == pytest.approx(expect, abs=0.03) and qv[0, 0] == pytest.approx(expect, abs=0.03)


class _StubNet:
    def __init__(self, qf, qv):
        self.qf, self.qv = np.asarray(qf, float)[None], np.asarray(qv, float)[None]

    def predict(self, obs):
        return self.qf, self.qv


class _StubBank:
    def __init__(self, qf, qv):
        self.nets = {0.975: _StubNet(qf, qv)}


def _ctrl(qf, qv, a_prev, **kw):
    c = PaperController(_StubBank(qf, qv), NAMES, ControllerConfig(**kw), v_max=0.22)
    c.a_prev = NAMES.index(a_prev)
    return NAMES[c.act(None, {"linear_velocity": 0.1})]


def test_paper_controller_rules():
    kw = dict(beta1=0.5, beta2=0.8)
    assert _ctrl([0.2, 0.9, 0.4, 0.6], [0.5] * 4, "FORWARD", **kw) == "LEFT"        # unsafe -> argmax
    assert _ctrl([0.2, 0.3, 0.3, 0.3], [0.5] * 4, "FORWARD", **kw) == "STOP"        # ties -> STOP
    assert _ctrl([0.9, 0.9, 0.9, 0.9], [0.5] * 4, "STOP", **kw) == "FORWARD"        # safe & slow -> accelerate
    assert _ctrl([0.9, 0.9, 0.9, 0.9], [0.99] * 4, "FORWARD", **kw) == "STOP"       # predicted too fast
    assert _ctrl([0.7, 0.9, 0.9, 0.9], [0.5] * 4, "FORWARD", **kw) == "FORWARD"     # hysteresis band
    assert _ctrl([0.9] * 4, [0.99] * 4, "STOP", use_speed_gvf=False, **kw) == "FORWARD"  # Alg.3 uses current v


def test_dqn_target_masks_termination_only():
    q_next = np.array([[1.0, 3.0], [1.0, 3.0]])
    y = dqn_targets(np.array([1.0, 1.0]), np.array([1.0, 0.0]), q_next, 0.9)
    assert y.tolist() == pytest.approx([1.0, 1.0 + 0.9 * 3.0])


def test_dqn_learns_terminal_value():
    ag = DQNAgent(4, 2, DQNConfig(hidden=(16,), lr=3e-3, batch=32, learning_starts=1, train_every=1,
                                  target_update=50, gamma=0.5), seed=0)
    s = np.full(4, 0.5, np.float32)
    for _ in range(64):
        ag.replay.add(s, 1, 2.0, s, True)
    for t in range(1, 3000):
        ag.maybe_train(t)
    assert ag.q.forward(s[None])[0, 1] == pytest.approx(2.0, abs=0.1)


def test_frozen_copy_is_independent():
    bank = GVFBank(4, 2, PredictorConfig(gammas=(0.95,), hidden=(8,), learning_starts=1, train_every=1), seed=0)
    fz = bank.frozen_copy()
    s = np.ones(4, np.float32)
    before = fz.nets[0.95].predict(s)[0].copy()
    for _ in range(10):
        bank.replay.add(s, 0, 1.0, 1.0, False, s, 0)
    for t in range(1, 50):
        bank.maybe_train(t)
    assert np.array_equal(before, fz.nets[0.95].predict(s)[0])
    assert not np.array_equal(before, bank.nets[0.95].predict(s)[0])
