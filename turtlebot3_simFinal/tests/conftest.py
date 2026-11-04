import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from simulator import TurtleBot3LidarEnv  # noqa: E402


@pytest.fixture
def make_env():
    """Factory: ``make_env(**config_overrides)`` -> fresh environment."""
    def _make(**cfg):
        return TurtleBot3LidarEnv(cfg or None)
    return _make


def rollout(env, actions, seed=0, options=None):
    """Run a fixed action list; return per-step tuples (obs, reward, term, trunc)."""
    obs, _ = env.reset(seed=seed, options=options)
    out = [obs.copy()]
    for a in actions:
        obs, r, te, tr, _ = env.step(a)
        out.append((obs.copy(), r, te, tr))
        if te or tr:
            break
    return out
