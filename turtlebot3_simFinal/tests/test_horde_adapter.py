"""A6: RL_BTP (unedited copies) runs on this simulator through the adapter."""
import hashlib
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from simulator import TurtleBot3LidarEnv, load_config  # noqa: E402
from agents.horde.adapter import RLBTP_DIR, HordeAdapter, load_rlbtp  # noqa: E402


def test_rlbtp_copies_are_unedited():
    for line in (RLBTP_DIR / "SHA256SUMS").read_text().splitlines():
        h, name = line.split()
        assert hashlib.sha256((RLBTP_DIR / name).read_bytes()).hexdigest() == h


def test_adapter_sensor_layout():
    env = TurtleBot3LidarEnv(load_config(ROOT / "configs" / "safety_task.yaml"))
    ad = HordeAdapter(env)
    assert ad.idx[0] == 0
    assert abs(ad.idx_angles[1] - 1.0) <= np.radians(2.5) and abs(ad.idx_angles[2] + 1.0) <= np.radians(2.5)
    _, info = env.reset(seed=0)
    s = ad.sensors(info)
    rl = load_rlbtp()["env"]
    assert s.shape == (rl.N_SENSORS,)
    assert s[rl.IR_F] == np.exp(-info["ranges_m"][0] / 0.6)
    assert s[rl.COMPASS] == np.cos(info["pose"][2]) and s[rl.BUMP] == 0.0
    assert [ad.rlbtp_action(i) for i in range(4)] == [rl.Action.FORWARD, rl.Action.LEFT,
                                                      rl.Action.RIGHT, rl.Action.STOP]
    assert ad.env_action(rl.Action.BACKWARD) is None


def test_rlbtp_horde_smoke_runs_on_simulator():
    rl = load_rlbtp()
    horde = rl["horde"].Horde(rl["demons"].DEMONS)
    env = TurtleBot3LidarEnv(load_config(ROOT / "configs" / "safety_task.yaml"))
    ad = HordeAdapter(env)
    rng = np.random.default_rng(0)
    _, info = env.reset(seed=0)
    obs, prev, episodes = ad.sensors(info), 0, 0
    n = env.action_space.n
    for t in range(3000):
        p = np.full(n, 0.2 / n); p[prev] += 0.8
        a = int(rng.choice(n, p=p)); prev = a
        _, _, te, tr, info = env.step(a)
        obs2 = ad.sensors(info)
        delta = horde.update(obs, ad.rlbtp_action(a), obs2, float(p[a]))
        assert np.all(np.isfinite(delta))
        obs = obs2
        if te or tr:
            _, info = env.reset(); obs = ad.sensors(info); horde.e[:] = 0.0; episodes += 1
    assert episodes >= 1 and np.all(np.isfinite(horde.theta))
    assert horde.q(obs).shape == (len(rl["demons"].DEMONS), rl["env"].N_ACTIONS)
