"""Is seek-light working? Learned greedy vs random vs a hand-coded 'turn to light, drive' policy."""
import sys; sys.path.insert(0, sys.argv[1])
import numpy as np
from demons import DEMONS
from env import LIGHT, LIGHT_FL, LIGHT_BR, N_ACTIONS, Action, Robot
from horde import Horde
ck = np.load(sys.argv[2]); horde = Horde(DEMONS); horde.theta = ck["theta"]
i = [d["name"] for d in DEMONS].index("seek-light [ctrl]"); r = DEMONS[i]["r"]
rng = np.random.default_rng(0); vis = ck["visited"]
starts = vis[rng.choice(len(vis), 200, replace=False)]
def bearing(sim):  # light direction relative to heading, in (-pi, pi], + = left
    dx, dy = LIGHT - [sim.x, sim.y]
    return (np.arctan2(dy, dx) - sim.th + np.pi) % (2 * np.pi) - np.pi
def oracle(sim, o):  # uses true geometry, so it ignores shadows and pillars
    b = bearing(sim)
    return Action.LEFT if b > 0.3 else Action.RIGHT if b < -0.3 else Action.FORWARD
sim = Robot(seed=1)
print(f"{'policy':<10}{'reward/step':>12}{'  lit start':>12}{'shadow start':>13}{'final dist':>11}{'%t <0.5m':>9}{'%reach<0.5m':>12}{'%t dark':>8}")
for name in ["learned", "random", "oracle"]:
    R, lit, dark_s, fd, near, reach, dark = [], [], [], [], [], [], []
    for st in starts:
        sim.set_state(tuple(st)); o = sim.obs(); start_lit = o[LIGHT_FL:LIGHT_BR + 1].max() > 0.02
        tot = n_near = n_dark = 0.0
        for _ in range(250):
            a = (Action(np.argmax(horde.q(o)[i])) if name == "learned" else
                 Action(rng.integers(N_ACTIONS)) if name == "random" else oracle(sim, o))
            o = sim.step(a); tot += r(o)
            dist = np.hypot(sim.x - LIGHT[0], sim.y - LIGHT[1]); n_near += dist < 0.5
            n_dark += o[LIGHT_FL:LIGHT_BR + 1].max() < 0.02
        R.append(tot / 250); (lit if start_lit else dark_s).append(tot / 250); fd.append(dist)
        near.append(n_near / 250); reach.append(n_near > 0); dark.append(n_dark / 250)
    print(f"{name:<10}{np.mean(R):>12.3f}{np.mean(lit):>12.3f}{np.mean(dark_s):>13.3f}{np.mean(fd):>11.2f}"
          f"{100*np.mean(near):>9.1f}{100*np.mean(reach):>12.1f}{100*np.mean(dark):>8.1f}")
print(f"starts lit / shadow: {len(lit)} / {len(dark_s)}")
# per-state: when the light is clearly to one side and visible, which way does seek-light want to turn?
cnt = {"left": np.zeros(N_ACTIONS), "right": np.zeros(N_ACTIONS)}
for st in vis[rng.choice(len(vis), 4000, replace=False)]:
    sim.set_state(tuple(st)); o = sim.obs()
    if o[LIGHT_FL:LIGHT_BR + 1].max() < 0.02: continue
    b = np.degrees(bearing(sim))
    side = "left" if 30 < b < 150 else "right" if -150 < b < -30 else None
    if side: cnt[side][np.argmax(horde.q(o)[i])] += 1
for side, c in cnt.items():
    print(f"light {side:<5} (30-150 deg, lit, n={int(c.sum())}): " + "  ".join(f"{Action(k).name} {100*c[k]/c.sum():.0f}%" for k in range(N_ACTIONS)))
