"""
Figures for the final off-policy (Horde) vs on-policy (Sarsa) comparison.

usage: PYTHONPATH=<code dir> python3 final_figures.py <runs dir with horde/, onpolicy_*/ and onpolicy_episodic_*/>
Writes PNGs + combine_variants.txt next to this file.
"""

import os
import re
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from demons import DEMONS
from env import ARENA, BUMP, LIGHT, LIGHT_FL, LIGHT_FR, N_ACTIONS, OBSTACLES, Action, Robot
from features import phi
from horde import Horde
from main import random_rewards, EVAL_STARTS, STICK

RUNS, OUT = sys.argv[1], os.path.dirname(os.path.abspath(__file__))
NAMES = [d["name"] for d in DEMONS]
SEEK, SPIN, AVOID = (NAMES.index(n) for n in ("seek-light [ctrl]", "spin [ctrl]", "avoid-obstacles [ctrl]"))
STO, BSOON = NAMES.index("steps-to-obstacle"), NAMES.index("bump-soon")
COL = {"seek": "#C97A12", "on": "#7A3E00", "avoid": "#2C62A0", "spin": "#4F5666", "rand": "#A9A39A"}
plt.rcParams.update({"font.size": 12, "axes.spines.top": False, "axes.spines.right": False, "figure.dpi": 150})

# ---------- logs -> arrays
rows, t = [], None
for line in open(f"{RUNS}/horde/train_log.txt"):
    m = re.match(r"steps ([\d,]+)", line)
    if m:
        t = int(m.group(1).replace(",", ""))
        rows.append({"t": t})
    m = re.match(r"\s+(\S+(?: \[ctrl\])?)\s+(-?[\d.]+)\s+(-?[\d.]+)\s+(-?[\d.]+)x?$", line)
    if m and t and m.group(1) in NAMES:
        rows[-1][m.group(1)] = tuple(float(m.group(k)) for k in (2, 3, 4))
rows = [r for r in rows if all(n in r for n in NAMES)]  # drop an eval still being written
T =np.array([r["t"] for r in rows]) / 1e6
H = {n: np.array([r[n] for r in rows]) for n in NAMES}  # (evals, 3)
# on-policy Sarsa(lambda) logs, one run per control demon: columns step, learned, random, ratio
# ON: plain (the robot follows its own policy) · EPI: episodic (random restart every 250 steps)
read_on = lambda d: np.array([[float(v.rstrip("x").replace(",", "")) for v in l.split()] for l in open(f"{RUNS}/{d}/onpolicy_log.txt") if re.match(r"\s+[\d,]+\s", l)])
CTRL3 = ((SEEK, "seek", "seek-light"), (AVOID, "avoid", "avoid-obstacles"), (SPIN, "spin", "spin"))
ON = {i: read_on(f"onpolicy_{key}") for i, key, _ in CTRL3}
EPI = {i: read_on(f"onpolicy_episodic_{key}") for i, key, _ in CTRL3}

# ---------- Fig 5 style: average reward per time step of each control policy (learning off)
fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
for k, (i, key, title) in enumerate(CTRL3):
    ax[k].plot(T, H[NAMES[i]][:, 0], color=COL[key], lw=2, label="off-policy (Horde)")
    ax[k].plot(ON[i][:, 0] / 1e6, ON[i][:, 1], color=COL[key], lw=2, ls="--", alpha=0.8, label="on-policy Sarsa(λ)")
    ax[k].axhline(random_rewards[i], color=COL["rand"], ls=":", lw=2, label="random policy")
    ax[k].set(title=title, xlabel="training steps (millions)")
    ax[k].legend(frameon=False, fontsize=10)
ax[0].set_ylabel("average reward per time step")
fig.suptitle("Control demons: 100 fixed starts × 250 steps, learning off (cf. Horde Fig. 5)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig5_learning_curves.png")

# ---------- Fig 5, one panel: rewards scaled to [0, 1] by their possible range (as the paper did),
# with "steps to competence": first eval where the trailing 100k-step (4-eval) mean reaches
# 90% of the final level (mean of the last 8 evals, i.e. the last 200k steps)
RANGE = {SEEK: (0.0, np.sqrt(2)), AVOID: (-1.0, 1.0), SPIN: (0.0, 1.0)}  # L_FL+L_FR, 1-maxIR-bump, |w|/3
trail = lambda y: np.array([y[max(0, j - 3) : j + 1].mean() for j in range(len(y))])
fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)  # one panel per demon
comp = []
for ax, (i, key, title) in zip(axs, CTRL3):
    lo, hi = RANGE[i]
    base = (random_rewards[i] - lo) / (hi - lo)
    ax.axhline(base, color=COL["rand"], lw=2, ls=":", label="random policy")
    for x, y, how, ls in ((T, H[NAMES[i]][:, 0], "off-policy", "-"), (ON[i][:, 0] / 1e6, ON[i][:, 1], "on-policy", "--")):
        y = (y - lo) / (hi - lo)
        s, final = trail(y), y[-8:].mean()
        j = int(np.argmax(s >= base + 0.9 * (final - base)))  # 90% of the way from random to final
        comp.append((f"{title} ({how})", x[j] * 1e6, final, base))
        ax.plot(x, y, color=COL[key], lw=2, ls=ls, label=how + (" (Horde)" if ls == "-" else " Sarsa(λ)"))
        ax.plot(x[j], y[j], "v" if how == "off-policy" else "^", color=COL[key], ms=12, mec="#141821")
    ax.set(title=title, xlabel="training steps (millions)")
    ax.legend(frameon=False, fontsize=10)
axs[0].set_ylabel("average reward per step, scaled to [0, 1]")
fig.suptitle("Control demons, learning off (cf. Horde Fig. 5) · ▼▲: 90% of final level")
fig.tight_layout()
fig.savefig(f"{OUT}/fig5_normalized.png")
with open(f"{OUT}/steps_to_competence.txt", "w") as fh:
    fh.write("first step where the trailing 100k mean covers 90% of the gap from random to the final level\n")
    fh.write(f"{'demon':<32}{'steps':>12}{'final (0-1)':>13}{'random (0-1)':>14}\n")
    for label, steps, final, base in comp:
        fh.write(f"{label:<32}{steps:>12,.0f}{final:>13.3f}{base:>14.3f}\n")

# ---------- separate Fig 5: off-policy Horde vs episodic on-policy (random restart every 250 steps)
fig, axs = plt.subplots(1, 3, figsize=(15, 4.4), sharey=True)
for ax, (i, key, title) in zip(axs, CTRL3):
    lo, hi = RANGE[i]
    base = (random_rewards[i] - lo) / (hi - lo)
    ax.axhline(base, color=COL["rand"], lw=2, ls=":", label="random policy")
    ax.plot(T, (H[NAMES[i]][:, 0] - lo) / (hi - lo), color=COL[key], lw=2, label="off-policy (Horde)")
    y = (EPI[i][:, 1] - lo) / (hi - lo)
    ax.plot(EPI[i][:, 0] / 1e6, y, color=COL[key], lw=2, ls="-.", label="on-policy Sarsa(λ), episodic")
    s, final = trail(y), y[-8:].mean()
    j = int(np.argmax(s >= base + 0.9 * (final - base)))
    comp.append((f"{title} (on-policy episodic)", EPI[i][j, 0], final, base))
    ax.set(title=title, xlabel="training steps (millions)")
    ax.legend(frameon=False, fontsize=10)
axs[0].set_ylabel("average reward per step, scaled to [0, 1]")
fig.suptitle("Off-policy Horde vs episodic on-policy Sarsa(λ), learning off (cf. Horde Fig. 5)")
fig.tight_layout()
fig.savefig(f"{OUT}/fig5_episodic_vs_horde.png")
with open(f"{OUT}/steps_to_competence.txt", "w") as fh:
    fh.write("first step where the trailing 100k mean covers 90% of the gap from random to the final level\n")
    fh.write(f"{'demon':<38}{'steps':>12}{'final (0-1)':>13}{'random (0-1)':>14}\n")
    for label, steps, final, base in comp:
        fh.write(f"{label:<38}{steps:>12,.0f}{final:>13.3f}{base:>14.3f}\n")

# ---------- learned / random multiplier (off-policy Horde vs episodic on-policy)
fig, ax = plt.subplots(figsize=(8, 4.2))
for i, key, title in CTRL3:
    ax.plot(T, H[NAMES[i]][:, 2], color=COL[key], lw=2, label=f"{title} (off-policy)")
    ax.plot(EPI[i][:, 0] / 1e6, EPI[i][:, 3], color=COL[key], lw=2, ls="--", alpha=0.8, label=f"{title} (on-policy episodic)")
ax.axhline(1, color=COL["rand"], ls=":", lw=2)
ax.set(xlabel="training steps (millions)", ylabel="learned ÷ random", title="Control demons: multiplier over random")
ax.legend(frameon=False, fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/multiplier.png")

# ---------- prediction demons: mean |q-hat - G| over 25 visited states per eval
fig, ax = plt.subplots(figsize=(8, 4.2))
for n, c in zip(NAMES[:4], ("#141821", "#2C62A0", "#C97A12", "#8A8274")):
    ax.plot(T, H[n][:, 2], color=c, lw=1.2, alpha=0.35)
    e = H[n][:, 2]  # trailing 200k-step (8-eval) mean on top of the raw evals
    ax.plot(T, [e[max(0, j - 7) : j + 1].mean() for j in range(len(e))], color=c, lw=2.2, label=n)
ax.set(xlabel="training steps (millions)", ylabel="mean absolute error", title="Prediction demons: |prediction − actual return|")
ax.legend(frameon=False, fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/prediction_mae.png")


# ---------- final policies
def load_horde():
    h, ck = Horde(DEMONS), np.load(f"{RUNS}/horde/checkpoint.npz")
    h.theta = ck["theta"]
    return h, int(ck["t"])


horde, t_h = load_horde()
ck_on = np.load(f"{RUNS}/onpolicy_episodic_seek/checkpoint_onpolicy.npz")
th_on = ck_on["theta"]
q_on = lambda o: np.array([th_on[phi(o, a)].sum() for a in Action])

POL = {  # Fig 6 as in the paper: the demon's learned policy vs the behaviour policy it learned from
    "seek-light demon (target policy)": lambda o, r, p: int(np.argmax(horde.q(o)[SEEK])),
    "behaviour (sticky random)": lambda o, r, p: int(r.choice(N_ACTIONS, p=np.where(np.arange(N_ACTIONS) == p, STICK, 0) + (1 - STICK) / N_ACTIONS)),
}

# ---------- Fig 6 style: centre of the arena, facing away from the light, 8 minutes (960 steps)
away = np.arctan2(LIGHT[1] - ARENA / 2, LIGHT[0] - ARENA / 2) + np.pi
N_RUN, STEPS6 = 10, 960
light, paths = {}, {}
for name, pol in POL.items():
    L, P = np.zeros((N_RUN, STEPS6)), []
    for k in range(N_RUN):
        sim, r, prev = Robot(seed=100 + k), np.random.default_rng(100 + k), int(Action.STOP)
        sim.set_state((ARENA / 2, ARENA / 2, away, 0.0, 0.0, 0.0))
        o, xy = sim.obs(), []
        for s in range(STEPS6):
            prev = pol(o, r, prev)
            o = sim.step(Action(prev))
            L[k, s] = o[LIGHT_FL] + o[LIGHT_FR]
            xy.append((sim.x, sim.y))
        P.append(np.array(xy))
    light[name], paths[name] = L, P

fig = plt.figure(figsize=(13, 4.6))
ax = fig.add_subplot(1, 3, 1)
mins = np.arange(STEPS6) * 0.5 / 60
for name, c in zip(POL, (COL["seek"], "#7F93B0")):  # trailing 10-step mean (no zero padding at the ends)
    ax.plot(mins[9:], np.convolve(light[name].mean(0), np.ones(10) / 10, "valid"), color=c, lw=2, label=name)
ax.set(xlabel="time (minutes)", ylabel="front light  L_FL + L_FR", title=f"mean of {N_RUN} runs from the centre")
ax.legend(frameon=False, fontsize=9)
gx = np.linspace(0, ARENA, 300)
X, Y = np.meshgrid(gx, gx)
dark = np.zeros_like(X, bool)
for cx, cy, rad in OBSTACLES:  # same shadow rule as env.py
    d = np.stack([LIGHT[0] - X, LIGHT[1] - Y]); dist = np.hypot(*d); u = d / (dist + 1e-9)
    c = np.stack([cx - X, cy - Y]); s = (c * u).sum(0)
    dark |= (np.hypot(*(c - s * u)) < rad) & (s > 0) & (s < dist)
# brightness the sensors see (env.py: softened inverse square, 0 in a pillar's shadow), sqrt so the falloff shows
B = 0.25 / (0.25 + (X - LIGHT[0]) ** 2 + (Y - LIGHT[1]) ** 2) * ~dark
glow = matplotlib.colors.LinearSegmentedColormap.from_list("glow", ["#1E2230", "#4A3F3A", "#B9822E", "#F6D58A", "#FFF8E6"])
for k, (name, c) in enumerate(zip(POL, (COL["seek"], "#7F93B0"))):
    a = fig.add_subplot(1, 3, k + 2)
    a.imshow(np.sqrt(B), origin="lower", extent=(0, ARENA, 0, ARENA), cmap=glow, vmin=0, vmax=1, interpolation="bilinear")
    a.contour(X, Y, B, levels=(0.05, 0.1, 0.2, 0.4), colors="#FFF8E6", linewidths=0.6, alpha=0.35)
    for cx, cy, rad in OBSTACLES:
        a.add_patch(plt.Circle((cx, cy), rad, color="#2C62A0", ec="#9DB8DA", lw=1.2, zorder=3))
    for p in paths[name]:
        a.plot(p[:, 0], p[:, 1], color=c, lw=0.9, alpha=0.7)
    a.plot(*LIGHT, "*", color="#E3A23B", ms=18, mec="#141821")
    a.plot(ARENA / 2, ARENA / 2, "o", color="#141821", ms=6)
    a.set(xlim=(0, ARENA), ylim=(0, ARENA), xticks=[], yticks=[], title=name, aspect="equal")
fig.suptitle("Light seeking from random behaviour (cf. Horde Fig. 6): start at centre facing away from the light, 8 min")
fig.tight_layout()
fig.savefig(f"{OUT}/fig6_light_seeking.png")

# ---------- Fig 2 style: steps-to-obstacle prediction vs actual return along a behaviour trajectory
sim, r, prev = Robot(seed=7), np.random.default_rng(7), int(Action.STOP)
sim.set_state((1.0, 1.0, 0.0, 0.0, 0.0, 0.0))
pred, act = [], []
for s in range(300):
    o, st = sim.obs(), sim.get_state()
    pred.append(horde.q(o)[STO][Action.FORWARD])
    ro, n = Robot(seed=s), 0  # counterfactual: keep driving forward from here
    ro.set_state(st)
    while n < 100:
        oo, n = ro.step(Action.FORWARD), n + 1
        if DEMONS[STO]["gamma"](oo) == 0.0:
            break
    act.append(n)
    prev = POL["behaviour (sticky random)"](o, r, prev)
    sim.step(Action(prev))
fig, ax = plt.subplots(figsize=(10, 3.6))
ax.plot(np.arange(300) * 0.5, act, color="#141821", lw=1.2, label="actual steps until obstacle (if driving forward)")
ax.plot(np.arange(300) * 0.5, pred, color=COL["seek"], lw=2.2, label="steps-to-obstacle prediction")
ax.set(xlabel="time (s) along a sticky-random trajectory", ylabel="steps", title="Predicting time-to-obstacle (cf. Horde Fig. 2)")
ax.legend(frameon=False, fontsize=10)
fig.tight_layout()
fig.savefig(f"{OUT}/fig2_time_to_obstacle.png")

# ---------- combine.py variants on EVAL_STARTS (same 100 starts x 250 steps)
rank = lambda v: (v - v.mean()) / (v.std() + 1e-9)  # per-state z-score across actions
VAR = [("random", None), ("seek alone", lambda q: q[SEEK]), ("on-policy seek (episodic)", "on"), ("avoid alone", lambda q: q[AVOID])]
VAR += [(f"sum  w={w}", (lambda w: lambda q: q[SEEK] + w * q[AVOID])(w)) for w in (0.1, 0.25, 0.5, 1.0)]
VAR += [(f"z-sum w={w}", (lambda w: lambda q: rank(q[SEEK]) + w * rank(q[AVOID]))(w)) for w in (0.5, 1.0)]
VAR += [(f"reflex: steps-to-obstacle < {k}", (lambda k: lambda q: q[AVOID] if q[STO][Action.FORWARD] < k else q[SEEK])(k)) for k in (2, 3, 5)]
VAR += [(f"reflex: bump-soon > {b}", (lambda b: lambda q: q[AVOID] if q[BSOON][Action.FORWARD] > b else q[SEEK])(b)) for b in (1.0, 2.0)]
res = []
for name, f in VAR:
    sim, r = Robot(seed=0), np.random.default_rng(0)
    tot = np.zeros(6)  # seek r, avoid r, bumps, pillar bumps, ever<0.5, ended<0.5
    for st in EVAL_STARTS:
        sim.set_state(st)
        o, near = sim.obs(), False
        for _ in range(250):
            if f is None:
                a = int(r.integers(N_ACTIONS))
            elif f == "on":
                a = int(np.argmax(q_on(o)))
            else:
                a = int(np.argmax(f(horde.q(o))))
            o = sim.step(Action(a))
            gap = np.hypot(sim.x - OBSTACLES[:, 0], sim.y - OBSTACLES[:, 1]) - OBSTACLES[:, 2]
            near |= np.hypot(sim.x - LIGHT[0], sim.y - LIGHT[1]) < 0.5
            tot[:4] += (DEMONS[SEEK]["r"](o), DEMONS[AVOID]["r"](o), o[BUMP], o[BUMP] * (gap.min() < 0.2))
        tot[4:] += (near, np.hypot(sim.x - LIGHT[0], sim.y - LIGHT[1]) < 0.5)
    n = len(EVAL_STARTS) * 250
    res.append((name, tot[0] / n, tot[1] / n, 1000 * tot[2] / n, 1000 * tot[3] / n, 100 * tot[4] / len(EVAL_STARTS), 100 * tot[5] / len(EVAL_STARTS)))
with open(f"{OUT}/combine_variants.txt", "w") as fh:
    fh.write(f"Horde checkpoint {t_h:,} | episodic on-policy checkpoint {int(ck_on['t']):,} | 100 fixed starts x 250 steps\n")
    fh.write(f"{'policy':<34}{'seek r':>9}{'avoid r':>9}{'bumps/1k':>10}{'pillar/1k':>11}{'ever<0.5m':>11}{'end<0.5m':>10}\n")
    for row in res:
        fh.write(f"{row[0]:<34}{row[1]:>9.3f}{row[2]:>9.3f}{row[3]:>10.1f}{row[4]:>11.1f}{row[5]:>10.0f}%{row[6]:>9.0f}%\n")
fig, ax = plt.subplots(figsize=(9, 5))
for name, s, _, b, *_ in res:
    c = COL["rand"] if name == "random" else COL["on"] if "on-policy" in name else COL["avoid"] if "avoid" in name else "#7B4FA0" if "reflex" in name else COL["seek"]
    ax.scatter(max(b, 0.3), s, s=70, color=c, marker="s" if "z-sum" in name else "o", zorder=3)
    ax.annotate(name.replace("reflex: ", ""), (max(b, 0.3), s), fontsize=8, xytext=(5, 3), textcoords="offset points")
ax.set(xscale="log", xlabel="bumps per 1,000 steps (log)", ylabel="seek-light reward per step", title="Combining seek-light and avoid (combine.py variants)")
fig.tight_layout()
fig.savefig(f"{OUT}/combine_tradeoff.png")
print(open(f"{OUT}/combine_variants.txt").read())
print(open(f"{OUT}/steps_to_competence.txt").read())
