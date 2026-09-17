"""Live training view. Entirely optional -- set VIZ = False in main.py to skip it.

Three panels:
  left   -- where the behaviour policy has taken the robot. A coverage check:
            demons can only be right where the robot has actually been.
  middle -- raw TD error per demon, sampled at each redraw, symlog scale.
            The divergence tripwire; a line running away means that demon is
            blowing up. NOTE this is one instantaneous sample every `every`
            steps, and delta is signed, so expect it to look noisy.
  right  -- evaluation quality as it accumulates. MAE for prediction demons,
            ratio-over-random for control demons.
"""

import matplotlib.pyplot as plt
import numpy as np

from demons import DEMONS
from env import ARENA, LIGHT


class Live:
    def __init__(self, every=2000, trail=1500):
        plt.ion()
        self.every, self.trail = every, trail
        self.xs, self.ys, self.ts, self.es = [], [], [], []
        self.fig, (a1, a2, self.a3) = plt.subplots(1, 3, figsize=(15, 4.3))

        a1.set(xlim=(0, ARENA), ylim=(0, ARENA), title="robot trail", aspect="equal")
        a1.plot(*LIGHT, marker="*", ms=18, color="gold", zorder=3)
        (self.path,) = a1.plot([], [], lw=0.5, alpha=0.6, color="steelblue")
        (self.dot,) = a1.plot([], [], "o", ms=8, color="crimson", zorder=4)

        a2.set(title="TD error (delta)", xlabel="step")
        a2.set_yscale("symlog", linthresh=1e-3)  # symlog: delta is signed
        self.dlines = [a2.plot([], [], lw=1, label=d["name"])[0] for d in DEMONS]
        a2.legend(fontsize=5, ncol=2)

        self.a3.set(title="eval: MAE (pred) / greedy/random (ctrl)", xlabel="step")
        self.elines = {
            d["name"]: self.a3.plot([], [], marker="o", ms=3, label=d["name"])[0]
            for d in DEMONS
        }
        self.a3.legend(fontsize=5, ncol=2)
        self.fig.tight_layout()

    def update(self, t, state, delta):
        """Call every step. Cheap: only redraws once every `every` steps."""
        if t % 20 == 0:
            self.xs.append(state[0])
            self.ys.append(state[1])
            self.xs, self.ys = self.xs[-self.trail :], self.ys[-self.trail :]
        if t % self.every:
            return
        self.ts.append(t)
        self.es.append(np.asarray(delta).copy())
        e = np.array(self.es)
        self.path.set_data(self.xs, self.ys)
        self.dot.set_data([state[0]], [state[1]])
        for i, ln in enumerate(self.dlines):
            ln.set_data(self.ts, e[:, i])
        for ax in (self.fig.axes[1],):
            ax.relim()
            ax.autoscale_view()
        self.fig.canvas.draw_idle()
        plt.pause(0.001)

    def record(self, t, rows):
        """Call at each evaluation checkpoint with the rows evaluate() returned."""
        for name, kind, x, y, z in rows:
            ln = self.elines[name]
            xs, ys = ln.get_data()
            ln.set_data(np.append(xs, t), np.append(ys, z if kind == "pred" else z))
        self.a3.relim()
        self.a3.autoscale_view()
        self.fig.canvas.draw_idle()
        plt.pause(0.001)
