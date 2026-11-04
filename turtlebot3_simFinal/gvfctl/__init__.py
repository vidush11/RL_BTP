"""gvfctl: GVF-based predictive controllers vs DQN on the headless TurtleBot3 sim.

Part 1: safety GVF + steering controller (fixed speed, 5 actions).
Part 2: safety GVF with speed-aware (headway) cumulant + future linear/angular
velocity GVFs + speed controller (15 actions = 3 speeds x 5 turn rates).
Pure NumPy. New files only: nothing in simulator/ or the original agents changes.
See GVFCTL_REPORT_README.md / results/<timestamp>/REPORT.md.
"""
