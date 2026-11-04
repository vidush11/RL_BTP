# RL_BTP (copied, unedited)

`env.py`, `features.py`, `horde.py`, `demons.py` are byte-for-byte copies of the
RL_BTP repository (Vidush Jindal; BTP under Prof. Sanjit Kaul). `SHA256SUMS`
records their hashes, and `tests/test_horde_adapter.py` checks them. They are
used through `agents/horde/adapter.py`, which maps this simulator's
LiDAR + info onto RL_BTP's 9-sensor vector.

Do not edit these files. Two side effects are relevant:
* `features.py` calls `np.random.seed(0)` at import time, which reseeds
  NumPy's global RNG. This project uses `numpy.random.Generator`, so nothing
  here is affected.
* The module names are `env`, `features`, `horde`, `demons`; the adapter puts
  this directory on `sys.path` to import them.
