"""H2: with existing configs the simulator must be bit-identical to the
original package. Golden data was recorded on the baseline commit by
tests/golden/make_golden.py (obs, reward, terminated, truncated, full info)."""
import pickle
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "tests" / "golden"))
from make_golden import record  # noqa: E402

GOLDEN = pickle.load(open(ROOT / "tests" / "golden" / "golden_v1.pkl", "rb"))


def _same(a, b):
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and a.dtype == b.dtype \
            and a.shape == b.shape and np.array_equal(a, b)
    return type(a) is type(b) and a == b


@pytest.mark.parametrize("name", list(GOLDEN["cases"]))
def test_bit_identical_to_original(name):
    want = GOLDEN["golden"][name]
    got = record(GOLDEN["cases"][name])
    assert got["obs_space"] == want["obs_space"] and got["n_actions"] == want["n_actions"]
    assert len(got["episodes"]) == len(want["episodes"])
    for ge, we in zip(got["episodes"], want["episodes"]):
        assert len(ge) == len(we)
        for gs, ws in zip(ge, we):
            assert set(gs) == set(ws)
            for k in ws:
                if k == "info":
                    assert set(gs["info"]) == set(ws["info"]), f"info keys changed: {set(gs['info']) ^ set(ws['info'])}"
                    for ik in ws["info"]:
                        assert _same(gs["info"][ik], ws["info"][ik]), f"{name}: info[{ik}] differs"
                else:
                    assert _same(gs[k], ws[k]), f"{name}: {k} differs"
