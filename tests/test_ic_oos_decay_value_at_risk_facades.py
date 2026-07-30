"""Tests for the oprim.ic_oos_decay / oprim.value_at_risk facade submodules.

These exist purely so Tide can do `from oprim.ic_oos_decay import ic_oos_decay`
/ `from oprim.value_at_risk import value_at_risk` (submodule-path import) --
the underlying implementations already live in oprim._signal_analysis /
oprim._finance and were already reachable via `from oprim import ic_oos_decay`
(package-level attribute, via the lazy __getattr__ element map). Same gap
oprim/video_generate.py was created to close for avatar/video facades.
"""

from __future__ import annotations

import pandas as pd

from oprim.ic_oos_decay import ic_oos_decay
from oprim.value_at_risk import value_at_risk


def test_ic_oos_decay_submodule_import_works() -> None:
    result = ic_oos_decay(ic_series=[0.1, 0.09, 0.08, 0.05, 0.03], oos_start_idx=2)
    assert "ic_mean" in result
    assert "ic_decay_slope" in result
    assert "stability" in result


def test_value_at_risk_submodule_import_works() -> None:
    returns = pd.Series([0.01, -0.02, 0.015, -0.01, 0.03, -0.025, 0.02, -0.015, 0.01, 0.005] * 4)
    result = value_at_risk(returns)
    assert "var" in result or "es" in result
