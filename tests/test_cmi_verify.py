"""Tests for oprim.cmi_verify — ≥10 cases."""

from __future__ import annotations

import pytest
from oprim.cmi_verify import cmi_verify


# 1. Empty treatment group
def test_empty_treatment():
    result = cmi_verify(treatment=[], control=[1.0, 2.0, 3.0])
    assert result["significant"] is False
    assert result["p_value"] == 1.0
    assert result["effect_size"] == 0.0
    assert result["n_treatment"] == 0
    assert result["n_control"] == 3
    assert result["causal_confidence"] == "none"


# 2. Empty control group
def test_empty_control():
    result = cmi_verify(treatment=[1.0, 2.0], control=[])
    assert result["significant"] is False
    assert result["p_value"] == 1.0
    assert result["n_control"] == 0
    assert result["causal_confidence"] == "none"


# 3. Both groups empty
def test_both_empty():
    result = cmi_verify(treatment=[], control=[])
    assert result["significant"] is False
    assert result["mean_diff"] == 0.0
    assert result["causal_confidence"] == "none"


# 4. Identical groups → no effect
def test_identical_groups():
    data = [1.0, 2.0, 3.0, 4.0, 5.0]
    result = cmi_verify(treatment=data, control=data)
    assert result["mean_diff"] == pytest.approx(0.0, abs=1e-10)
    assert result["significant"] is False
    assert result["causal_confidence"] == "none"


# 5. Clear strong effect (large separation, large n)
def test_strong_effect():
    import numpy as np

    rng = np.random.default_rng(42)
    treatment = list(rng.normal(10.0, 1.0, 100))
    control = list(rng.normal(0.0, 1.0, 100))
    result = cmi_verify(treatment=treatment, control=control)
    assert result["significant"] is True
    assert result["causal_confidence"] == "strong"
    assert result["effect_size"] > 0.8
    assert result["mean_diff"] > 0


# 6. No real effect (same distribution, moderate n)
def test_no_effect_same_distribution():
    import numpy as np

    rng = np.random.default_rng(7)
    data1 = list(rng.normal(5.0, 2.0, 200))
    data2 = list(rng.normal(5.0, 2.0, 200))
    result = cmi_verify(treatment=data1, control=data2)
    # p_value should be > 0.05 (not significant)
    assert result["p_value"] > 0.05
    assert result["significant"] is False


# 7. effect_size range — negative treatment effect
def test_negative_effect_size():
    import numpy as np

    rng = np.random.default_rng(0)
    treatment = list(rng.normal(-5.0, 1.0, 50))
    control = list(rng.normal(5.0, 1.0, 50))
    result = cmi_verify(treatment=treatment, control=control)
    assert result["effect_size"] < 0
    assert result["mean_diff"] < 0


# 8. p_value in [0, 1]
def test_p_value_bounds():
    result = cmi_verify(
        treatment=[1.0, 2.0, 3.0, 4.0, 5.0],
        control=[10.0, 11.0, 12.0, 13.0, 14.0],
    )
    assert 0.0 <= result["p_value"] <= 1.0


# 9. causal_confidence "moderate" for medium effect
def test_moderate_causal_confidence():
    import numpy as np

    # Effect size ~0.6 with enough samples
    rng = np.random.default_rng(99)
    treatment = list(rng.normal(3.0, 1.0, 300))
    control = list(rng.normal(2.4, 1.0, 300))
    result = cmi_verify(treatment=treatment, control=control)
    assert result["significant"] is True
    # effect size should be around 0.6, so moderate
    assert result["causal_confidence"] in ("moderate", "strong")


# 10. n values returned correctly
def test_n_values():
    result = cmi_verify(
        treatment=[1.0, 2.0, 3.0],
        control=[4.0, 5.0],
    )
    assert result["n_treatment"] == 3
    assert result["n_control"] == 2


# 11. mean_diff sign matches direction
def test_mean_diff_sign():
    result = cmi_verify(
        treatment=[10.0, 10.0, 10.0],
        control=[5.0, 5.0, 5.0],
    )
    assert result["mean_diff"] == pytest.approx(5.0)
    assert result["mean_diff"] > 0


# 12. large samples, significant strong effect
def test_large_samples():
    import numpy as np

    rng = np.random.default_rng(123)
    treatment = list(rng.normal(8.0, 1.0, 500))
    control = list(rng.normal(0.0, 1.0, 500))
    result = cmi_verify(treatment=treatment, control=control)
    assert result["significant"] is True
    assert result["causal_confidence"] == "strong"
    assert result["n_treatment"] == 500
    assert result["n_control"] == 500


# 13. alpha parameter respected
def test_alpha_parameter():
    # With very strict alpha (0.001), mild effect is not significant
    import numpy as np

    rng = np.random.default_rng(55)
    treatment = list(rng.normal(5.1, 2.0, 30))
    control = list(rng.normal(5.0, 2.0, 30))
    result_strict = cmi_verify(treatment=treatment, control=control, alpha=0.001)
    result_loose = cmi_verify(treatment=treatment, control=control, alpha=0.99)
    # loose alpha should typically be significant
    assert result_loose["significant"] is True
    # strict alpha should not be significant for near-zero effect
    assert result_strict["significant"] is False


# 14. single-element groups still return valid dict
def test_single_element_groups():
    result = cmi_verify(treatment=[5.0], control=[3.0])
    assert "effect_size" in result
    assert "p_value" in result
    assert "causal_confidence" in result
    # pooled_std falls back to 1.0 for single-element, so effect_size = mean_diff
    assert result["mean_diff"] == pytest.approx(2.0)
