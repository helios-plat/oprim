"""Tests for oprim.risk.net_exposure_clip."""

import pytest

from oprim.risk.net_exposure_clip import net_exposure_clip


def test_no_correlated_positions_passes_through():
    result = net_exposure_clip(proposed=0.1, correlated_positions=[], max_net_exposure=0.25)
    assert result == {
        "allowed_size": 0.1,
        "net_exposure": 0.1,
        "was_clipped": False,
        "was_rejected": False,
    }


def test_within_bound_with_correlated_position():
    # net = 0.1 + 0.1*0.85 = 0.185 <= 0.25
    result = net_exposure_clip(
        proposed=0.1, correlated_positions=[(0.1, 0.85)], max_net_exposure=0.25
    )
    assert result["was_clipped"] is False
    assert result["net_exposure"] == pytest.approx(0.185)
    assert result["allowed_size"] == 0.1


def test_exceeds_bound_clips_down_to_exact_max():
    # already_used = 0.2*0.85 = 0.17; max_allowed = 0.25-0.17 = 0.08
    result = net_exposure_clip(
        proposed=0.15, correlated_positions=[(0.2, 0.85)], max_net_exposure=0.25
    )
    assert result["was_clipped"] is True
    assert result["was_rejected"] is False
    assert result["allowed_size"] == pytest.approx(0.08)
    assert result["net_exposure"] == pytest.approx(0.25)


def test_already_used_alone_exceeds_bound_rejects():
    # already_used = 0.5*0.85 = 0.425 > 0.25 -> max_allowed clipped to 0
    result = net_exposure_clip(
        proposed=0.1, correlated_positions=[(0.5, 0.85)], max_net_exposure=0.25
    )
    assert result["allowed_size"] == 0.0
    assert result["was_rejected"] is True


def test_multiple_correlated_positions_summed():
    result = net_exposure_clip(
        proposed=0.05,
        correlated_positions=[(0.1, 0.85), (0.05, 0.5)],
        max_net_exposure=1.0,
    )
    expected_net = 0.05 + 0.1 * 0.85 + 0.05 * 0.5
    assert result["net_exposure"] == pytest.approx(expected_net)
    assert result["was_clipped"] is False


def test_negative_proposed_clips_toward_negative_bound():
    result = net_exposure_clip(
        proposed=-0.2, correlated_positions=[(-0.1, 0.85)], max_net_exposure=0.25
    )
    assert result["was_clipped"] is True
    assert result["allowed_size"] < 0
    assert result["allowed_size"] == pytest.approx(-(0.25 - 0.1 * 0.85))


def test_max_net_exposure_zero_raises():
    with pytest.raises(ValueError):
        net_exposure_clip(proposed=0.1, correlated_positions=[], max_net_exposure=0.0)


def test_max_net_exposure_negative_raises():
    with pytest.raises(ValueError):
        net_exposure_clip(proposed=0.1, correlated_positions=[], max_net_exposure=-0.1)
