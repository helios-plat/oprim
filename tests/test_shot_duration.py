"""Tests for _shot_duration_defaults constants."""
from __future__ import annotations

from oprim._shot_duration_defaults import SHOT_DURATION_DEFAULTS, TRANSITION_RULES


class TestShotDurationDefaults:

    def test_closeup_is_single_float(self):
        assert SHOT_DURATION_DEFAULTS["closeup"] == 5.0

    def test_action_medium_is_range_tuple(self):
        val = SHOT_DURATION_DEFAULTS["action_medium"]
        assert isinstance(val, tuple)
        assert val[0] < val[1]

    def test_info_is_range_tuple(self):
        val = SHOT_DURATION_DEFAULTS["info"]
        assert isinstance(val, tuple)
        assert val[0] == 10.0 and val[1] == 15.0

    def test_all_expected_keys_present(self):
        assert "closeup" in SHOT_DURATION_DEFAULTS
        assert "action_medium" in SHOT_DURATION_DEFAULTS
        assert "info" in SHOT_DURATION_DEFAULTS

    def test_transition_rules_same_scene_near_is_hard(self):
        assert TRANSITION_RULES["same_scene_near"] == "hard"

    def test_transition_rules_cross_scene_is_dissolve(self):
        assert TRANSITION_RULES["cross_scene"] == "dissolve"

    def test_transition_rules_emotion_shift_is_flash(self):
        assert TRANSITION_RULES["emotion_shift"] == "flash"

    def test_all_transition_keys_present(self):
        for key in ("same_scene_near", "cross_scene", "emotion_shift"):
            assert key in TRANSITION_RULES
