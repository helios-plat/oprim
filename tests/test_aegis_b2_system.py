"""B2 System focused wrapper tests — system_cpu_usage / system_ram_usage / system_load_avg."""

from __future__ import annotations

import pytest

from oprim import system_cpu_usage, system_ram_usage, system_load_avg


def test_system_cpu_usage_returns_float():
    result = system_cpu_usage()
    assert isinstance(result, float)


def test_system_cpu_usage_in_range():
    result = system_cpu_usage()
    assert 0.0 <= result <= 100.0


def test_system_cpu_usage_is_rounded():
    result = system_cpu_usage()
    # Should be rounded to 2 decimal places
    assert result == round(result, 2)


def test_system_ram_usage_returns_dict():
    result = system_ram_usage()
    assert isinstance(result, dict)


def test_system_ram_usage_has_required_keys():
    result = system_ram_usage()
    for key in ("total_bytes", "used_bytes", "available_bytes", "used_percent"):
        assert key in result, f"Missing key: {key}"


def test_system_ram_usage_positive_values():
    result = system_ram_usage()
    assert result["total_bytes"] > 0
    assert result["available_bytes"] >= 0
    assert result["used_bytes"] >= 0


def test_system_ram_usage_percent_in_range():
    result = system_ram_usage()
    assert 0.0 <= result["used_percent"] <= 100.0


def test_system_ram_usage_total_consistent():
    result = system_ram_usage()
    # used + available should be <= total (some may be kernel reserved)
    assert result["used_bytes"] + result["available_bytes"] <= result["total_bytes"] * 1.05


def test_system_load_avg_returns_dict():
    result = system_load_avg()
    assert isinstance(result, dict)


def test_system_load_avg_has_keys():
    result = system_load_avg()
    assert "load_1m" in result
    assert "load_5m" in result
    assert "load_15m" in result


def test_system_load_avg_values_are_floats():
    result = system_load_avg()
    for key in ("load_1m", "load_5m", "load_15m"):
        assert isinstance(result[key], float)


def test_system_load_avg_non_negative():
    result = system_load_avg()
    for key in ("load_1m", "load_5m", "load_15m"):
        assert result[key] >= 0.0
