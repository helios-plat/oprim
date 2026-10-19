"""Tests for oprim.disk_smart_probe."""

from __future__ import annotations

import json
import shutil

import pytest

from oprim import disk_smart_probe
from oprim._disk_smart_probe import SmartHealth
from oprim._exceptions import OprimError, OprimNotFoundError

# 仿 smartctl -j 的 ATA(HDD)输出
_FAKE_ATA = {
    "device": {"protocol": "ATA", "type": "sat"},
    "model_name": "WDC WD40EFRX",
    "serial_number": "WD-XYZ",
    "smart_status": {"passed": True},
    "temperature": {"current": 38},
    "ata_smart_attributes": {
        "table": [
            {"id": 5, "name": "Reallocated_Sector_Ct", "raw": {"value": 0}},
            {"id": 9, "name": "Power_On_Hours", "raw": {"value": 26280}},
            {"id": 12, "name": "Power_Cycle_Count", "raw": {"value": 42}},
            {"id": 194, "name": "Temperature_Celsius", "raw": {"value": 38}},
            {"id": 197, "name": "Current_Pending_Sector", "raw": {"value": 0}},
        ]
    },
}

# 仿 NVMe 输出
_FAKE_NVME = {
    "device": {"protocol": "NVMe", "type": "nvme"},
    "model_name": "Samsung SSD 980",
    "serial_number": "S1",
    "smart_status": {"passed": True},
    "nvme_smart_health_information_log": {
        "temperature": 45,
        "power_on_hours": 1200,
        "power_cycles": 88,
        "percentage_used": 3,
    },
}


class TestValidation:
    def test_empty_device(self):
        with pytest.raises(OprimNotFoundError):
            disk_smart_probe(device="")

    def test_non_dev_path(self):
        with pytest.raises(OprimNotFoundError):
            disk_smart_probe(device="sda")


class TestAta:
    @pytest.fixture
    def fake(self, monkeypatch):
        monkeypatch.setattr("oprim._disk_smart_probe._run_smartctl", lambda device: _FAKE_ATA)

    def test_health_and_attrs(self, fake):
        r = disk_smart_probe(device="/dev/sda")
        assert isinstance(r, SmartHealth)
        assert r.available is True
        assert r.passed is True
        assert r.protocol == "ATA"
        assert r.model == "WDC WD40EFRX"
        assert r.temperature_celsius == 38
        assert r.power_on_hours == 26280
        assert r.power_cycles == 42
        assert r.reallocated_sectors == 0
        assert r.pending_sectors == 0
        assert len(r.attributes) == 5


class TestNvme:
    @pytest.fixture
    def fake(self, monkeypatch):
        monkeypatch.setattr("oprim._disk_smart_probe._run_smartctl", lambda device: _FAKE_NVME)

    def test_nvme_extract(self, fake):
        r = disk_smart_probe(device="/dev/nvme0n1")
        assert r.protocol == "NVMe"
        assert r.temperature_celsius == 45
        assert r.power_on_hours == 1200
        assert r.power_cycles == 88
        assert r.available is True


class TestInjected:
    def test_parse_injected_json_no_subprocess(self, monkeypatch):
        def _boom(device):
            raise AssertionError("_run_smartctl should not be called when smartctl_json given")

        monkeypatch.setattr("oprim._disk_smart_probe._run_smartctl", _boom)
        r = disk_smart_probe(device="/dev/sda", smartctl_json=json.dumps(_FAKE_ATA))
        assert r.passed is True and r.power_on_hours == 26280

    def test_injected_bad_json_raises(self):
        with pytest.raises(OprimError):
            disk_smart_probe(device="/dev/sda", smartctl_json="{bad")


class TestDegrade:
    def test_missing_smartctl_raises(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda _: None)
        from oprim import _disk_smart_probe as m

        with pytest.raises(OprimError):
            m._run_smartctl("/dev/sda")
