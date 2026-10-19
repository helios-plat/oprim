"""Tests for oprim.usb_device_list."""

from __future__ import annotations

import shutil

import pytest

from oprim import usb_device_list
from oprim._exceptions import OprimError
from oprim._usb_device_list import UsbDeviceList, _parse_line

_FAKE = """Bus 001 Device 001: ID 1d6b:0002 Linux Foundation 2.0 root hub
Bus 001 Device 002: ID 05e3:0608 Genesys Logic, Inc. Hub
Bus 001 Device 004: ID 0781:5567 SanDisk Corp. Cruzer Blade
Bus 002 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub
"""


@pytest.fixture
def fake_lsusb(monkeypatch):
    monkeypatch.setattr("oprim._usb_device_list._run_lsusb", lambda: _FAKE)


class TestParseLine:
    def test_parses_fields(self):
        d = _parse_line("Bus 001 Device 004: ID 0781:5567 SanDisk Corp. Cruzer Blade")
        assert d is not None
        assert d.bus == 1 and d.device == 4
        assert d.vendor_id == "0781" and d.product_id == "5567"
        assert d.description == "SanDisk Corp. Cruzer Blade"
        assert d.is_root_hub is False

    def test_root_hub_flag(self):
        d = _parse_line("Bus 002 Device 001: ID 1d6b:0003 Linux Foundation 3.0 root hub")
        assert d is not None and d.is_root_hub is True

    def test_junk_line_returns_none(self):
        assert _parse_line("not a lsblk line") is None
        assert _parse_line("") is None

    def test_hex_lowercased(self):
        d = _parse_line("Bus 001 Device 003: ID 048D:5702 Foo")
        assert d is not None and d.vendor_id == "048d" and d.product_id == "5702"


class TestUsbDeviceList:
    def test_root_hubs_excluded_by_default(self, fake_lsusb):
        r = usb_device_list()
        assert isinstance(r, UsbDeviceList)
        descs = [d.description for d in r.devices]
        assert "Linux Foundation 2.0 root hub" not in descs
        assert r.count == 2  # Hub + SanDisk

    def test_include_root_hubs(self, fake_lsusb):
        r = usb_device_list(include_root_hubs=True)
        assert r.count == 4

    def test_parse_injected_output_no_subprocess(self, monkeypatch):
        def _boom():
            raise AssertionError("_run_lsusb should not be called when lsusb_output given")

        monkeypatch.setattr("oprim._usb_device_list._run_lsusb", _boom)
        r = usb_device_list(lsusb_output=_FAKE)
        assert r.count == 2  # root hubs 默认剔除

    def test_missing_lsusb_raises(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda _: None)
        from oprim import _usb_device_list as m

        with pytest.raises(OprimError):
            m._run_lsusb()


@pytest.mark.skipif(shutil.which("lsusb") is None, reason="lsusb not installed")
class TestRealLsusb:
    def test_smoke(self):
        r = usb_device_list(include_root_hubs=True)
        assert isinstance(r, UsbDeviceList)
        assert r.count == len(r.devices)
