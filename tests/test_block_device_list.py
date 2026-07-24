"""Tests for oprim.block_device_list."""

from __future__ import annotations

import json
import shutil

import pytest

from oprim import block_device_list
from oprim._block_device_list import (
    BlockDevice,
    BlockDeviceList,
    _parse_node,
    _to_bool,
)
from oprim._exceptions import OprimError

# 一个仿真的 lsblk -J 输出:一块 NVMe SSD(2 分区)+ 一块 USB HDD + 一个 loop.
_FAKE = {
    "blockdevices": [
        {
            "name": "nvme0n1",
            "kname": "nvme0n1",
            "path": "/dev/nvme0n1",
            "size": 512110190592,
            "type": "disk",
            "fstype": None,
            "mountpoint": None,
            "model": "Samsung SSD 980",
            "serial": "S1",
            "vendor": None,
            "label": None,
            "uuid": None,
            "rota": False,
            "tran": "nvme",
            "hotplug": False,
            "rm": False,
            "ro": False,
            "state": "live",
            "children": [
                {
                    "name": "nvme0n1p1",
                    "type": "part",
                    "size": 536870912,
                    "fstype": "vfat",
                    "mountpoint": "/boot/efi",
                    "rota": False,
                },
                {
                    "name": "nvme0n1p2",
                    "type": "part",
                    "size": 511560000000,
                    "fstype": "ext4",
                    "mountpoint": "/",
                    "rota": False,
                },
            ],
        },
        {
            "name": "sda",
            "kname": "sda",
            "path": "/dev/sda",
            "size": 4000787030016,
            "type": "disk",
            "fstype": None,
            "mountpoint": None,
            "model": "Elements 25A3",
            "serial": "S2",
            "tran": "usb",
            "rota": True,
            "hotplug": True,
            "rm": True,
            "ro": False,
            "state": "running",
            "children": [
                {
                    "name": "sda1",
                    "type": "part",
                    "size": 4000000000000,
                    "fstype": "ntfs",
                    "mountpoints": ["/mnt/backup"],
                    "rota": True,
                },
            ],
        },
        {
            "name": "loop0",
            "type": "loop",
            "size": 70123520,
            "fstype": "squashfs",
            "mountpoint": "/snap/core/1",
            "rota": False,
        },
    ]
}


@pytest.fixture
def fake_lsblk(monkeypatch):
    monkeypatch.setattr("oprim._block_device_list._run_lsblk", lambda: _FAKE)


class TestToBool:
    def test_variants(self):
        assert _to_bool(True) is True
        assert _to_bool(False) is False
        assert _to_bool(1) is True
        assert _to_bool(0) is False
        assert _to_bool("1") is True
        assert _to_bool("0") is False
        assert _to_bool("true") is True
        assert _to_bool("no") is False
        assert _to_bool(None) is None
        assert _to_bool("weird") is None


class TestParseNode:
    def test_mountpoints_list_fallback(self):
        # 新版 lsblk 用 mountpoints 列表而非 mountpoint 标量
        dev = _parse_node({"name": "sda1", "type": "part", "mountpoints": [None, "/mnt/x"]})
        assert dev.mountpoint == "/mnt/x"

    def test_empty_strings_become_none(self):
        dev = _parse_node({"name": "x", "model": "", "serial": "", "label": ""})
        assert dev.model is None and dev.serial is None and dev.label is None


class TestBlockDeviceList:
    def test_returns_model(self, fake_lsblk):
        r = block_device_list()
        assert isinstance(r, BlockDeviceList)
        assert all(isinstance(d, BlockDevice) for d in r.devices)

    def test_loop_excluded_by_default(self, fake_lsblk):
        r = block_device_list()
        names = [d.name for d in r.devices]
        assert "loop0" not in names
        assert names == ["nvme0n1", "sda"]
        assert r.count == 2

    def test_include_loop(self, fake_lsblk):
        r = block_device_list(include_loop=True)
        assert "loop0" in [d.name for d in r.devices]
        assert r.count == 3

    def test_disks_only(self, fake_lsblk):
        r = block_device_list(disks_only=True)
        assert {d.type for d in r.devices} == {"disk"}
        assert r.count == 2

    def test_children_and_fields_parsed(self, fake_lsblk):
        r = block_device_list()
        nvme = r.devices[0]
        assert nvme.transport == "nvme"
        assert nvme.rotational is False  # SSD
        assert nvme.size_bytes == 512110190592
        assert [c.mountpoint for c in nvme.children] == ["/boot/efi", "/"]
        sda = r.devices[1]
        assert sda.transport == "usb" and sda.removable is True and sda.rotational is True
        assert sda.children[0].mountpoint == "/mnt/backup"  # mountpoints list form

    def test_parse_injected_json_no_subprocess(self, monkeypatch):
        # 传 lsblk_json 时绝不应触碰 _run_lsblk(执行位置由调用方决定)
        def _boom():
            raise AssertionError("_run_lsblk should not be called when lsblk_json given")

        monkeypatch.setattr("oprim._block_device_list._run_lsblk", _boom)
        r = block_device_list(lsblk_json=json.dumps(_FAKE))
        assert r.count == 2
        assert r.devices[0].name == "nvme0n1"

    def test_injected_bad_json_raises(self):
        with pytest.raises(OprimError):
            block_device_list(lsblk_json="{not json")

    def test_missing_lsblk_raises(self, monkeypatch):
        monkeypatch.setattr(shutil, "which", lambda _: None)
        # 绕过 monkeypatched _run_lsblk,直接触发真实的 which 检查
        from oprim import _block_device_list as m

        with pytest.raises(OprimError):
            m._run_lsblk()


@pytest.mark.skipif(shutil.which("lsblk") is None, reason="lsblk not installed")
class TestRealLsblk:
    def test_smoke(self):
        r = block_device_list(include_loop=True)
        assert isinstance(r, BlockDeviceList)
        assert r.count == len(r.devices)
        # 真机上至少应有一个块设备
        assert r.count >= 1
