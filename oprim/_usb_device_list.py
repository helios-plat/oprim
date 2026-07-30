"""USB device enumeration oprim — lsusb USB 设备枚举(只读 R0)."""

from __future__ import annotations

import re
import shutil
import subprocess

from pydantic import BaseModel

from oprim._exceptions import OprimError

# `lsusb` 一行形如: "Bus 001 Device 003: ID 048d:5702 <描述>"
_LINE_RE = re.compile(
    r"^Bus\s+(?P<bus>\d+)\s+Device\s+(?P<device>\d+):\s+"
    r"ID\s+(?P<vid>[0-9a-fA-F]{4}):(?P<pid>[0-9a-fA-F]{4})\s*(?P<desc>.*)$"
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class UsbDevice(BaseModel):
    bus: int
    device: int
    vendor_id: str  # 4-hex, 如 "048d"
    product_id: str  # 4-hex
    description: str | None = None
    is_root_hub: bool = False


class UsbDeviceList(BaseModel):
    devices: list[UsbDevice]
    count: int


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _run_lsusb() -> str:
    """跑 `lsusb` 返回原始 stdout. 供测试 monkeypatch."""
    if shutil.which("lsusb") is None:
        raise OprimError("lsusb not found on host (usbutils required)")
    try:
        proc = subprocess.run(
            ["lsusb"],
            capture_output=True,
            text=True,
            timeout=10,
            check=True,
        )
    except subprocess.TimeoutExpired as e:
        raise OprimError("lsusb timed out", cause=e) from e
    except subprocess.CalledProcessError as e:
        raise OprimError(f"lsusb failed: {e.stderr.strip()}", cause=e) from e
    return proc.stdout


def _parse_line(line: str) -> UsbDevice | None:
    m = _LINE_RE.match(line.strip())
    if not m:
        return None
    desc = (m.group("desc") or "").strip() or None
    return UsbDevice(
        bus=int(m.group("bus")),
        device=int(m.group("device")),
        vendor_id=m.group("vid").lower(),
        product_id=m.group("pid").lower(),
        description=desc,
        is_root_hub=bool(desc and "root hub" in desc.lower()),
    )


# ---------------------------------------------------------------------------
# usb_device_list
# ---------------------------------------------------------------------------


def usb_device_list(
    *,
    include_root_hubs: bool = False,
    lsusb_output: str | None = None,
) -> UsbDeviceList:
    """枚举 USB 设备,解析 `lsusb` 输出.

    只读(R0). 用于存储/外设面板识别插入的 U 盘、移动硬盘、外设等.

    执行位置由调用方决定:不传 lsusb_output 时本地跑 lsusb;传入则只解析,
    调用方可在别处(特权 host-shell / 远端节点)取到原始输出再交本原语解析.

    Args:
        include_root_hubs: 是否保留根 hub(Linux Foundation root hub). 默认剔除以减噪.
        lsusb_output: 可选. 预先取到的 `lsusb` 原始 stdout;给定则解析它而非本地执行.

    Returns:
        UsbDeviceList: devices 为解析出的 USB 设备, count 为其数量.

    Raises:
        OprimError: lsusb 缺失、超时或非零退出.
    """
    raw = _run_lsusb() if lsusb_output is None else lsusb_output
    devices: list[UsbDevice] = []
    for line in raw.splitlines():
        dev = _parse_line(line)
        if dev is None:
            continue
        if not include_root_hubs and dev.is_root_hub:
            continue
        devices.append(dev)
    return UsbDeviceList(devices=devices, count=len(devices))
