"""Disk SMART health oprim — smartctl 磁盘健康探测(只读 R0)."""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

from pydantic import BaseModel

from oprim._exceptions import OprimError, OprimNotFoundError

# 关注的 SMART 属性 id → 友好名(ATA). NVMe 走 nvme_smart_health_information_log.
_ATA_ATTRS = {
    5: "reallocated_sectors",
    9: "power_on_hours",
    12: "power_cycles",
    194: "temperature_celsius",
    197: "pending_sectors",
    198: "uncorrectable_sectors",
}


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class SmartAttribute(BaseModel):
    id: int | None = None
    name: str
    value: int | None = None  # 归一化/原始值(取原始 raw)


class SmartHealth(BaseModel):
    device: str
    available: bool  # smartctl 是否成功读到 SMART
    passed: bool | None = None  # 整体健康自评(SMART overall-health)
    protocol: str | None = None  # ATA / NVMe / SCSI
    model: str | None = None
    serial: str | None = None
    temperature_celsius: int | None = None
    power_on_hours: int | None = None
    power_cycles: int | None = None
    reallocated_sectors: int | None = None
    pending_sectors: int | None = None
    attributes: list[SmartAttribute] = []
    message: str | None = None  # 降级/告警说明(如 smartctl 缺失、非 SMART 设备)


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _run_smartctl(device: str) -> dict[str, Any]:
    """跑 `smartctl -j -H -A -i <device>` 返回解析后的 JSON. 供测试 monkeypatch.

    注意 smartctl 即便读到数据也常以非零退出码表示 SMART 位标志(bit0-7),
    因此不能用 returncode 判失败 —— 只要有合法 JSON 就用它.
    """
    if shutil.which("smartctl") is None:
        raise OprimError("smartctl not found on host (smartmontools required)")
    try:
        proc = subprocess.run(
            ["smartctl", "-j", "-H", "-A", "-i", device],
            capture_output=True,
            text=True,
            timeout=20,
            check=False,  # smartctl 用退出码表 SMART 位,不能据此判失败
        )
    except subprocess.TimeoutExpired as e:
        raise OprimError(f"smartctl timed out on {device}", cause=e) from e
    if not proc.stdout.strip():
        raise OprimError(f"smartctl produced no output for {device}: {proc.stderr.strip()}")
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise OprimError(f"smartctl produced invalid JSON for {device}", cause=e) from e


def _extract_ata(attrs_table: list[dict[str, Any]]) -> dict[str, int]:
    """从 ata_smart_attributes.table 抽关注属性的 raw.value."""
    out: dict[str, int] = {}
    for row in attrs_table:
        aid = row.get("id")
        if aid in _ATA_ATTRS:
            raw = (row.get("raw") or {}).get("value")
            if isinstance(raw, int):
                out[_ATA_ATTRS[aid]] = raw
    return out


# ---------------------------------------------------------------------------
# disk_smart_probe
# ---------------------------------------------------------------------------


def disk_smart_probe(
    *,
    device: str,
    smartctl_json: str | None = None,
) -> SmartHealth:
    """读单块盘的 SMART 健康与关键属性,解析 `smartctl -j` 输出.

    只读(R0). 需 root 与 smartmontools. 用于存储面板给盘打健康徽章、
    在 uptime/自愈里预警坏盘.

    执行位置由调用方决定:不传 smartctl_json 时本地跑 smartctl;传入则只解析,
    调用方可在别处(特权 host-shell / 远端节点)取到原始输出再交本原语解析.

    Args:
        device: 块设备路径, 如 "/dev/sda" 或 "/dev/nvme0n1".
        smartctl_json: 可选. 预先取到的 `smartctl -j ...` 原始 JSON 字符串.

    Returns:
        SmartHealth: available 表示是否读到 SMART; passed 为整体健康自评;
            温度/通电时长/重映射扇区等关键项已抽平, 完整项在 attributes.

    Raises:
        OprimNotFoundError: device 参数为空/非法.
        OprimError: smartctl 缺失、超时或输出非法.
    """
    if not device or not device.startswith("/dev/"):
        raise OprimNotFoundError(f"invalid device path: {device!r}")

    if smartctl_json is not None:
        try:
            data = json.loads(smartctl_json)
        except json.JSONDecodeError as e:
            raise OprimError(f"smartctl_json is not valid JSON for {device}", cause=e) from e
    else:
        data = _run_smartctl(device)

    info = data.get("device") or {}
    protocol = info.get("protocol") or info.get("type")
    passed = (data.get("smart_status") or {}).get("passed")
    available = (
        "smart_status" in data
        or "ata_smart_attributes" in data
        or "nvme_smart_health_information_log" in data
    )

    extracted: dict[str, int] = {}
    attributes: list[SmartAttribute] = []

    ata = data.get("ata_smart_attributes")
    if ata and isinstance(ata.get("table"), list):
        extracted = _extract_ata(ata["table"])
        for row in ata["table"]:
            attributes.append(
                SmartAttribute(
                    id=row.get("id"),
                    name=str(row.get("name", "")),
                    value=(row.get("raw") or {}).get("value"),
                )
            )

    nvme = data.get("nvme_smart_health_information_log")
    if nvme:
        if isinstance(nvme.get("temperature"), int):
            extracted.setdefault("temperature_celsius", nvme["temperature"])
        if isinstance(nvme.get("power_on_hours"), int):
            extracted.setdefault("power_on_hours", nvme["power_on_hours"])
        if isinstance(nvme.get("power_cycles"), int):
            extracted.setdefault("power_cycles", nvme["power_cycles"])
        for k, v in nvme.items():
            if isinstance(v, int):
                attributes.append(SmartAttribute(name=str(k), value=v))

    # 顶层温度兜底(smartctl 常给 temperature.current)
    top_temp = (data.get("temperature") or {}).get("current")
    if isinstance(top_temp, int):
        extracted.setdefault("temperature_celsius", top_temp)

    return SmartHealth(
        device=device,
        available=bool(available),
        passed=passed,
        protocol=protocol,
        model=info.get("model_name") or data.get("model_name"),
        serial=data.get("serial_number"),
        temperature_celsius=extracted.get("temperature_celsius"),
        power_on_hours=extracted.get("power_on_hours"),
        power_cycles=extracted.get("power_cycles"),
        reallocated_sectors=extracted.get("reallocated_sectors"),
        pending_sectors=extracted.get("pending_sectors"),
        attributes=attributes,
        message=None if available else "SMART data unavailable for this device",
    )
