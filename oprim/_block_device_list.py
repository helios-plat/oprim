"""Block device enumeration oprim — lsblk 块设备/分区枚举(只读 R0)."""

from __future__ import annotations

import json
import shutil
import subprocess
from typing import Any

from pydantic import BaseModel

from oprim._exceptions import OprimError

# 显式指定 lsblk 列(而非 -O),按需 .get() 容忍旧版缺列。
_LSBLK_COLUMNS = (
    "NAME,KNAME,PATH,SIZE,TYPE,FSTYPE,MOUNTPOINT,MODEL,SERIAL,"
    "VENDOR,LABEL,UUID,ROTA,TRAN,HOTPLUG,RM,RO,STATE"
)


# ---------------------------------------------------------------------------
# Models
# ---------------------------------------------------------------------------


class BlockDevice(BaseModel):
    name: str
    kname: str | None = None
    path: str | None = None
    size_bytes: int | None = None
    type: str | None = None  # disk / part / loop / rom / lvm / crypt ...
    fstype: str | None = None
    mountpoint: str | None = None
    model: str | None = None
    serial: str | None = None
    vendor: str | None = None
    label: str | None = None
    uuid: str | None = None
    rotational: bool | None = None  # True=HDD, False=SSD/NVMe
    transport: str | None = None  # sata / nvme / usb / ...
    hotplug: bool | None = None
    removable: bool | None = None
    read_only: bool | None = None
    state: str | None = None
    children: list[BlockDevice] = []


class BlockDeviceList(BaseModel):
    devices: list[BlockDevice]
    count: int  # 顶层设备数


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _run_lsblk() -> dict[str, Any]:
    """跑 `lsblk -J -b -O` 并返回解析后的 JSON dict. 供测试 monkeypatch."""
    if shutil.which("lsblk") is None:
        raise OprimError("lsblk not found on host (util-linux required)")
    try:
        proc = subprocess.run(
            ["lsblk", "-J", "-b", "-o", _LSBLK_COLUMNS],
            capture_output=True,
            text=True,
            timeout=15,
            check=True,
        )
    except subprocess.TimeoutExpired as e:
        raise OprimError("lsblk timed out", cause=e) from e
    except subprocess.CalledProcessError as e:
        raise OprimError(f"lsblk failed: {e.stderr.strip()}", cause=e) from e
    try:
        return json.loads(proc.stdout)
    except json.JSONDecodeError as e:
        raise OprimError("lsblk produced invalid JSON", cause=e) from e


def _to_bool(v: Any) -> bool | None:
    """lsblk 的布尔列可能是 true/false、"1"/"0"、1/0 或 None."""
    if v is None:
        return None
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    s = str(v).strip().lower()
    if s in ("1", "true", "yes"):
        return True
    if s in ("0", "false", "no", ""):
        return False
    return None


def _parse_node(node: dict[str, Any]) -> BlockDevice:
    mp = node.get("mountpoint")
    # 新版 lsblk 用 mountpoints(列表);取第一个非空
    if mp is None:
        mps = node.get("mountpoints") or []
        mp = next((m for m in mps if m), None)
    children = [_parse_node(c) for c in (node.get("children") or [])]
    return BlockDevice(
        name=node.get("name", ""),
        kname=node.get("kname"),
        path=node.get("path"),
        size_bytes=node.get("size") if isinstance(node.get("size"), int) else None,
        type=node.get("type"),
        fstype=node.get("fstype"),
        mountpoint=mp,
        model=(node.get("model") or None),
        serial=(node.get("serial") or None),
        vendor=(node.get("vendor") or None),
        label=(node.get("label") or None),
        uuid=(node.get("uuid") or None),
        rotational=_to_bool(node.get("rota")),
        transport=(node.get("tran") or None),
        hotplug=_to_bool(node.get("hotplug")),
        removable=_to_bool(node.get("rm")),
        read_only=_to_bool(node.get("ro")),
        state=(node.get("state") or None),
        children=children,
    )


# ---------------------------------------------------------------------------
# block_device_list
# ---------------------------------------------------------------------------


def block_device_list(
    *,
    disks_only: bool = False,
    include_loop: bool = False,
    lsblk_json: str | None = None,
) -> BlockDeviceList:
    """枚举块设备(盘/分区),解析 `lsblk -J -b` 输出.

    只读(R0). 用于存储面板列出物理盘与分区、识别 HDD/SSD、USB、挂载点等.

    执行位置由调用方决定:不传 lsblk_json 时本地 `subprocess` 跑 lsblk(适合
    直接跑在目标主机的场景,如节点 agent);传 lsblk_json 时只解析、不执行,
    调用方可在别处(如特权 host-shell / 远端节点)取到原始输出再交给本原语解析.

    Args:
        disks_only: 只保留顶层 type=="disk" 的设备(去掉 loop/rom/lvm 等).
        include_loop: 是否保留 loop 设备(snap/镜像回环). 默认剔除以减噪.
        lsblk_json: 可选. 预先取到的 `lsblk -J` 原始 JSON 字符串;给定则解析它而非本地执行.

    Returns:
        BlockDeviceList: devices 为顶层设备(分区在各自 children), count 为顶层数.

    Raises:
        OprimError: lsblk 缺失、超时、非零退出或输出非法 JSON.
    """
    if lsblk_json is not None:
        try:
            data = json.loads(lsblk_json)
        except json.JSONDecodeError as e:
            raise OprimError("lsblk_json is not valid JSON", cause=e) from e
    else:
        data = _run_lsblk()
    tops = [_parse_node(n) for n in (data.get("blockdevices") or [])]

    def _keep(d: BlockDevice) -> bool:
        if disks_only and d.type != "disk":
            return False
        return include_loop or d.type != "loop"

    tops = [d for d in tops if _keep(d)]
    return BlockDeviceList(devices=tops, count=len(tops))


BlockDevice.model_rebuild()
