"""3O 范式 Phase 2 — oprim 原子操作测试 (obase 依赖注入).

覆盖: 扁平命名空间惰性暴露、§4.4 位置参数规约、四大原子的行为语义。
"""

from __future__ import annotations

import inspect

import pytest
from obase import KVStoreProvider, LocalVFS, SQLiteKVStore, VFSProvider
from obase.contracts import IVFS_Sandbox

from oprim import (
    execute_sandbox_cmd,
    put_state,
    read_isolated_file,
    write_isolated_file,
)

ATOMS = [
    read_isolated_file,
    write_isolated_file,
    execute_sandbox_cmd,
    put_state,
]


def test_flat_namespace_lazy_exposure():
    """扁平命名空间直接暴露能力, 无模块前缀 (惰性发现)。"""
    for fn in ATOMS:
        assert callable(fn)
    assert "read_isolated_file" in __import__("oprim").__all__
    assert "write_isolated_file" in __import__("oprim").__all__
    assert "execute_sandbox_cmd" in __import__("oprim").__all__
    assert "put_state" in __import__("oprim").__all__


def test_signature_rule_sec44():
    """§4.4: 最多 1 个位置参数, 其余强制 keyword-only。"""
    for fn in ATOMS:
        params = list(inspect.signature(fn).parameters.values())
        positional = [p for p in params if p.kind in (p.POSITIONAL_ONLY, p.POSITIONAL_OR_KEYWORD)]
        assert len(positional) <= 1, f"{fn.__name__} 位置参数超限: {[p.name for p in positional]}"
        for p in params[1:]:
            assert p.kind is p.KEYWORD_ONLY, f"{fn.__name__} 参数 {p.name} 必须 keyword-only"


def test_atoms_depend_on_obase_contracts():
    """依赖注入类型指向 obase 契约 (面向契约编程)。"""
    assert "VFSProvider" in str(read_isolated_file.__annotations__["vfs"])
    assert "VFSProvider" in str(write_isolated_file.__annotations__["vfs"])
    assert "VFSProvider" in str(execute_sandbox_cmd.__annotations__["vfs"])
    assert "KVStoreProvider" in str(put_state.__annotations__["store"])
    assert VFSProvider is IVFS_Sandbox  # 别名即契约, 非实现
    assert KVStoreProvider is not None


async def test_read_write_roundtrip(tmp_path):
    """read_isolated_file + write_isolated_file 经 LocalVFS 往返。"""
    vfs = LocalVFS(str(tmp_path))
    await write_isolated_file("hello 3O", path="a/b.txt", vfs=vfs)
    assert await read_isolated_file("a/b.txt", vfs=vfs) == "hello 3O"


async def test_execute_sandbox_cmd_success(tmp_path):
    """命令成功: 原样返回结果, 不判断业务含义。"""
    vfs = LocalVFS(str(tmp_path))
    result = await execute_sandbox_cmd("echo atom-ok", vfs=vfs, timeout=10)
    assert result["exit_code"] == 0
    assert "atom-ok" in result["stdout"]


async def test_execute_sandbox_cmd_nonzero_not_raised(tmp_path):
    """常规命令失败 (exit_code != 0) 不拦截, 返回原样结果。"""
    vfs = LocalVFS(str(tmp_path))
    result = await execute_sandbox_cmd("exit 3", vfs=vfs, timeout=10)
    assert result["exit_code"] == 3


async def test_execute_sandbox_cmd_sandbox_crash_raises(tmp_path):
    """物理沙盒崩溃 (exit_code == -1) 抛 RuntimeError, 由上层编排引擎处理。"""
    vfs = LocalVFS(str(tmp_path))
    with pytest.raises(RuntimeError, match="VFS Sandbox failed"):
        await execute_sandbox_cmd("sleep 5", vfs=vfs, timeout=1)


async def test_put_state_roundtrip(tmp_path):
    """put_state 经 SQLiteKVStore 持久化, 不感知业务概念。"""
    store = SQLiteKVStore(str(tmp_path / "state.db"))
    await put_state("k/1", value={"n": 42}, store=store)
    assert await store.get("k/1") == {"n": 42}
