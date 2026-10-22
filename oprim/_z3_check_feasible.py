"""oprim._z3_check_feasible — 单次调用 Z3/SMT 求解器验证约束可满足性.

铁律: ≤1 个位置参数，其余 keyword-only.

组合: SMTSolverAdapter (Protocol 注入).

例:
    >>> result = _z3_check_feasible(
    ...     "(declare-const x Int) (assert (> x 0))",
    ...     solver_op=my_solver,
    ... )
    >>> result["verdict"]
    'sat'
"""

from __future__ import annotations

from typing import Any, Protocol


class SMTSolverProtocol(Protocol):
    """SMT 求解器 Protocol — 不 import obase，由调用方注入."""

    def check(self, *constraints: Any) -> str: ...
    def create_solver(self) -> Any: ...


def _z3_check_feasible(
    smt2_script: str,
    *,
    solver_op: SMTSolverProtocol | None = None,
) -> dict[str, Any]:
    """单次调用 Z3/SMT 求解器验证 SMT-LIB2 约束的可满足性.

    签名遵循 oprim 铁律：最多 1 个位置参数，其余 kw-only.

    如果 solver_op 未注入，将尝试惰性 import z3。

    Args:
        smt2_script: SMT-LIB2 格式的约束脚本
        solver_op: SMT 求解器适配器（注入 SMTSolverProtocol）

    Returns:
        {
            "verdict": "sat" | "unsat" | "unknown",
            "error": str | None,
        }
    """
    if solver_op is not None:
        try:
            verdict = solver_op.check(smt2_script)
            return {"verdict": verdict, "error": None}
        except Exception as e:
            return {"verdict": "unknown", "error": str(e)}

    # 回退：惰性 import z3
    try:
        import z3

        # 解析简单 SMT2 脚本（此处为简化实现）
        # 生产使用 z3.parse_smt2_string
        solver = z3.Solver()
        solver.set("timeout", 5000)
        solver.from_string(smt2_script)
        result = solver.check()
        if result == z3.sat:
            verdict = "sat"
        elif result == z3.unsat:
            verdict = "unsat"
        else:
            verdict = "unknown"
        return {"verdict": verdict, "error": None}
    except ImportError:
        return {"verdict": "unknown", "error": "z3-solver not installed"}
    except Exception as e:
        return {"verdict": "unknown", "error": str(e)}
