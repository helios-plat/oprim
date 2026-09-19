"""O1/O2/O3 oprim 层测试: _scipy_linear_assign, _z3_check_feasible, _sandbox_execute_probe."""

from __future__ import annotations

import inspect
from unittest.mock import MagicMock


class TestScipyLinearAssign:
    def test_basic_assignment(self):
        from oprim._scipy_linear_assign import _scipy_linear_assign

        workers = ["W1", "W2"]
        tasks = ["T1", "T2"]
        matrix = [[1, 5], [5, 1]]
        result = _scipy_linear_assign(matrix, workers=workers, tasks=tasks)
        assert result["status"] == "success"
        assert result["total_cost"] == 2.0

    def test_shape_mismatch(self):
        from oprim._scipy_linear_assign import _scipy_linear_assign

        result = _scipy_linear_assign([[1, 2]], workers=["W1", "W2"], tasks=["T1"])
        assert result["status"] == "error"

    def test_inf_handling(self):
        from oprim._scipy_linear_assign import _scipy_linear_assign

        inf_val = float("inf")
        result = _scipy_linear_assign(
            [[1, inf_val], [inf_val, 1]],
            workers=["A", "B"],
            tasks=["X", "Y"],
        )
        assert result["status"] == "success"

    def test_positional_only_one(self):
        from oprim._scipy_linear_assign import _scipy_linear_assign

        sig = inspect.signature(_scipy_linear_assign)
        positional = [
            p for p in sig.parameters.values() if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        assert len(positional) <= 1


class TestZ3CheckFeasible:
    def test_without_z3_returns_unknown(self):
        from oprim._z3_check_feasible import _z3_check_feasible

        result = _z3_check_feasible("(assert false)")
        assert result["verdict"] in ("unknown", "unsat", "sat")

    def test_with_solver_protocol(self):
        from oprim._z3_check_feasible import _z3_check_feasible

        mock_solver = MagicMock()
        mock_solver.check.return_value = "sat"
        result = _z3_check_feasible("(assert true)", solver_op=mock_solver)
        assert result["verdict"] == "sat"

    def test_positional_only_one(self):
        from oprim._z3_check_feasible import _z3_check_feasible

        sig = inspect.signature(_z3_check_feasible)
        positional = [
            p for p in sig.parameters.values() if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        assert len(positional) <= 1


class TestSandboxExecuteProbe:
    def test_simple_command(self):
        from oprim._sandbox_execute_probe import _sandbox_execute_probe

        result = _sandbox_execute_probe("echo test")
        assert result["status"] == "success"
        assert "test" in result["stdout"]

    def test_timeout(self):
        from oprim._sandbox_execute_probe import _sandbox_execute_probe

        result = _sandbox_execute_probe("sleep 10", timeout=0.1)
        assert result["status"] == "timeout"

    def test_positional_only_one(self):
        from oprim._sandbox_execute_probe import _sandbox_execute_probe

        sig = inspect.signature(_sandbox_execute_probe)
        positional = [
            p for p in sig.parameters.values() if p.kind == inspect.Parameter.POSITIONAL_OR_KEYWORD
        ]
        assert len(positional) <= 1
