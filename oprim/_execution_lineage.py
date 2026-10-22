"""Immutable execution-lineage primitive."""
from __future__ import annotations

from dataclasses import dataclass, replace


@dataclass(frozen=True, slots=True)
class ExecutionLineage:
    """Provider-neutral ancestry for one execution attempt."""

    root_run_id: str
    parent_execution_id: str | None = None
    execution_id: str | None = None
    task_id: str | None = None
    attempt_id: str | None = None
    controller_generation: int = 0

    def __post_init__(self) -> None:
        if not self.root_run_id:
            raise ValueError("root_run_id must not be empty")
        if self.controller_generation < 0:
            raise ValueError("controller_generation must be >= 0")

    def child(
        self,
        *,
        execution_id: str,
        task_id: str | None = None,
        attempt_id: str | None = None,
        controller_generation: int | None = None,
    ) -> ExecutionLineage:
        """Create the next immutable lineage node."""
        if not execution_id:
            raise ValueError("execution_id must not be empty")
        return replace(
            self,
            parent_execution_id=self.execution_id,
            execution_id=execution_id,
            task_id=task_id,
            attempt_id=attempt_id,
            controller_generation=(
                self.controller_generation
                if controller_generation is None
                else controller_generation
            ),
        )
