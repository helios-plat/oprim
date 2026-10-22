"""Public facade for oprim.correlation_options.

oprim 3.x moved the implementations into the private ``oprim._correlation_options`` module
and, for most symbols, exposes them through thin public facade modules (see
``signal_processing.py`` / ``ic_oos_decay.py``). This aggregate facade was
missing after the 2.x→3.x refactor — callers doing ``from oprim.correlation_options
import ...`` (oskill.signal_fusion_skills, helios analytics/collectors, and
oprim's own test_correlation_* suite) hit ModuleNotFoundError.
Restored 2026-07-10 to re-export the full public surface of _correlation_options.
"""

from oprim._correlation_options import (
    OprimError,
    compute_rolling_correlation_heatmap,
    compute_option_skew_curve_data,
)

__all__ = [
    "OprimError",
    "compute_rolling_correlation_heatmap",
    "compute_option_skew_curve_data",
]
