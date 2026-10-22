"""Tests for the restored aggregate facade submodules.

`oprim.signal_analysis`, `oprim.correlation_options` and `oprim.data_fetch` are
public facade modules over the private implementations (`oprim._signal_analysis`,
`oprim._correlation_options`, `oprim._data_fetch`). The 2.x→3.x refactor kept the
implementations but dropped these aggregate facades, so consumers doing
`from oprim.signal_analysis import correlation_matrix` hit ModuleNotFoundError.

These tests pin two things:
1. the submodule-path import works at all (the regression), and
2. every name the facade declares in `__all__` is the *same object* as the one in
   the private implementation — i.e. the facade re-exports, it does not fork.
"""

from __future__ import annotations

import importlib

import pytest

FACADES = [
    ("oprim.signal_analysis", "oprim._signal_analysis"),
    ("oprim.correlation_options", "oprim._correlation_options"),
    ("oprim.data_fetch", "oprim._data_fetch"),
]


@pytest.mark.parametrize(("facade_path", "impl_path"), FACADES)
def test_facade_module_imports(facade_path: str, impl_path: str) -> None:
    facade = importlib.import_module(facade_path)
    impl = importlib.import_module(impl_path)
    assert facade is not None and impl is not None
    assert facade.__all__, f"{facade_path} must declare __all__"


@pytest.mark.parametrize(("facade_path", "impl_path"), FACADES)
def test_every_facade_name_is_the_implementation_object(facade_path: str, impl_path: str) -> None:
    facade = importlib.import_module(facade_path)
    impl = importlib.import_module(impl_path)
    for name in facade.__all__:
        assert hasattr(facade, name), f"{facade_path}.{name} is in __all__ but not bound"
        assert getattr(facade, name) is getattr(impl, name), (
            f"{facade_path}.{name} is a different object than {impl_path}.{name} "
            "— the facade must re-export, not re-implement (SPEC §1.4)"
        )


@pytest.mark.parametrize(("facade_path", "_impl"), FACADES)
def test_facade_names_are_reachable_from_package(facade_path: str, _impl: str) -> None:
    """`from oprim import X` must resolve for the public names the facade owns."""
    import oprim

    facade = importlib.import_module(facade_path)
    for name in facade.__all__:
        if name.endswith("Error"):
            continue
        assert hasattr(oprim, name), f"oprim.{name} missing from the package element map"


def test_signal_analysis_key_elements_callable() -> None:
    from oprim.signal_analysis import correlation_matrix, factor_attribution

    assert callable(correlation_matrix)
    assert callable(factor_attribution)


def test_correlation_options_key_elements_callable() -> None:
    from oprim.correlation_options import (
        compute_option_skew_curve_data,
        compute_rolling_correlation_heatmap,
    )

    assert callable(compute_option_skew_curve_data)
    assert callable(compute_rolling_correlation_heatmap)


def test_data_fetch_key_elements_callable() -> None:
    from oprim.data_fetch import fetch_funding_rate, fetch_order_book_depth

    assert callable(fetch_funding_rate)
    assert callable(fetch_order_book_depth)
