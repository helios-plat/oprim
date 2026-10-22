"""Regression for atoms whose callable name matches the module path."""

import importlib

import oprim


def test_same_named_export_supports_callable_and_module_access() -> None:
    module = importlib.import_module("oprim.template_render")
    package_value = oprim.template_render

    assert package_value is module
    assert callable(package_value)
    assert callable(package_value.template_render)
    assert package_value.template_render is module.template_render


def test_public_from_import_remains_callable() -> None:
    from oprim import template_render

    assert callable(template_render)
