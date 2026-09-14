"""Small, consistent boundary for optional feature dependencies."""

from __future__ import annotations


class MissingOptionalDependency(ImportError):  # noqa: N818 - public error name is contractual
    def __init__(self, *, feature: str, extra: str, package: str) -> None:
        super().__init__(
            f"Optional feature {feature!r} requires {package!r}; "
            f"install with `pip install oprim[{extra}]`."
        )
        self.feature = feature
        self.extra = extra
        self.package = package


def require_optional(module: object | None, *, feature: str, extra: str, package: str) -> object:
    if module is None:
        raise MissingOptionalDependency(feature=feature, extra=extra, package=package)
    return module
