"""Compatibility data contracts for historical storage tests."""

from dataclasses import dataclass


@dataclass(frozen=True)
class StorageFile:
    file_id: str
    name: str = ""
    size: int = 0


@dataclass(frozen=True)
class UploadResult:
    file_id: str
    size: int
    md5: str = ""


__all__ = ["StorageFile", "UploadResult"]
