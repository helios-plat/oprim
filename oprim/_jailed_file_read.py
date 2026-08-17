"""oprim.jailed_file_read — one file read under PathJail."""

from __future__ import annotations

from obase.sandbox.path_jail import PathJail


def jailed_file_read(jail: PathJail, *, file_path: str) -> str:
    """Read a text file after the path prison accepts it."""
    safe = jail.resolve_and_verify(file_path)
    return safe.read_text(encoding="utf-8")
