"""Compatibility import for the retired database read primitive."""

from obase.persistence.crud import read_one as db_read

__all__ = ["db_read"]
