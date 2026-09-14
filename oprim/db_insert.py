"""Compatibility import for the retired database insert primitive."""

from obase.persistence.crud import insert_one as db_insert

__all__ = ["db_insert"]
