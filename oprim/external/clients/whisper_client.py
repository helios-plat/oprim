"""oprim.external.clients.whisper_client — UNIMPLEMENTED PLACEHOLDER.

This module is an empty shell: every class below has no behaviour. It is not a
capability and must not be imported as one.

It is kept only because `oskill.knowledge/*` still imports these names, so
deleting the file would break oskill at import time. Fixing that properly means
either implementing the clients or migrating oskill off them — separate work.

P0-D removed these names from `oprim.__all__`: they were advertising as public
capabilities while doing nothing. Direct module-path imports still work.
"""

__oprim_layer__ = "infra"  # placeholder shell, not a capability


class WhisperClient:
    """Placeholder — no behaviour implemented."""


class WhisperSegment:
    """Placeholder — no behaviour implemented."""
