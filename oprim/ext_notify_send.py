"""oprim.ext_notify_send — render a template and send a notification via a
registered obase.NotificationProvider (category="notification").

Design note: obase.NotificationProvider only exposes concrete send_email
(to/subject/body) and send_sms (to/message) methods — there is no generic
"send(channel, template, data)" provider method. This atom is the templating
+ channel-dispatch layer: `template` is rendered as a Jinja2 template string
against `data`, then routed to the matching provider method. `data` must
supply "to" (both channels) and "subject" (channel="email" only).
"""

from __future__ import annotations

from typing import Any

from jinja2 import Template
from obase import ProviderRegistry
from obase.exceptions import ProviderNotFoundError

from ._exceptions import NotifyOprimError

_VALID_CHANNELS = {"email", "sms"}


async def ext_notify_send(
    provider: str, *, channel: str, template: str, data: dict[str, Any]
) -> bool:
    """Render `template` with `data` and send it over `channel`.

    Args:
        provider: Provider name registered in ProviderRegistry (category="notification").
        channel: "email" or "sms".
        template: Jinja2 template string for the message body.
        data: Template context; must include "to" (recipient), and "subject"
            when channel="email".

    Returns:
        True on success.

    Raises:
        NotifyOprimError: Unknown channel, missing required `data` keys,
            provider not registered, or the provider call failed.
    """
    if channel not in _VALID_CHANNELS:
        raise NotifyOprimError(f"unknown channel: {channel!r} (expected email or sms)")
    if "to" not in data:
        raise NotifyOprimError("data must include 'to'")
    if channel == "email" and "subject" not in data:
        raise NotifyOprimError("data must include 'subject' for channel='email'")

    try:
        notification_provider = ProviderRegistry.get().generic("notification", provider)
    except ProviderNotFoundError as exc:
        raise NotifyOprimError(f"notification provider not found: {provider!r}", cause=exc) from exc

    rendered = Template(template).render(**data)

    try:
        if channel == "email":
            await notification_provider.send_email(
                to=data["to"], subject=data["subject"], body=rendered
            )
        else:
            await notification_provider.send_sms(to=data["to"], message=rendered)
    except Exception as exc:
        raise NotifyOprimError(f"send failed: {exc}", cause=exc) from exc

    return True
