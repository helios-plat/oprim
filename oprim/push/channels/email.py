"""Email push channel via SMTP (stdlib smtplib, no external deps)."""
from __future__ import annotations

import asyncio
import smtplib
from datetime import datetime, timezone
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText

from oprim._logging import log
from oprim.push.errors import PushConfigError, PushDeliveryError
from oprim.push.protocol import PushResult

_DEFAULT_TEMPLATE = """\
<html><body>
<h2>{title}</h2>
<p>{body}</p>
{deep_link_html}
</body></html>
"""

_DEEP_LINK_HTML = '<p><a href="{url}">Open in Stratum</a></p>'


class EmailPushChannel:
    """Sends notifications via SMTP (TLS/STARTTLS)."""

    name = "email"

    def __init__(
        self,
        smtp_host: str,
        smtp_port: int,
        smtp_user: str,
        smtp_password: str,
        from_address: str,
        use_tls: bool = True,
    ) -> None:
        if not smtp_host:
            raise PushConfigError("smtp_host is required for EmailPushChannel")
        if not from_address:
            raise PushConfigError("from_address is required for EmailPushChannel")
        self._host = smtp_host
        self._port = smtp_port
        self._user = smtp_user
        self._password = smtp_password
        self._from = from_address
        self._use_tls = use_tls

    async def send(
        self,
        recipient: str,
        title: str,
        body: str,
        deep_link: str | None = None,
        metadata: dict | None = None,
    ) -> PushResult:
        deep_link_html = _DEEP_LINK_HTML.format(url=deep_link) if deep_link else ""
        html_body = _DEFAULT_TEMPLATE.format(
            title=title,
            body=body,
            deep_link_html=deep_link_html,
        )

        def _send_smtp():
            msg = MIMEMultipart("alternative")
            msg["Subject"] = title
            msg["From"] = self._from
            msg["To"] = recipient
            msg.attach(MIMEText(body, "plain"))
            msg.attach(MIMEText(html_body, "html"))

            if self._use_tls:
                server = smtplib.SMTP_SSL(self._host, self._port)
            else:
                server = smtplib.SMTP(self._host, self._port)
                server.starttls()
            try:
                if self._user and self._password:
                    server.login(self._user, self._password)
                server.sendmail(self._from, [recipient], msg.as_string())
            finally:
                server.quit()

        try:
            await asyncio.to_thread(_send_smtp)
            log.info("email_push_sent", recipient=recipient)
            return PushResult(
                channel=self.name,
                success=True,
                recipient=recipient,
                sent_at=datetime.now(tz=timezone.utc),
            )
        except smtplib.SMTPException as exc:
            log.error("email_push_failed", recipient=recipient, error=str(exc))
            return PushResult(
                channel=self.name,
                success=False,
                recipient=recipient,
                error_message=str(exc),
            )
        except Exception as exc:
            log.error("email_push_unexpected_error", recipient=recipient, error=str(exc))
            return PushResult(
                channel=self.name,
                success=False,
                recipient=recipient,
                error_message=str(exc),
            )

    async def health_check(self) -> bool:
        def _check():
            if self._use_tls:
                server = smtplib.SMTP_SSL(self._host, self._port, timeout=5)
            else:
                server = smtplib.SMTP(self._host, self._port, timeout=5)
            try:
                server.noop()
                return True
            finally:
                server.quit()

        try:
            return await asyncio.to_thread(_check)
        except Exception as exc:
            log.warning("email_health_check_failed", error=str(exc))
            return False
