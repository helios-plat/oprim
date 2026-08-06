"""oprim.p2p_mailbox — ClawTeam-style P2P agent mailbox (ZeroMQ pub/sub).

Each agent binds a PUB socket on a local port, registers the port in the
team's peer directory, and connects SUB sockets to all known peers.  Messages
are JSON-serialised and sent in envelopes with a request_id for claimed
delivery.  Falls back gracefully to an in-process dict-based transport when
pyzmq is not installed.

3O element: ``oprim.p2p_mailbox`` (``P2PMailbox`` class).
"""

from __future__ import annotations

import json
import os
import socket
import time
import uuid
from pathlib import Path
from typing import Any


class P2PMailbox:
    """ZeroMQ pub/sub agent mailbox with graceful file/dict fallback.

    Usage::

        box = P2PMailbox(team_name="swarm-1", agent_name="leader")
        box.send(to="worker-a", content="process task", msg_type="message")
        msgs = box.receive("leader", limit=5)
    """

    def __init__(
        self,
        team_name: str,
        agent_name: str,
        base_dir: str | Path | None = None,
        use_zmq: bool = True,
    ) -> None:
        self.team_name = team_name
        self.agent_name = agent_name
        self._base = Path(base_dir) if base_dir else Path.home() / ".clawteam"
        self._base.mkdir(parents=True, exist_ok=True)
        self._zmq_ctx = None
        self._zmq_pub = None
        self._port = 0
        self._use_zmq = use_zmq

        # ZeroMQ setup
        if use_zmq:
            try:
                import zmq
                self._zmq_ctx = zmq.Context()
                self._zmq_pub = self._zmq_ctx.socket(zmq.PUB)
                self._port = self._zmq_pub.bind_to_random_port("tcp://127.0.0.1")
                self._register_peer()
            except Exception:
                self._zmq_ctx = None
                self._zmq_pub = None

    # -- send ---------------------------------------------------------------
    def send(
        self,
        to: str,
        content: str | None = None,
        msg_type: str = "message",
        request_id: str | None = None,
        **fields: Any,
    ) -> dict[str, Any]:
        msg = {
            "type": msg_type, "from": self.agent_name, "to": to,
            "content": content, "request_id": request_id or uuid.uuid4().hex[:8],
            "timestamp": time.time(), **{k: v for k, v in fields.items() if v is not None},
        }
        if self._zmq_pub is not None:
            try:
                self._zmq_pub.send_string(f"{to} {json.dumps(msg, default=str)}")
            except Exception:
                pass
        # always write to file inbox as fallback / reliable delivery
        inbox = self._inbox_dir(to)
        inbox.mkdir(parents=True, exist_ok=True)
        (inbox / f"{msg['request_id']}.json").write_text(
            json.dumps(msg, ensure_ascii=False, default=str), encoding="utf-8"
        )
        return msg

    def broadcast(self, content: str, msg_type: str = "broadcast") -> list[dict[str, Any]]:
        sent: list[dict[str, Any]] = []
        for peer in self._list_peers():
            if peer == self.agent_name:
                continue
            sent.append(self.send(to=peer, content=content, msg_type=msg_type))
        return sent

    # -- receive ------------------------------------------------------------
    def receive(self, agent_name: str | None = None, limit: int = 10) -> list[dict[str, Any]]:
        target = agent_name or self.agent_name
        inbox = self._inbox_dir(target)
        msgs: list[dict[str, Any]] = []
        for f in sorted(inbox.glob("*.json")):
            try:
                m = json.loads(f.read_text(encoding="utf-8"))
                msgs.append(m)
            except Exception:
                pass
            f.unlink(missing_ok=True)
            if len(msgs) >= limit:
                break
        return msgs

    def peek(self, agent_name: str | None = None) -> list[dict[str, Any]]:
        target = agent_name or self.agent_name
        inbox = self._inbox_dir(target)
        msgs = []
        for f in sorted(inbox.glob("*.json")):
            try:
                msgs.append(json.loads(f.read_text(encoding="utf-8")))
            except Exception:
                pass
        return msgs

    def peek_count(self, agent_name: str | None = None) -> int:
        target = agent_name or self.agent_name
        inbox = self._inbox_dir(target)
        return len(list(inbox.glob("*.json")))

    # -- lifecycle ----------------------------------------------------------
    def close(self) -> None:
        if self._zmq_pub is not None:
            try:
                self._zmq_pub.close()
            except Exception:
                pass
        if self._zmq_ctx is not None:
            try:
                self._zmq_ctx.term()
            except Exception:
                pass
        self._deregister_peer()

    # -- peer directory ----------------------------------------------------
    def _inbox_dir(self, agent_name: str) -> Path:
        return self._base / "teams" / self.team_name / "inboxes" / agent_name

    def _peers_dir(self) -> Path:
        d = self._base / "teams" / self.team_name / "peers"
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _register_peer(self) -> None:
        info = {
            "name": self.agent_name, "host": "127.0.0.1", "port": self._port,
            "pid": os.getpid(), "timestamp": time.time(),
        }
        (self._peers_dir() / f"{self.agent_name}.json").write_text(
            json.dumps(info, ensure_ascii=False), encoding="utf-8"
        )

    def _deregister_peer(self) -> None:
        (self._peers_dir() / f"{self.agent_name}.json").unlink(missing_ok=True)

    def _list_peers(self) -> list[str]:
        d = self._peers_dir()
        if not d.exists():
            return []
        peers = []
        for f in d.glob("*.json"):
            try:
                info = json.loads(f.read_text(encoding="utf-8"))
                peers.append(info.get("name", ""))
            except Exception:
                pass
        return peers
