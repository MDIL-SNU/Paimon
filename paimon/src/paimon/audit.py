"""Unified token usage audit via contextvar-scoped collector.

Usage:
    with audit.scope(env_id="abc", sub_wd="01_relax"):
        # any LLM call inside here (including tool-internal calls)
        # will have entries pushed to this scope automatically
        ...
    audit.flush(env)  # writes .audit.jsonl
"""

from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass, field
from datetime import datetime

from paimon.util.log import debug
from paimon.world import get_env
from paimon.token_sum import TokenUsageEntry


@dataclass
class AuditScope:
    sub_wd: str | None
    env_id: str | None
    entries: list[dict] = field(default_factory=list)


_scope: ContextVar[AuditScope | None] = ContextVar("_audit_scope", default=None)


@contextmanager
def scope(env_id: str | None = None, sub_wd: str | None = None):
    """Create an audit scope. Entries pushed inside are tagged with sub_wd."""
    s = AuditScope(sub_wd=sub_wd, env_id=env_id)
    token = _scope.set(s)
    try:
        yield s
    finally:
        _scope.reset(token)


def push(entry: TokenUsageEntry) -> None:
    """Push entry to active scope. No-op if no scope."""
    s = _scope.get(None)
    if s is None:
        return
    d = entry.to_dict()
    d["time"] = datetime.now().isoformat()
    d["sub_wd"] = s.sub_wd
    s.entries.append(d)


def flush() -> None:
    """Write scope entries to .audit.jsonl and clear."""
    s = _scope.get(None)
    if s is None or not s.entries:
        return
    env = get_env(s.env_id)
    debug(f"[audit] flush to sub_wd: {s.sub_wd}")
    env.append_jsonl(s.entries, ".audit.jsonl", sub_wd=s.sub_wd)
    s.entries.clear()
