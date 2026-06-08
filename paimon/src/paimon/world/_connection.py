"""
TODO: [IMPORTANT] async, non-blocking run
"""

import time
import functools

import paramiko
import paramiko.ssh_exception
from fabric import Connection

from paimon.util.log import debug

MAX_RETRIES = 3
BACKOFF = 1

RETRY_EXC = (paramiko.SSHException, EOFError, OSError)


def with_retry(func):
    """Retry Fabric call if SSH connection drops."""

    @functools.wraps(func)
    def wrapper(self, *args, **kwargs):
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                return func(self, *args, **kwargs)
            except RETRY_EXC as e:
                debug(
                    f"[with_retry] {func.__name__} failed, attempt {attempt}: {e!r}"
                )
                self.close()
                if attempt == MAX_RETRIES:
                    raise
                time.sleep(BACKOFF * attempt)
                self.open()

    return wrapper


class SafeConnection(Connection):
    @with_retry
    def run(self, *args, **kwargs):
        return super().run(*args, **kwargs)

    @with_retry
    def put(self, *args, **kwargs):
        return super().put(*args, **kwargs)

    @with_retry
    def get(self, *args, **kwargs):
        return super().get(*args, **kwargs)
