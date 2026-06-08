from paimon import cfg
from paimon.world.environment import (
    Environment,
    new_environment,
    is_id_already_present
)
from paimon.world._connection import SafeConnection

_envs = {}

_conn = None


def get_connection() -> SafeConnection:
    global _conn
    if not _conn:
        _conn = SafeConnection(
            user=cfg.paimon_user,
            host=cfg.paimon_host,
            port=cfg.paimon_port,
        )
    return _conn


def register_env(env: Environment) -> None:
    _envs[env.id] = env


def get_env(id: str | None) -> Environment:
    return _envs[id]


def remove_env(id: str | None) -> None:
    del _envs[id]
