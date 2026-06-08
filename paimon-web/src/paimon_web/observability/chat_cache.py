import asyncio
from typing import TypedDict


class CachedChat(TypedDict):
    messages: list[dict]
    version: int


class ChatCache:
    """In-memory cache for chat messages to prevent UI flickering.

    This cache stores the latest chat state for each run. When reading from
    filesystem (which can be slow or incomplete), we compare with cached
    version and use the more complete one.

    Assumes monotonic growth: messages only append, never delete/modify.
    """

    def __init__(self):
        self._cache: dict[str, CachedChat] = {}
        self._lock = asyncio.Lock()

    async def get(self, env_id: str) -> list[dict] | None:
        """Get cached messages for a run, or None if not cached."""
        async with self._lock:
            cached = self._cache.get(env_id)
            if cached:
                return cached["messages"].copy()
            return None

    async def update(self, env_id: str, messages: list[dict]) -> None:
        """Update cache with new messages.

        Only updates if new messages list is longer (monotonic growth).
        """
        async with self._lock:
            cached = self._cache.get(env_id)
            if cached is None:
                self._cache[env_id] = {
                    "messages": messages,
                    "version": len(messages),
                }
            elif len(messages) > cached["version"]:
                self._cache[env_id] = {
                    "messages": messages,
                    "version": len(messages),
                }

    async def get_or_update(
        self, env_id: str, fs_messages: list[dict]
    ) -> tuple[list[dict], bool]:
        """Get cached messages or update cache with filesystem messages.

        Returns (messages, used_cache) tuple.
        - If cache is newer/equal → returns (cached, True)
        - If filesystem is newer → returns (fs_messages, False) and updates cache
        """
        async with self._lock:
            cached = self._cache.get(env_id)
            fs_len = len(fs_messages)

            if cached is None:
                self._cache[env_id] = {
                    "messages": fs_messages,
                    "version": fs_len,
                }
                return fs_messages, False

            if fs_len <= cached["version"]:
                return cached["messages"].copy(), True

            self._cache[env_id] = {
                "messages": fs_messages,
                "version": fs_len,
            }
            return fs_messages, False

    async def clear(self, env_id: str) -> None:
        """Clear cache for a specific run."""
        async with self._lock:
            self._cache.pop(env_id, None)

    async def clear_all(self) -> None:
        """Clear all cached chats."""
        async with self._lock:
            self._cache.clear()


chat_cache = ChatCache()
