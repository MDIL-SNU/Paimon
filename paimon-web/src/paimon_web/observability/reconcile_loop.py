# Status resolution is centralized in ProcessManager.get_effective_status().
# This loop's sole job is persisting that status to DB for:
#   1. Index page queries (list_runs, filtering)
#   2. Fallback when no handle exists (after server restart)
# Endpoints read status directly from get_effective_status()
# for real-time accuracy.

import asyncio
from datetime import datetime
from typing import cast

from paimon_web.observability.models import RunStatusLiteral
from paimon_web.observability.fs_reader import fs_reader
from paimon_web.observability.process_manager import (
    EffectiveStatus,
    process_manager as pm,
)
from paimon_web.observability.run_index import run_index, RunRow
from paimon_web.util.log import warning, debug


RECONCILE_INTERVAL = 3.0  # seconds
LAUNCH_GRACE = 180.0  # seconds

active_states: list[RunStatusLiteral] = [
    "pending",
    "running",
    "waiting_input",
]


def _upsert_from_row(run, new_status):
    run.update({"status": new_status})
    run_index.upsert_run(
        env_id=run["env_id"],
        task_name=run["task_name"],
        task=run["task"],
        status=cast(RunStatusLiteral, run["status"]),
        agent=run["agent"],
        total_cost=run["total_cost"],
        working_dir=run["working_dir"],
    )


async def _reconcile() -> None:
    debug("[debug] recon once started")
    active_runs: list[RunRow] = run_index.list_runs_by_status(active_states)
    now = datetime.now()

    for run in active_runs:
        env_id = run["env_id"]
        es = pm.get_effective_status(env_id)

        if es.status == "pending":
            # Use handle.started_at (launch time of current process),
            # not DB created_at which is stale after resurrection.
            handle = pm.get_status(env_id)
            ts = handle.started_at if handle else run["created_at"]
            try:
                t = datetime.fromisoformat(ts)
                if (now - t).total_seconds() > LAUNCH_GRACE:
                    warning(f"[recon] {env_id} not responsive, terminating")
                    await pm.terminate(env_id)
                    es = EffectiveStatus("interrupted", None, False, False)
            except ValueError as e:
                warning(f"[recon] {env_id} timestamp corrupt: {e}")
                es = EffectiveStatus("interrupted", None, False, False)

        # Enrich with FS metadata when available
        if es.ready:
            globals_ = fs_reader.read_json_or_empty(env_id, ".globals.json")
            tokens = fs_reader.get_token_usage(env_id)
            working_dir = str(fs_reader.get_working_dir(env_id))
            run.update(
                {
                    "agent": globals_.get("agent", ""),
                    "task": globals_.get("task", "UNKNOWN"),
                    "task_name": globals_.get("task_name", "UNKNOWN"),
                    "working_dir": working_dir,
                    "total_cost": tokens.total_cost,
                }
            )

        _upsert_from_row(run, es.status)
    debug("[debug] recon once ended")


async def reconcile_loop(
    stop_event: asyncio.Event | None = None,
) -> None:
    """Periodically persist effective status to DB."""

    while True:
        if stop_event is not None and stop_event.is_set():
            break

        started_at = datetime.now()

        try:
            await _reconcile()
        except Exception as e:
            warning(f"[reconciler] failed during reconciliation: {e}")

        elapsed = datetime.now() - started_at
        sleep_for = max(0.0, RECONCILE_INTERVAL - elapsed.total_seconds())

        await asyncio.sleep(sleep_for)
