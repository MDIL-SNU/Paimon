"""Minimal reproducer for orphaned remote python after timeout.

Requires:
    source ~/paimon/wd/dev/source_this

Example:
    python -m paimon.developers.repro_timeout_orphan --env-id timeout-repro
"""

from __future__ import annotations

import argparse
import shlex
import traceback
import uuid

import paimon.world as world
from paimon.world.environment import new_environment

SCRIPT_FILENAME = "sleep.py"
SUB_WD = "timeout_repro"
VENV_NAME = "base"
TIMEOUT_SECONDS = 10
SLEEP_SECONDS = 100000


def _build_sleep_script(sleep_seconds: int) -> str:
    return f"""import os
import sys
import time

print(f"REPRO_START pid={{os.getpid()}} argv={{sys.argv}}", flush=True)
time.sleep({sleep_seconds})
print("REPRO_END", flush=True)
"""


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run a long python process with a short timeout over remote env.run, "
            "then check whether process remains alive."
        )
    )
    parser.add_argument("--env-id", default="timeout-repro")
    args = parser.parse_args()

    env_id = new_environment(id="timeout-repro")
    env = world.get_env(env_id)

    marker = f"PAIMON_REPRO_{uuid.uuid4().hex}"
    py_code = _build_sleep_script(SLEEP_SECONDS)
    env.write_file(content=py_code, remote_path=SCRIPT_FILENAME, sub_wd=SUB_WD)

    quoted_filename = shlex.quote(SCRIPT_FILENAME)
    quoted_marker = shlex.quote(marker)
    run_cmd = f"python -u {quoted_filename} --marker {quoted_marker}"

    print(f"[info] env_id={env_id}")
    print(f"[info] wd={env.wd}")
    print(f"[info] sub_wd={SUB_WD}")
    print(f"[info] filename={SCRIPT_FILENAME}")
    print(f"[info] marker={marker}")
    print(f"[info] run_cmd={run_cmd}")
    print(f"[info] timeout={TIMEOUT_SECONDS}s sleep={SLEEP_SECONDS}s")

    try:
        env.run(
            run_cmd,
            wrap_for_llm=False,
            sub_wd=SUB_WD,
            timeout=TIMEOUT_SECONDS,
            venv_name=VENV_NAME,
        )
        print("[warn] Command finished without timeout. Increase sleep or reduce timeout.")
    except Exception as exc:
        print(f"[timeout/exception] {type(exc).__name__}: {exc}")
        print("[traceback]")
        print(traceback.format_exc().rstrip())

    print("\n[probe] process state right after timeout/exception")
    probe_before = env.sys_run(
        f"pgrep -af {quoted_marker} | grep -v 'pgrep -af' || true"
    )
    print(probe_before.strip() or "(none)")


if __name__ == "__main__":
    main()
