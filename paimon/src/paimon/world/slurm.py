import fcntl
import asyncio
from datetime import datetime
from typing import Any


from paimon import cfg
from paimon.util.log import debug
from ._connection import SafeConnection


_tracker = None

TERMINAL_STATES = frozenset(
    {
        "COMPLETED",
        "FAILED",
        "CANCELLED",
        "TIMEOUT",
        "NODE_FAIL",
        "PREEMPTED",
    }
)


async def get_slurm_tracker() -> "SlurmTracker":
    global _tracker
    if _tracker is None:
        _tracker = SlurmTracker()
        await asyncio.sleep(_tracker.poll_interval_sec)
    return _tracker


class SlurmTracker:
    """Poll sacct and track enrolled Slurm jobs by 'who' id."""

    def __init__(
        self,
        poll_interval_sec: float = 5,
    ) -> None:
        self._conn = SafeConnection(
            user=cfg.paimon_slurm_user,
            host=cfg.paimon_slurm_host,
            port=cfg.paimon_slurm_port,
        )
        self.poll_interval_sec = poll_interval_sec

        self._server_state = None
        self._last_check = None

        self._job_states = {}  # type: dict[str, dict[int, dict[str, Any]]]
        self._parse_fn = (
            self.parse_sacct if cfg.slurm_parse_cmd == "sacct" else self.parse_squeue
        )

        loop = asyncio.get_event_loop()
        loop.create_task(self._poll_loop())

    def _lock(self, fd):
        fcntl.flock(fd, fcntl.LOCK_EX)

    def _unlock(self, fd):
        fcntl.flock(fd, fcntl.LOCK_UN)

    def _active_job_ids(self) -> list[str]:
        """Return job IDs that are not yet in a terminal state."""
        return [
            str(jid)
            for jobs in self._job_states.values()
            for jid, job in jobs.items()
            if job.get("state") not in TERMINAL_STATES
        ]

    async def _update_server_state(self) -> None:
        cmd = 'sinfo -o "%P %a %D %c %C"'
        res = self._conn.run(cmd, hide=True)
        self._server_state = res.stdout

    async def _update_job_states(self) -> None:
        active_jids = self._active_job_ids()
        if not active_jids:
            return

        if self._parse_fn == self.parse_sacct:
            sacct = await self.parse_sacct(active_jids)
        else:
            sacct = await self.parse_squeue()

        for jobs in self._job_states.values():
            for jid, job in jobs.items():
                jid = int(jid)
                if jid in sacct:
                    raw_state = sacct[jid]["State"]
                    # sacct may return "CANCELLED by ..." — normalize
                    job["state"] = raw_state.split()[0] if raw_state else raw_state
                elif job["state"] == "WAITING_UPDATE":
                    pass
                elif job["state"] not in TERMINAL_STATES:
                    job["state"] = "UNKNOWN"

    async def _poll_loop(self) -> None:
        """Background loop to refresh sacct cache."""
        while True:
            try:
                await self._update_job_states()
                await self._update_server_state()
                self._last_check = datetime.now().isoformat()
            except Exception as e:
                debug(f"Slurm pool loop exception {e}")
                pass
            await asyncio.sleep(self.poll_interval_sec)

    def is_cancel_valid(self, env_id: str, job_id: int) -> bool:
        """When LLM tries to kill job, check whether the job is enrolled by the same
        env"""
        return job_id in self._job_states.get(env_id, {})

    async def parse_sacct(self, job_ids: list[str] | None = None) -> dict:
        """Run sacct for specific job IDs and return {job_id: info}."""
        if not job_ids:
            return {}

        fields = [
            "JobID",
            "JobName",
            "Partition",
            "Account",
            "AllocCPUs",
            "State",
            "ExitCode",
        ]
        fmt = ",".join(fields)
        cmd = (
            "sacct --noheader --parsable2 "
            f"-j {','.join(job_ids)} "
            f"--format={fmt}"
        )
        res = self._conn.run(cmd, hide=True)
        out = {}
        for ln in res.stdout.strip().splitlines():
            cols = ln.split("|")
            if "." in cols[0]:
                continue
            jid = int(cols[0])
            out[jid] = dict(zip(fields[1:], cols[1:]))
        return out

    async def parse_squeue(self) -> dict:
        fields = ["JobID", "JobName", "Partition", "Account", "CPUs", "State"]
        fmt = "%i|%j|%P|%a|%C|%T"
        cmd = f"squeue -h -o '{fmt}'"
        ret = self._conn.run(cmd, hide=True)
        out = {}
        for ln in ret.stdout.strip().splitlines():
            cols = ln.split("|")
            jid = int(cols[0])
            out[jid] = dict(zip(fields[1:], cols[1:]))
        return out

    def enroll_job(
        self, env_id: str, job_id: int, script: str, sbatch_opts: dict
    ) -> None:
        """Enroll a job under 'who' with its script and sbatch options.
        Must be used with env.run, that do 'sbatch'

        TODO: add some check routines (job id in sacct)
        """
        if env_id not in self._job_states:
            self._job_states[env_id] = {}

        assert job_id not in self._job_states[env_id]
        self._job_states[env_id][job_id] = dict(
            script=script,
            enroll_time=datetime.now().isoformat(),
            sbatch_opts=sbatch_opts,
            state="WAITING_UPDATE",
        )

    @property
    def job_states(self) -> dict[str, dict[int, dict[str, Any]]]:
        """get job states: dict[env_id, dict[job_id, job_meta_dict]]"""
        return self._job_states

    @property
    def server_state(self) -> str:
        if not self._server_state:
            raise ValueError("Server not ready")
        return self._server_state
