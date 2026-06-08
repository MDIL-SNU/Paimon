import os
import sys
import json
import random
import shutil
import string
import asyncio
from pathlib import Path
from dataclasses import dataclass, field
from datetime import datetime
from asyncio import StreamReader, StreamWriter
from typing import TextIO, BinaryIO, Protocol

from paimon_web import cfg
from paimon_web.util.log import debug, info, warning
from paimon_web.observability.models import RunStatusLiteral
from paimon_web.observability.run_index import run_index


# TODO: why don't import UploadFile from fastAPI directly?
class UploadFileLike(Protocol):
    """Protocol for file-like upload objects."""

    filename: str | None
    file: BinaryIO


_alphabet = string.ascii_lowercase + string.digits


def _id_gen():
    return "".join(random.choices(_alphabet, k=8))


@dataclass
class IdleWorker:
    """Pre-warmed worker waiting for init command."""

    process: asyncio.subprocess.Process
    stdin: asyncio.StreamWriter
    stdout: asyncio.StreamReader
    stderr: asyncio.StreamReader


def save_uploaded_files(env_id: str, files: list[UploadFileLike]) -> list[str]:
    """Save uploaded files to temp directory, return paths."""
    if not files or all(f.filename == "" for f in files):
        return []

    upload_dir = Path(f"/tmp/paimon_files_{env_id}")
    upload_dir.mkdir(exist_ok=True)

    saved_paths = []
    for file in files:
        if not file.filename:
            continue
        file_path = upload_dir / file.filename
        with open(file_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        saved_paths.append(str(file_path))
        info(f"[pm] Saved uploaded file: {file_path}")

    return saved_paths


@dataclass
class RunnerHandle:
    """Handle for a running runner process with its socket server."""

    env_id: str
    pid: int
    started_at: str
    task: str
    socket_path: str
    process: asyncio.subprocess.Process
    server: asyncio.Server
    stdout_file: TextIO
    stderr_file: TextIO
    # Queue for passing messages from API to socket handler
    message_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    # Queue for receiving streamed response chunks
    response_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    # Flag to signal streaming is complete
    streaming_done: asyncio.Event = field(default_factory=asyncio.Event)
    # Background task for monitoring process
    monitor_task: asyncio.Task | None = None
    # Background tasks for piping stdout/stderr to log files
    stdout_pipe_task: asyncio.Task | None = None
    stderr_pipe_task: asyncio.Task | None = None
    # Set when runner connects to the Unix socket (= "real ready")
    ready: asyncio.Event = field(default_factory=asyncio.Event)
    # Captured logs if process died before reaching ready
    early_error: str | None = None

    @property
    def returncode(self) -> int | None:
        return self.process.returncode


@dataclass
class EffectiveStatus:
    status: RunStatusLiteral
    early_error: str | None
    process_alive: bool
    ready: bool


class ProcessManager:
    """
    Manages multiple runner processes with Unix domain socket communication.
    Each runner gets its own socket server for IPC.

    Maintains a pool of pre-warmed workers to reduce launch latency.
    """

    def __init__(self, log_dir: str = "/tmp", pool_size: int = 2):
        self.log_dir = Path(log_dir)
        self.runners: dict[str, RunnerHandle] = {}
        self.pool_size = pool_size
        self.idle_workers: list[IdleWorker] = []
        self._pool_lock = asyncio.Lock()
        self._pool_task: asyncio.Task | None = None
        self._worker_script = (
            Path(__file__).parent.parent.parent.parent.parent
            / "paimon"
            / "src"
            / "paimon"
            / "runtime"
            / "worker.py"
        )

    def _get_socket_path(self, env_id: str) -> str:
        return f"/tmp/paimon_socket_{env_id}"

    async def start_pool(self) -> None:
        """Start pool maintenance. Call once at server startup."""
        self._pool_task = asyncio.create_task(self._maintain_pool())
        info(f"[pm] Worker pool started (size={self.pool_size})")

    async def _spawn_idle_worker(self) -> IdleWorker:
        """Spawn a pre-warmed worker waiting for init command."""
        proc = await asyncio.create_subprocess_exec(
            sys.executable,
            str(self._worker_script),
            stdin=asyncio.subprocess.PIPE,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        try:
            ready_line = await asyncio.wait_for(
                proc.stdout.readline(), timeout=60.0
            )
        except asyncio.TimeoutError:
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
            raise RuntimeError("Worker not ready within 60s")

        if ready_line.strip() != b"ready":
            try:
                proc.kill()
            except ProcessLookupError:
                pass
            await proc.wait()
            stderr = (await proc.stderr.read()).decode(
                errors="replace"
            )
            raise RuntimeError(
                f"Worker startup failed "
                f"(exit={proc.returncode}): {stderr}"
            )

        debug("[pm] Spawned idle worker")
        return IdleWorker(
            process=proc,
            stdin=proc.stdin,
            stdout=proc.stdout,
            stderr=proc.stderr,
        )

    async def _maintain_pool(self) -> None:
        """Background task to keep pool at desired size."""
        while True:
            try:
                async with self._pool_lock:
                    # Remove dead workers
                    self.idle_workers = [
                        w for w in self.idle_workers if w.process.returncode is None
                    ]
                    # Replenish pool
                    while len(self.idle_workers) < self.pool_size:
                        worker = await self._spawn_idle_worker()
                        self.idle_workers.append(worker)
            except asyncio.CancelledError:
                break
            except Exception as e:
                warning(f"[pm] Pool maintenance error: {e}")
            await asyncio.sleep(0.5)

    async def _get_worker(self) -> IdleWorker:
        """Get a worker from pool or spawn a new one."""
        async with self._pool_lock:
            # Remove dead workers first
            self.idle_workers = [
                w for w in self.idle_workers if w.process.returncode is None
            ]
            if self.idle_workers:
                return self.idle_workers.pop()
        # Pool empty, spawn new worker
        return await self._spawn_idle_worker()

    async def _pipe_to_file(
        self, reader: asyncio.StreamReader, file: TextIO
    ) -> None:
        """Copy from async reader to file."""
        try:
            while True:
                line = await reader.readline()
                if not line:
                    break
                file.write(line.decode("utf-8", errors="replace"))
                file.flush()
        except asyncio.CancelledError:
            pass

    async def _cleanup_runner(self, env_id: str):
        """Clean up resources for a runner."""
        handle = self.runners.get(env_id)
        if not handle:
            return

        # Cancel pipe tasks
        for task in (handle.stdout_pipe_task, handle.stderr_pipe_task):
            if task:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

        # Close socket server
        handle.server.close()
        await handle.server.wait_closed()

        # Close log files
        handle.stdout_file.close()
        handle.stderr_file.close()

        # Remove socket file
        if os.path.exists(handle.socket_path):
            os.remove(handle.socket_path)

        info(f"[pm:{env_id}] Resources cleaned up")

    async def _monitor_process(self, env_id: str):
        """Monitor a runner process and cleanup when it exits."""
        handle = self.runners.get(env_id)
        if not handle:
            return

        # Wait for process to exit
        await handle.process.wait()
        info(f"[pm:{env_id}] Process exited with code {handle.returncode}")

        # Wait for pipe tasks so logs are fully flushed
        for task in (
            handle.stdout_pipe_task,
            handle.stderr_pipe_task,
        ):
            if task:
                try:
                    await asyncio.wait_for(task, timeout=2.0)
                except (asyncio.TimeoutError, asyncio.CancelledError):
                    pass

        # Capture logs if process died before reaching ready
        if not handle.ready.is_set():
            handle.early_error = self._read_early_error(env_id)
            warning(f"[pm:{env_id}] Process died before ready")

        # Cleanup resources
        await self._cleanup_runner(env_id)

    def _read_early_error(self, env_id: str) -> str:
        """Read stderr/stdout logs for a process that died early."""
        parts: list[str] = []
        for suffix in ("stderr", "stdout"):
            path = self.log_dir / f"paimon_{env_id}.{suffix}.log"
            try:
                text = path.read_text().strip()
                if text:
                    parts.append(f"--- {suffix} ---\n{text}")
            except OSError:
                pass
        return "\n\n".join(parts) or "Process exited with no output."

    def get_effective_status(self, env_id: str) -> EffectiveStatus:
        """Centralized status resolution from live signals.

        Combines handle state, FS events, and process liveness
        into a single authoritative status.
        """
        from paimon_web.observability.fs_reader import fs_reader

        handle = self.runners.get(env_id)

        # No handle: fall back to DB status.
        # After server restart, handles are lost. Determine
        # status from DB + FS presence.
        if not handle:
            row = run_index.get_run(env_id)
            if not row:
                return EffectiveStatus("unknown", None, False, False)
            db_status: RunStatusLiteral = row["status"]  # type: ignore[assignment]
            fs_present = fs_reader.is_present(env_id)

            # Terminal states: keep as-is, mark ready if FS exists
            if db_status in ("succeeded", "failed", "interrupted"):
                return EffectiveStatus(db_status, None, False, fs_present)

            # Active states with no process => interrupted
            if db_status in ("pending", "running", "waiting_input"):
                return EffectiveStatus("interrupted", None, False, fs_present)

            return EffectiveStatus(db_status, None, False, fs_present)

        alive = handle.returncode is None
        ready = handle.ready.is_set()

        # Early error captured by monitor
        if handle.early_error:
            return EffectiveStatus("interrupted", handle.early_error, False, ready)

        # Not ready yet
        if not ready:
            if not alive:
                # Killed before socket connected (e.g. by terminate())
                return EffectiveStatus("interrupted", None, False, False)
            return EffectiveStatus("pending", None, True, False)

        # Ready: derive from FS events
        evt = fs_reader.get_last_event(env_id)
        evt_name = evt.get("name", "") if evt else ""

        if evt_name == "TaskComplete":
            return EffectiveStatus("succeeded", None, alive, True)
        if evt_name == "TaskFail":
            return EffectiveStatus("failed", None, alive, True)
        if (
            evt_name
            in (
                "InputRequiredEvent",
                "InputRequiredWithStepEvent",
            )
            and alive
        ):
            return EffectiveStatus("waiting_input", None, True, True)
        if alive:
            return EffectiveStatus("running", None, True, True)

        # Dead after being ready => interrupted
        return EffectiveStatus("interrupted", None, False, True)

    async def launch(self, config: dict, env_id: str | None = None) -> str:
        """
        Launch a new runner process with its own socket server.

        Uses pre-warmed worker from pool for reduced latency.

        Parameters
        ----------
        config
            Configuration dict containing task and other settings
        env_id
            Optional existing env_id for resurrection. If None, generates new one.

        Returns
        -------
        str
            Generated or provided env_id for the launched process
        """
        if env_id is None:
            env_id = _id_gen()
        else:
            # Resurrection: remove old handle if exists
            if env_id in self.runners:
                del self.runners[env_id]
                info(f"[pm:{env_id}] Removed old handle for resurrection")
        socket_path = self._get_socket_path(env_id)

        # Remove old socket if exists
        if os.path.exists(socket_path):
            os.remove(socket_path)

        info(f"[pm] Launching runner for {env_id}")
        info(f"[pm] Socket: {socket_path}")

        # Create queues for this runner
        message_queue: asyncio.Queue = asyncio.Queue()
        response_queue: asyncio.Queue = asyncio.Queue()
        streaming_done = asyncio.Event()
        ready = asyncio.Event()

        # Create socket handler that uses queues
        async def handle_connection(reader: StreamReader, writer: StreamWriter):
            """Handle a connection from the runner."""
            ready.set()
            debug(f"[pm:{env_id}] Runner connected, waiting for message")

            # Wait for a message from the API
            try:
                request = await message_queue.get()
            except asyncio.CancelledError:
                writer.close()
                await writer.wait_closed()
                return

            debug(f"[pm:{env_id}] Sending request to runner: {request['type']}")

            # Send request to runner
            payload = json.dumps(request).encode()
            writer.write(payload)
            await writer.drain()
            writer.write_eof()

            # Read response as JSON lines or streaming text
            buffer = b""
            while True:
                chunk = await reader.read(4096)
                if not chunk:
                    break
                buffer += chunk
                # Process complete lines (for JSON line protocol)
                while b"\n" in buffer:
                    line, buffer = buffer.split(b"\n", 1)
                    if line:
                        decoded = line.decode("utf-8")
                        await response_queue.put(decoded)

            # Flush any remaining buffer (for streaming text mode)
            if buffer:
                decoded = buffer.decode("utf-8", errors="replace")
                if decoded:
                    await response_queue.put(decoded)

            # Signal completion
            streaming_done.set()

            writer.close()
            await writer.wait_closed()
            debug(f"[pm:{env_id}] Connection closed")

        # Start socket server
        server = await asyncio.start_unix_server(handle_connection, path=socket_path)
        debug(f"[pm:{env_id}] Socket server started")

        # Open log files (named with env_id, created at launch time)
        stdout_path = self.log_dir / f"paimon_{env_id}.stdout.log"
        stderr_path = self.log_dir / f"paimon_{env_id}.stderr.log"
        stdout_file = open(stdout_path, "a")
        stderr_file = open(stderr_path, "a")

        # Get pre-warmed worker from pool
        worker = await self._get_worker()
        proc = worker.process

        # Send init command to worker via stdin
        init_cmd = json.dumps(
            {
                "env_id": env_id,
                "config": config,
                "socket_path": socket_path,
            }
        )
        worker.stdin.write((init_cmd + "\n").encode())
        await worker.stdin.drain()

        # Start piping stdout/stderr to log files
        stdout_pipe_task = asyncio.create_task(
            self._pipe_to_file(worker.stdout, stdout_file)
        )
        stderr_pipe_task = asyncio.create_task(
            self._pipe_to_file(worker.stderr, stderr_file)
        )

        run_index.upsert_run(
            env_id=env_id,
            task_name=config.get("task_name", "Unknown"),
            task=config.get("task", "Unknown"),
            status="pending",
            agent=config.get("agent", ""),
            total_cost=0,
            working_dir="Unknown",
        )

        # Create runner handle
        handle = RunnerHandle(
            env_id=env_id,
            pid=proc.pid,
            started_at=datetime.now().isoformat(),
            task=config.get("task", ""),
            socket_path=socket_path,
            process=proc,
            server=server,
            stdout_file=stdout_file,
            stderr_file=stderr_file,
            message_queue=message_queue,
            response_queue=response_queue,
            streaming_done=streaming_done,
            stdout_pipe_task=stdout_pipe_task,
            stderr_pipe_task=stderr_pipe_task,
            ready=ready,
        )
        self.runners[env_id] = handle

        # Start background monitor task
        handle.monitor_task = asyncio.create_task(self._monitor_process(env_id))

        info(f"[pm] Launched process {env_id} (PID: {proc.pid})")
        return env_id

    def get_status(self, env_id: str) -> RunnerHandle | None:
        """Get runner handle for a given env_id."""
        return self.runners.get(env_id)

    def list_active(self) -> list[RunnerHandle]:
        """List all active runners."""
        return [h for h in self.runners.values() if h.returncode is None]

    async def terminate(self, env_id: str) -> bool:
        """
        Terminate a running process.

        Returns
        -------
        bool
            True if process was terminated, False if not found
        """
        handle = self.runners.get(env_id)
        if not handle:
            return False

        try:
            # Cancel monitor task
            if handle.monitor_task:
                handle.monitor_task.cancel()
                try:
                    await handle.monitor_task
                except asyncio.CancelledError:
                    pass

            # Terminate process if still running
            if handle.returncode is None:
                handle.process.terminate()
                await handle.process.wait()

            # Cleanup resources
            await self._cleanup_runner(env_id)

            info(f"[pm] Terminated process {env_id} (PID: {handle.pid})")
            return True
        except ProcessLookupError:
            return False
        except Exception as e:
            warning(f"[pm] Error terminating process {env_id}: {e}")
            return False

    def _is_session_dead(self, env_id: str) -> bool:
        """Check if session is dead (not found or process exited)."""
        handle = self.runners.get(env_id)
        if not handle:
            return True
        return handle.returncode is not None

    async def _resurrect_session(self, env_id: str) -> str:
        """Resurrect a dead session by relaunching with empty config."""
        info(f"[pm:{env_id}] Resurrecting dead session")
        await self.launch({}, env_id=env_id)
        return env_id

    async def send_message_stream(
        self, env_id: str, message: str, files: list[str] | None = None
    ):
        """
        Send a message to a runner and yield streaming response chunks.
        If session is dead, resurrect it before sending the message.

        Parameters
        ----------
        env_id
            Environment ID
        message
            User message to send
        files
            Optional list of file paths to transfer to agent

        Yields
        ------
        str
            Response chunks as they arrive
        """
        # Resurrect dead session if needed
        if self._is_session_dead(env_id):
            await self._resurrect_session(env_id)

        handle = self.runners.get(env_id)
        if not handle:
            yield "error: Failed to resurrect session"
            return

        # Clear previous state
        handle.streaming_done.clear()
        while not handle.response_queue.empty():
            try:
                handle.response_queue.get_nowait()
            except asyncio.QueueEmpty:
                break

        # Put message in queue for the socket handler
        request = {"type": "user_msg", "user_msg": message}
        if files:
            request["files"] = files
        await handle.message_queue.put(request)
        debug(f"[pm:{env_id}] Message queued, waiting for response")

        # Yield response chunks as they arrive
        while True:
            if handle.streaming_done.is_set() and handle.response_queue.empty():
                break

            try:
                chunk = await asyncio.wait_for(
                    handle.response_queue.get(), timeout=0.1
                )
                yield chunk
            except asyncio.TimeoutError:
                if handle.streaming_done.is_set():
                    break
                if handle.returncode is not None:
                    debug(f"[pm:{env_id}] Process died during stream")
                    break

        debug(f"[pm:{env_id}] Streaming complete")

    async def cleanup(self):
        """Clean up all processes, sockets, and worker pool."""
        # Stop pool maintenance
        if self._pool_task:
            self._pool_task.cancel()
            try:
                await self._pool_task
            except asyncio.CancelledError:
                pass

        # Terminate idle workers
        async with self._pool_lock:
            for worker in self.idle_workers:
                worker.process.terminate()
            self.idle_workers.clear()

        # Terminate active runners
        for env_id in list(self.runners.keys()):
            await self.terminate(env_id)


info("[pm] Initializing ProcessManager")
process_manager = ProcessManager(log_dir=cfg.log_dir)
info("[pm] ProcessManager initialized successfully")
