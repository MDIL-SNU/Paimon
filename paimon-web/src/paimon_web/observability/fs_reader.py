import asyncio
import json
import re
from pathlib import Path
from typing import cast
from cachetools import TTLCache

from llama_index.core.memory import ChatMemoryBuffer
from llama_index.core.base.llms.types import ChatMessage, MessageRole

from paimon_web import cfg
from paimon_web.observability.chat import parse_chat_json
from paimon_web.observability.run_index import run_index, RunStatusLiteral
from paimon_web.observability.models import (
    RunConfig,
    RunDetail,
    SubtaskDetail,
    SubtaskInfo,
    TokenUsageSummary,
    TokenUsageItem,
    FileNode,
    DirectoryNode,
    FileSystemNode,
)
from paimon_web.util.log import debug, info


STRUCTURE_EXTENSIONS = {".extxyz", ".xyz", ".cif", ".poscar", ".lammps-data"}
EXTERNAL_FILES = "external_files"


class LocalReader:
    def __init__(self, base_wd: str):
        self.base_wd = Path(base_wd)
        self._cache = TTLCache(maxsize=256, ttl=1.0)
        self._lock = asyncio.Lock()

    def read_json(self, env_id: str, path: Path | str) -> dict:
        if not (run_dir := self.get_working_dir(env_id)):
            raise FileNotFoundError(f"No such env_id dir: {env_id}")

        fpath = run_dir / path
        if not fpath.exists():
            raise FileNotFoundError(f"No such file: {fpath}")

        return json.loads(fpath.read_text())

    def read_json_or_empty(self, env_id: str, path: Path | str) -> dict:
        try:
            return self.read_json(env_id, path)
        except (FileNotFoundError, json.JSONDecodeError) as e:
            return {}

    def get_last_event(self, env_id: str) -> dict[str, str] | None:
        dct = self.read_json_or_empty(env_id, ".event.json")
        try:
            last_event = dct["events"][-1]
        except (KeyError, IndexError) as e:
            debug(f"[reader] latest event get failed: {e}")
            return None
        return last_event

    def get_events(self, env_id: str) -> list[dict]:
        """Read events from .event.json."""
        data = self.read_json_or_empty(env_id, ".event.json")
        return data.get("events", [])

    def is_new_event_format(self, env_id: str) -> bool:
        """Check if run uses new event format with rich fields."""
        events = self.get_events(env_id)
        if not events:
            return False
        # New format events have extra fields beyond just "name"
        for ev in events:
            if len(ev) > 1:
                return True
        return False

    def run_dir(self, env_id: str) -> Path:
        return self.base_wd / f"__{env_id}"

    def _read_json(self, path: Path) -> dict:
        """Read and parse JSON file"""
        if not path.exists():
            return {}
        return json.loads(path.read_text())

    def _determine_status(self, run_dir: Path) -> str:
        """Determine run status from directory contents"""
        if (run_dir / "TASK_SUCCESS").exists():
            return "succeeded"
        elif (run_dir / "TASK_FAIL").exists():
            return "failed"
        else:
            return "unknown"

    def _subtask_name_to_dir(self, subtask_name: str, task_number: int) -> str:
        """Convert subtask name to directory name using same logic as models.py"""
        sub_wd = re.sub(r"[^A-Za-z0-9_-]", "", subtask_name.replace(" ", "_"))
        return f"{task_number:02d}_{sub_wd}"

    def find_subtask_info(
        self, env_id: str, dir_name: str
    ) -> SubtaskInfo:
        """Find subtask in plan by dir_name."""
        plan_data = self.read_json_or_empty(env_id, ".plan.json")
        for idx, st in enumerate(plan_data.get("subtasks", [])):
            computed = self._subtask_name_to_dir(
                st["name"], idx + 1
            )
            if computed == dir_name:
                info = SubtaskInfo(**st)
                info.dir_name = computed
                return info
        raise FileNotFoundError(
            f"Subtask {dir_name} not found in plan"
        )

    def _parse_token_usage(self, token_data: dict) -> TokenUsageSummary:
        """Parse token usage JSON into model"""
        # items = [TokenUsageItem(**item) for item in token_data.get("items", [])]

        items = []
        for item in token_data.get("items", []):
            item["tool_call"] = (
                [item["tool_call"]]
                if isinstance(item["tool_call"], str)
                else item["tool_call"]
            )
            items.append(TokenUsageItem(**item))

        return TokenUsageSummary(
            total_input_tokens=token_data.get("total_input_tokens", 0),
            total_cached_tokens=token_data.get("total_cached_tokens", 0),
            total_output_tokens=token_data.get("total_output_tokens", 0),
            total_reasoning_tokens=token_data.get("total_reasoning_tokens"),
            total_tokens=token_data.get("total_tokens", 0),
            total_cost=token_data.get("total_cost", 0.0),
            items=items,
        )

    def get_token_usage(self, env_id: str) -> TokenUsageSummary:
        """Return token usage summary from env. If not found, return usage summary
        filled with 0
        """
        tok_raw = self.read_json_or_empty(
            env_id, ".token.json"
        ) or self.read_json_or_empty(env_id, ".agent_tokens.json")
        if not tok_raw:
            tok_raw = {}
        return self._parse_token_usage(tok_raw)

    def is_present(self, env_id: str) -> bool:
        run_dir = self.run_dir(env_id)
        return run_dir.exists() and run_dir.is_dir()

    def get_working_dir(self, env_id: str) -> Path | None:
        return self.base_wd / f"__{env_id}" if self.is_present(env_id) else None

    def _build_run_detail(self, env_id: str) -> RunDetail:
        runrow = run_index.get_run(env_id)
        if not runrow:
            raise ValueError(f"[reader] No such run found: {env_id}")

        status = cast(RunStatusLiteral, runrow["status"])

        token_usage = self.get_token_usage(env_id)
        globals_data = self.read_json_or_empty(env_id, ".globals.json")
        plan_data = self.read_json_or_empty(env_id, ".plan.json")
        complete_data = self.read_json_or_empty(env_id, "complete.json")
        subtasks = []
        for idx, st in enumerate(plan_data.get("subtasks", [])):
            info = SubtaskInfo(**st)
            info.dir_name = self._subtask_name_to_dir(
                st["name"], idx + 1
            )
            subtasks.append(info)

        run_config = RunConfig(**self.read_json_or_empty(env_id, ".config.json"))

        if status == "succeeded" and not complete_data:
            info(f"{env_id} is succeeded but completion data is not found")

        ef_dir = self.run_dir(env_id) / EXTERNAL_FILES
        has_ef = ef_dir.exists() and ef_dir.is_dir()

        return RunDetail(
            env_id=globals_data["env_id"],
            status=status,
            task_name=globals_data.get("task_name", "-"),
            task=globals_data.get("task", "-"),
            agent=globals_data.get("agent", ""),
            subtasks=subtasks,
            config=run_config,
            token_usage=token_usage,
            completion_report=complete_data.get("report") if complete_data else None,
            completion_data=complete_data if complete_data else None,
            has_external_files=has_ef,
        )

    async def get_run_detail(self, env_id: str) -> RunDetail:
        """Get detailed information about a specific run"""
        # fast path
        cached = self._cache.get(env_id)
        if cached is not None:
            return cached

        async with self._lock:
            # double check after acquiring lock
            cached = self._cache.get(env_id)
            if cached is not None:
                return cached

            result = self._build_run_detail(env_id)
            self._cache[env_id] = result
            return result

    def _get_external_files_detail(
        self, env_id: str, run_dir: Path
    ) -> SubtaskDetail:
        ef_dir = run_dir / EXTERNAL_FILES
        if not ef_dir.exists() or not ef_dir.is_dir():
            raise FileNotFoundError(f"external_files not found in run {env_id}")
        output_files = [
            f.name
            for f in ef_dir.iterdir()
            if f.is_file() and not f.name.startswith(".")
        ]
        output_tree = self._walk_directory(ef_dir, ef_dir)
        return SubtaskDetail(
            env_id=env_id,
            subtask_name="External Files",
            dir_name=EXTERNAL_FILES,
            agent_memory=None,
            token_usage=None,
            critic_token_usage=None,
            output_files=output_files,
            output_tree=output_tree,
            has_critic=False,
        )

    def get_subtask(self, env_id: str, dir_name: str) -> SubtaskDetail:
        """Get detailed information about a specific subtask."""
        debug(f"[reader] Getting subtask {dir_name} for run {env_id}")

        run_dir = self.run_dir(env_id)

        if dir_name == EXTERNAL_FILES:
            return self._get_external_files_detail(env_id, run_dir)

        subtask_dir = self.get_subtask_dir(env_id, dir_name)
        info = self.find_subtask_info(env_id, dir_name)

        debug("[reader] Found subtask directory, reading agent data")
        agent_memory_data = self.read_json_or_empty(
            env_id, subtask_dir / ".agent_memory.json"
        )
        if not agent_memory_data:
            debug("[reader] WARNING: .agent_memory.json is missing or empty")

        agent_token_data = self._parse_token_usage(
            self.read_json_or_empty(env_id, subtask_dir / ".agent_tokens.json")
        )
        if not agent_token_data:
            debug("[reader] WARNING: .agent_tokens.json is missing or empty")

        agent_messages = []
        if agent_memory_data:
            try:
                buffer = ChatMemoryBuffer.from_dict(
                    agent_memory_data
                )
                all_messages = buffer.get_all()
                # Filter out system and developer messages
                agent_messages = [
                    msg
                    for msg in all_messages
                    if msg.role not in (MessageRole.SYSTEM, MessageRole.DEVELOPER)
                ]
            except Exception as e:
                debug(f"[reader] WARNING: Failed to parse agent memory: {e}")

        output_files = [
            f.name
            for f in subtask_dir.iterdir()
            if f.is_file() and not f.name.startswith(".")
        ]

        # Build tree structure for arbitrary depth support
        output_tree = self._walk_directory(subtask_dir, subtask_dir)

        return SubtaskDetail(
            env_id=env_id,
            subtask_name=info.name,
            dir_name=dir_name,
            agent_memory=agent_messages,
            token_usage=agent_token_data,
            critic_token_usage=None,
            output_files=output_files,
            output_tree=output_tree,
            has_critic=False,
        )

    def get_planner_chat(self, env_id: str) -> list[ChatMessage]:
        """Get planner chat history for a run"""
        debug(f"[reader] Getting planner chat for run {env_id}")

        runrow = run_index.get_run(env_id)
        if not runrow:
            raise ValueError(f"[reader] No such run found: {env_id}")

        chat_data = self.read_json_or_empty(env_id, ".chat.json")
        return parse_chat_json(chat_data)

    def get_agent_chat(
        self, env_id: str, dir_name: str
    ) -> list[ChatMessage]:
        """Get agent chat history for a subtask."""
        debug(
            f"[reader] Getting chat for subtask {dir_name}"
            f" in run {env_id}"
        )
        subtask_dir = self.get_subtask_dir(env_id, dir_name)

        agent_memory_data = self.read_json_or_empty(
            env_id, subtask_dir / ".agent_memory.json"
        )
        if not agent_memory_data:
            debug(
                "[reader] WARNING: .agent_memory.json is"
                " missing or empty"
            )
            return []

        if "memory" in agent_memory_data:
            return parse_chat_json(agent_memory_data["memory"])
        return parse_chat_json(agent_memory_data)

    def get_subtask_dir(self, env_id: str, dir_name: str) -> Path:
        """Get subtask directory path by dir_name."""
        if dir_name == EXTERNAL_FILES:
            ef_dir = self.run_dir(env_id) / EXTERNAL_FILES
            if not ef_dir.exists():
                raise FileNotFoundError(
                    "external_files directory not found"
                )
            return ef_dir

        subtask_dir = self.run_dir(env_id) / dir_name
        if not subtask_dir.exists():
            raise FileNotFoundError(
                f"Subtask directory {dir_name} not found"
            )
        return subtask_dir

    def _walk_directory(
        self, root_dir: Path, relative_to: Path
    ) -> list[FileSystemNode]:
        """Recursively build tree structure."""
        nodes: list[FileSystemNode] = []
        items = sorted(root_dir.iterdir(), key=lambda p: (not p.is_dir(), p.name))

        for item in items:
            if item.name.startswith("."):
                continue

            rel_path = str(item.relative_to(relative_to))

            if item.is_file():
                nodes.append(
                    FileNode(name=item.name, path=rel_path, size=item.stat().st_size)
                )
            elif item.is_dir():
                children = self._walk_directory(item, relative_to)
                nodes.append(
                    DirectoryNode(name=item.name, path=rel_path, children=children)
                )

        return nodes

    def _validate_and_resolve_path(
        self, subtask_dir: Path, relative_path: str
    ) -> Path:
        """Validate path and prevent traversal attacks."""
        normalized = Path(relative_path).as_posix()

        # Reject path traversal attempts
        if ".." in normalized or normalized.startswith("/"):
            raise FileNotFoundError("Invalid path")

        file_path = (subtask_dir / normalized).resolve()

        # Ensure resolved path is within subtask_dir
        try:
            file_path.relative_to(subtask_dir.resolve())
        except ValueError:
            raise FileNotFoundError("Path outside subtask directory")

        if not file_path.exists():
            raise FileNotFoundError(f"File not found: {relative_path}")

        return file_path

    def _is_text_file(self, file_path: Path, sample_size: int = 8192) -> bool:
        """Heuristic check if file is text by reading initial bytes."""
        try:
            with open(file_path, "rb") as f:
                chunk = f.read(sample_size)
            if not chunk:
                return True  # empty file is text
            if b"\x00" in chunk:
                return False  # null byte indicates binary
            chunk.decode("utf-8")
            return True
        except (UnicodeDecodeError, OSError):
            return False

    def read_output_file(
        self, env_id: str, dir_name: str, filepath: str
    ) -> tuple[str, bool]:
        """Read output file content. Returns (content, is_text)."""
        subtask_dir = self.get_subtask_dir(env_id, dir_name)
        file_path = self._validate_and_resolve_path(subtask_dir, filepath)

        if not file_path.is_file():
            raise FileNotFoundError(f"{filepath} is not a file")

        if self._is_text_file(file_path):
            content = file_path.read_text(errors="replace")
            return content, True
        return f"[Binary file: {file_path.stat().st_size} bytes]", False

    def get_output_file_path(
        self, env_id: str, dir_name: str, filepath: str
    ) -> Path:
        """Get full path to output file for download."""
        subtask_dir = self.get_subtask_dir(env_id, dir_name)
        return self._validate_and_resolve_path(subtask_dir, filepath)

    def is_structure_file(self, filename: str) -> bool:
        """Check if file is a supported structure format."""
        return Path(filename).suffix.lower() in STRUCTURE_EXTENSIONS

    def read_structure(self, env_id: str, dir_name: str, filepath: str) -> dict:
        """Read structure file and return content with format for 3Dmol.js.

        - cif: returns raw content (3Dmol parses natively with unit cell)
        - extxyz with cell: converts to cif so 3Dmol can render unit cell
        - others: converts to xyz via ASE
        """
        from ase.io import read
        from io import StringIO, BytesIO

        subtask_dir = self.get_subtask_dir(env_id, dir_name)
        file_path = self._validate_and_resolve_path(subtask_dir, filepath)
        suffix = Path(filepath).suffix.lower()

        # 3Dmol supports cif natively with unit cell
        if suffix == ".cif":
            content = file_path.read_text()
            return {"content": content, "format": "cif"}

        # extxyz: convert to cif if has cell, otherwise xyz
        if suffix == ".extxyz":
            atoms = read(str(file_path), index=0)
            if atoms.cell is not None and atoms.cell.rank == 3:
                output = BytesIO()
                atoms.write(output, format="cif")
                return {"content": output.getvalue().decode(), "format": "cif"}
            output = StringIO()
            atoms.write(output, format="xyz")
            return {"content": output.getvalue(), "format": "xyz"}

        # Other formats: convert to xyz via ASE
        atoms = read(str(file_path), index=0)
        output = StringIO()
        atoms.write(output, format="xyz")
        return {"content": output.getvalue(), "format": "xyz"}


info(f"[reader] Initializing fs_reader: wd={cfg.paimon_wd}")
fs_reader = LocalReader(cfg.paimon_wd)
info("[reader] Reader initialized successfully")
