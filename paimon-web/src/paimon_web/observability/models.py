from typing import Literal

from pydantic import BaseModel, Field
from llama_index.core.base.llms.types import ChatMessage


RunStatusLiteral = Literal[
    "pending",  # Process launched, not yet connected to socket
    "running",
    "waiting_input",
    "failed",
    "succeeded",
    "interrupted",
    "unknown",
]


class TokenUsageItem(BaseModel):
    name: str
    llm_model: str
    tool_call: list[str]
    input_tokens: int
    reasoning_tokens: int | None = None
    cached_tokens: int
    output_tokens: int
    total_tokens: int
    total_cost: float


class TokenUsageSummary(BaseModel):
    total_input_tokens: int
    total_cached_tokens: int
    total_output_tokens: int
    total_reasoning_tokens: int | None = None
    total_tokens: int
    total_cost: float
    items: list[TokenUsageItem]


class ProcessStatus(BaseModel):
    """Live process status for web-launched runs."""

    is_live: bool = False
    pid: int | None = None
    started_at: str | None = None
    returncode: int | None = None


class LaunchRequest(BaseModel):
    """Full internal request to launch new run."""

    task_name: str = Field(..., description="Task description from user")
    agent: str | None = Field(default=None, description="Agent to use")
    llm: str | None = None
    reasoning: str | None = None
    user_preference: str | None = None
    uploaded_files: list[str] | None = Field(
        default=None, description="Paths to uploaded files in /tmp"
    )


class RunSummary(BaseModel):
    env_id: str = Field(..., description="Environment ID (env_id)")
    task_name: str
    task: str = Field(..., description="Original user request")
    agent: str
    total_cost: float
    status: str = Field(default="unknown", description="run status")
    created_at: str | None = None
    updated_at: str | None = None
    working_dir: str | None = None
    process_status: ProcessStatus | None = None


class SubtaskInfo(BaseModel):
    name: str
    dir_name: str = ""
    primary_task_type: str
    secondary_task_type: str
    agent: str
    instruction: str
    output_files: list[dict]
    output_values: list[dict]
    dependencies: list[str]


class RunConfig(BaseModel):
    task_name: str = ""
    agent: str = ""
    llm: str = ""
    reasoning: str = ""
    workflow_type: str = ""
    user_preference: str = ""
    extra: dict[str, str | int | float | bool] = {}


class RunDetail(BaseModel):
    env_id: str
    task_name: str
    task: str
    agent: str
    status: str = Field(default="unknown", description="run status")
    subtasks: list[SubtaskInfo]
    config: RunConfig
    token_usage: TokenUsageSummary
    completion_report: str | None = None
    completion_data: dict | None = None
    process_status: ProcessStatus | None = None
    has_external_files: bool = False


class FileNode(BaseModel):
    type: Literal["file"] = "file"
    name: str  # filename only
    path: str  # relative path from subtask root (e.g., "subdir/file.txt")
    size: int | None = None


class DirectoryNode(BaseModel):
    type: Literal["directory"] = "directory"
    name: str  # directory name only
    path: str  # relative path from subtask root
    children: list["FileNode | DirectoryNode"] = []


FileSystemNode = FileNode | DirectoryNode


class SubtaskDetail(BaseModel):
    env_id: str
    subtask_name: str
    dir_name: str = ""
    agent_memory: list[ChatMessage] | None = None
    token_usage: TokenUsageSummary | None = None
    critic_token_usage: TokenUsageSummary | None = None
    output_files: list[str] | None = None  # Keep for backward compat
    output_tree: list[FileSystemNode] | None = None  # NEW
    has_critic: bool = False
