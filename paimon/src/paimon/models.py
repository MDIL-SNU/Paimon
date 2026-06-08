from typing import Literal, Any, Type
from enum import StrEnum
import re

from pydantic import (
    BaseModel,
    Field,
    ConfigDict,
    create_model,
    field_serializer,
    field_validator,
)

from llama_index.core.llms import ChatMessage, LLM
from llama_index.core.base.llms.types import ToolCallBlock
from llama_index.core.workflow import Context, Workflow
from llama_index.core.memory import BaseMemory

from paimon.token_sum import TokenSum
from paimon import cfg


FF_Literal = Literal["SevenNet", "MACE"]


TaskTypeLiteral = Literal[
    "MD",
    "Relaxation",
    "Analysis",
    "Structure Generation",
    "Preparation",
    "Property Computation",
    "None",
]


class StrictBaseModel(BaseModel):
    # prevent LLM error
    model_config = ConfigDict(extra="forbid")


class NormalizedToolCall(ToolCallBlock):
    """
    Canonical representation of a tool call, with provider-specific inconsistencies
    removed. Initialized from paimon.llm.pipeline module.
    """

    normalized_tool_kwargs: dict[str, Any] = Field(default_factory=dict)


class File(StrictBaseModel):
    filename: str = Field(
        ...,
        description="Literal filename or template using '{rep}'. Path is not allowed.",
    )
    replicates: int = Field(
        1,
        ge=1,
        description="Number of replicas; 1 means a single file with no '{rep}'.",
    )

    def enumerate(self) -> list[str]:
        if self.replicates == 1:
            return [self.filename]
        return [
            self.filename.replace("{rep}", str(i)) for i in range(self.replicates)
        ]


SAFE_PATTERN = re.compile(r"^[a-zA-Z0-9_.-]{1,64}$")


class Value(StrictBaseModel):
    """Numeric values"""

    name: str = Field(
        ...,
        description="Must match ^[a-zA-Z0-9_.-]{1,64}$",
    )
    unit: str = Field(...)

    @field_validator("name")
    def validate_name(cls, v):
        if not SAFE_PATTERN.fullmatch(v):
            raise ValueError(
                "Invalid name: only a-z, A-Z, 0-9, _, ., - allowed, length 1–64."
            )
        return v

    def __str__(self) -> str:
        return f"{self.name} ({self.unit})"


class Subtask(StrictBaseModel):
    """Creates a subtask and executes it using the assigned agent."""

    name: str = Field(
        ...,
        description="Use only letters, numbers, and spaces. For example: Relax perfect BaTiO3 crystal",
    )
    primary_task_type: TaskTypeLiteral = Field(
        ...,
        description="The primary task type, referenced externally using the [[TaskType]] format. The value 'None' is not permitted.",
    )
    secondary_task_type: TaskTypeLiteral = Field(
        default="None", description="An optional secondary task type."
    )
    force_field_family: FF_Literal | None = Field(
        default=None,
        description="Force field family for this subtask (e.g., 'SevenNet', 'MACE'). If not specified, uses workflow-level default.",
    )
    agent: str = Field(...)
    instruction: str = Field(...)
    output_files: list[File] = Field(
        default=[],
        description="A list of output files an agent has to create.",
    )
    output_values: list[Value] = Field(
        default=[],
        description="A list of values an agent has to report",
    )
    dependencies: list[str] = Field(
        default=[],
        description="A list of subtask names that must be completed before this subtask.",
    )
    example_ids: list[str] = Field(
        default=[],
        description="Optional IDs of successful past subtask examples to provide as a hint (format: expert_knowledge/trajectory_id:task_number).",
    )


class SubtaskWithDir(Subtask):
    """Subtask with sub working directory field. Not used by llm"""

    sub_wd: str = Field(..., description="sub working directory name (not abs path)")
    task_number: int
    tool_id: str = ""
    agent_wf: Workflow | None = None
    agent_ctx: Context | None = None
    memory: BaseMemory | None = None
    reported_output_values: dict[str, float] | None = None
    reported_output_files_summary: dict[str, str] | None = None
    reported_file_usage_summary: str | None = None
    reported_message_to_planner: str | None = None
    retry_cnt: int = 0

    class Config:
        arbitrary_types_allowed = True

    @field_serializer("agent_wf")
    def serialize_agent_wf(self, wf: Workflow | None, _info):
        return ""  # wf can not be serialized

    @field_serializer("agent_ctx")
    def serialize_agent_ctx(self, _ctx: Context | None, _info):
        if not _ctx:
            return None
        return _ctx.to_dict()

    @classmethod
    def from_subtask(
        cls, subtask: Subtask, task_number: int, tool_id: str = ""
    ) -> "SubtaskWithDir":
        sub_wd = re.sub(r"[^A-Za-z0-9_-]", "", subtask.name.replace(" ", "_"))
        assert task_number >= 1
        sub_wd = f"{task_number:02d}_{sub_wd}"
        return cls(
            sub_wd=sub_wd,
            task_number=task_number,
            tool_id=tool_id,
            **subtask.model_dump(),
        )


class SubtaskStub(StrictBaseModel):
    name: str = Field(
        ...,
        description="Use only letters, numbers, and spaces. For example: Relax perfect BaTiO3 crystal",
    )
    primary_task_type: TaskTypeLiteral = Field(
        ...,
        description="The primary task type, referenced externally using the [[TaskType]] format. The value 'None' is not permitted.",
    )
    secondary_task_type: TaskTypeLiteral | None = Field(
        default=None, description="An optional secondary task type."
    )
    agent: str = Field(...)
    dependencies: list[str] = Field(
        default=[],
        description="A list of subtask names that must be completed before this subtask.",
    )


class ForceField(StrictBaseModel):
    """Defines a force field for molecular dynamics."""

    model: Literal["SevenNet-0", "SevenNet-MF-ompa"] = Field("SevenNet-0")
    include_dispersion: bool = Field(
        True, description="Include a D3 dispersion correction"
    )

    def __str__(self) -> str:
        return f"model={self.model}, include_dispersion={self.include_dispersion}"


class Plan(StrictBaseModel):
    """A series of subtasks to accomplish an overall task."""

    subtasks: list[Subtask] = Field(...)

    def get_subtask(self, name: str) -> Subtask:
        for st in self.subtasks:
            if st.name == name:
                return st
        raise ValueError(f"No such subtask: {name}")


class PlanOutline(StrictBaseModel):
    """Create or update an outline of your plan. It covers the entire process required to complete the task. It does not execute a subtask but serves to communicate with a user and scaffold a plan."""

    outline: list[SubtaskStub] = Field(...)


class CompleteTask(StrictBaseModel):
    """Complete the task and report the findings to the user."""

    report: str = Field(..., description="A report to the user")
    deliverables: list[File] = Field(..., description="Files to report")

    @classmethod
    def get_model_with_output_values(
        cls, output_values: list[Value], model_name="complete_task"
    ) -> Type["CompleteTask"]:
        addi = {
            ov.name: (float, Field(..., description=f"unit: {ov.unit}"))
            for ov in output_values
        }
        model = create_model(
            model_name, 
            __base__=cls, 
            __doc__=cls.__doc__,
            **addi  # type: ignore
        )
        return model


class AbortTask(StrictBaseModel):
    """Abort the task. Use this when the task is irrelvant to science or impossible to complete."""

    excuse: str = Field(..., description="A report to the user")


# Context models #


class PlanState(BaseModel):
    """Context state of dynamic plan workflow"""

    env_id: str | None = None
    task: str | None = None
    task_name: str | None = None
    user_requested_values: list[Value] = Field(default_factory=list)
    remaining_request_user_input: int = 10**18  # second phase ask user budgetmod

    current_plan: Plan | PlanOutline | None = None
    completed_subtasks: dict[str, SubtaskWithDir] = Field(default_factory=dict)
    remaining_subtasks: dict[str, SubtaskWithDir] = Field(default_factory=dict)

    # planner action counter
    action_count: int = 0
    token_sum: TokenSum | None = None
    plan_prompt_txt: str = ""
    chat_history: list[ChatMessage] = Field(default_factory=list)

    # general tool-specific state data
    auxiliary_data: dict[str, Any] = Field(default_factory=dict)

    dumps: dict[str, Any] = Field(default_factory=dict)


class TaskStatus(StrEnum):
    INIT = "init"
    SUCCESS = "success"
    FAIL = "fail"
    WAIT = "wait"  # agent wait for slurm state change
    HOLD = "hold"  # sign of 'do not end the task' and return err msg to the agent


class SubtaskFailReason(StrEnum):
    ABORT = "abort"  # agent used abort_task tool
    MAX_ITERATION = "max_iteration"  # maximum step reached
    CRITIC_MAX_ITERATION = "critic_max_turns"  # critic max turn reached
    CRITIC_MALICIOUS = "critic_malicious"  # critic judged the agent malicious
    HOOK_FAILURE = "hook_failure"  # pre-init hook failed
    UNKNOWN = "unknown"


class SubtaskAgentState(BaseModel):
    """Context state of paimon agent (with state)
    Initialized before the subtask starts
    """

    env_id: str = ""
    agent_name: str = ""
    system_prompt: str = ""
    sub_wd: str = ""
    instruction: str = ""
    required_output_files: list[File] = Field(default_factory=list)
    required_output_values: list[Value] = Field(default_factory=list)

    task_status: TaskStatus = TaskStatus.INIT

    # Tool specific states
    ## venv
    current_venv: str = "base"
    ## slurm
    slurm_job_id: int | None = None
    max_wait_minutes: int | None = None
    ## general tool-specific state data
    auxiliary_data: dict[str, Any] = Field(default_factory=dict)

    # After subtask completion
    message_to_planner: str | None = None
    file_usage_summary: str | None = None
    output_values: dict[str, float] | None = None
    subtask_fail_reason: SubtaskFailReason = SubtaskFailReason.UNKNOWN


# Critic verdict
class Verdict(StrEnum):
    PASS = "Pass"
    CONCERN = "Concern"
    REJECT = "Reject"
    MALICIOUS = "Malicious"


class CriticalOpinion(BaseModel):
    """Your critical opinion"""

    verdict: Verdict
    why: str
    evidence: str


from llama_index.llms.openai import OpenAIResponses
class CriticCommitteeState(BaseModel):
    """Per subtask critic committee state, initialized from 'criticize_agent'"""

    num_critics: int = cfg.critic_config.num_critics
    maximum_turn: int = cfg.critic_config.max_turns
    need_actions_concern_ratio: float = cfg.critic_config.need_actions_concern_ratio

    current_turn: int = 0
    last_agent_traj_index: int | None = None
    # Below share their indices
    critic_llms: list[OpenAIResponses] | None = None
    critic_memories: list[list[ChatMessage]] | None = None
    last_verdicts: list[Verdict | None] | None = None
    submit_ticket: bool = False


# Experimental todo items
class TodoItem(BaseModel):
    id: int
    desc: str
    tools: str

    def tools_list(self) -> list[str]:
        return [x.strip() for x in self.tools.split(",") if x.strip()]
