from workflows.events import InputRequiredEvent, HumanResponseEvent
from llama_index.core.workflow import Event, StartEvent, StopEvent
from llama_index.core.agent.workflow.workflow_events import ToolCall, ToolCallResult

from pydantic import BaseModel, Field
from paimon.models import (
    Plan,
    PlanOutline,
    SubtaskWithDir,
    Value,
    CompleteTask,
    SubtaskFailReason,
)


class PaimonStartEvent(StartEvent):
    user_msg: str  # start message
    session_name: str = "Default"  # Identifier for human, saved in .globals.json.
    env_id: str
    user_requested_values: list[Value] = Field(default_factory=list)

    files: list[str] | None = None
    expert_knowledge: str | None = None  # If provided, use.


class SetupEnvironment(Event):
    pass


class StartPlanning(Event):
    task: str
    expert_knowledge: str
    user_requested_values: list[Value]


class StartPlanningWithRetrieval(Event):
    """Planning event that uses retrieval-based expert knowledge."""

    task: str
    files: list[str] | None = None
    expert_knowledge: str | None = None  # If provided, use.
    user_requested_values: list[Value]


class StartPlanningWithRetrievalLoop(Event):
    pass


class ResumeCheckAndStart(Event):
    pass


class PlanStream(Event):
    partial_plan: BaseModel


class PlanningDone(Event):
    plan: Plan | PlanOutline


class StartTasks(Event):
    plan: Plan | PlanOutline
    env_id: str
    task_name: str


class StartSubtask(Event):
    subtask: SubtaskWithDir


class SubtaskRetryCheck(Event):
    subtask: SubtaskWithDir
    reason: SubtaskFailReason


class SubtaskSuccess(Event):
    subtask: SubtaskWithDir


class SubtaskFail(Event):
    subtask: SubtaskWithDir
    reason: SubtaskFailReason
    message: str


# legacy event for static plan
class SubtasksAllDone(Event):
    report: str | CompleteTask


class TaskComplete(Event):
    complete_task: CompleteTask
    tool_id: str


class TaskFail(Event):
    excuse: str
    tool_id: str


class InputRequiredWithStepEvent(InputRequiredEvent):
    prefix: str
    step: str


class HumanResponsedWithStepEvent(HumanResponseEvent):
    response: str
    step: str
    files: list[str] | None = None


class SimpleToolCallResultEvent(Event):
    tool_name: str
    tool_output: str
    tool_id: str
    is_error: bool
