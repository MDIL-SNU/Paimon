"""Paimon interface for terminal users."""

import sys
import textwrap
import time
from enum import IntEnum
from typing import TextIO
from termcolor import colored

import paimon.workflow.events as evt
from paimon.util.artifacts import save_global_run_artifacts
from workflows.handler import WorkflowHandler  # llamaindex
from llama_index.core.workflow import (
    Event,
    InputRequiredEvent,
    HumanResponseEvent,
    StopEvent,
)
from llama_index.core.agent.workflow import (
    AgentStream,
    AgentInput,
    AgentOutput,
    ToolCallResult,
    ToolCall,
)

from paimon.workflow.events import (
    StartTasks,
    StartSubtask,
    SubtaskSuccess,
    SubtaskFail,
    SubtaskRetryCheck,
    SubtasksAllDone,
    PlanStream,
    TaskComplete,
    TaskFail,
)
from paimon import cfg


class ReportLevel(IntEnum):
    USER = 0
    # TASK = 1  # likely to be a user configurable option in future
    DEBUG = 2
    TRACE = 3


# alias for this file
_user = ReportLevel.USER
_debug = ReportLevel.DEBUG
_trace = ReportLevel.TRACE


class BrailleSpinner:
    """Costmetic for loading"""

    def __init__(self, delay: float = 0.08, color: str = "green") -> None:
        """
        Parameters
        ----------
        delay
            delta between ticks in second
        color
            color of spinner
        """

        self.frames = ["⠋", "⠙", "⠹", "⠸", "⠼", "⠴", "⠦", "⠧", "⠇", "⠏"]
        self.delay = delay
        self.color = color
        self._index = 0
        self._last_update = 0

    def spin(self, message: str = "Loading") -> None:
        """Update spinner only if enough time has passed

        Parameters
        ----------
        message
            string displayed after spinner
        """
        current_time = time.time()

        # Only advance the frame if enough time has passed
        if current_time - self._last_update >= self.delay:
            self._index = (self._index + 1) % len(self.frames)
            self._last_update = current_time

        # Always display current frame and message
        frame = self.frames[self._index]
        content = colored(f"{frame} {message}", self.color)
        sys.stdout.write(f"\r{content: <100}")
        sys.stdout.flush()


class CommandLineInterface:
    """Interface to console.

    Example:
        from paimon.interface.cli import CommandLineInterface

        cli = CommandLineInterface()
        response = await cli.attach_workflow(my_workflow.run())
    """

    def __init__(
        self,
        file: TextIO | None = None,
        report_level: ReportLevel | None = None,
        width: int | None = cfg.cli_max_width,
        handle_stream: bool = True,
        prompt: str = "\n>>> ",
    ):
        """
        Parameters
        ----------
        file
            output IO addition to console
        report_level
            default to use DEBUG if cfg.debug else USER
        width
            max line width of terminal. Defaults to cfg.cli_max_width
        handle_stream
            if False, ignore stream outputs
        prompt
            displayed when waiting for user input

        Returns
        -------
        None
        """

        super().__init__()
        self.handler = None
        self.file = file
        self.width = width
        self.report_level = report_level or (
            ReportLevel.DEBUG if cfg.debug else ReportLevel.USER
        )
        self.handle_stream = handle_stream
        self.prompt = prompt

        self._spinner = BrailleSpinner()

    def _color_by_level(self, content: str, level: ReportLevel) -> str:
        if level is ReportLevel.USER:
            return colored(content, "green")
        elif level is ReportLevel.DEBUG:
            return content  # yellow too shiny
        elif level is ReportLevel.TRACE:
            return colored(content, "cyan")
        return content

    def print(
        self,
        content: str,
        end: str = "\n",
        color: str | None = None,
        write_file: bool = True,
        level: ReportLevel = ReportLevel.USER,
    ) -> None:
        """
        print content to user.

        Parameters
        ----------
        content
            str content to print
        end
            if given, ends print with the str
        color
            color of text. Only applied to consol
        write_file
            if False, do not write its output to file.
        level
            report level

        Returns
        -------
        None
        """
        if self.width:
            content = "\n".join(
                textwrap.fill(ll, width=self.width) for ll in content.splitlines()
            )
        if self.report_level.value >= level.value:
            if self.file and write_file:  # file should have no color
                print(content, end=end, flush=True, file=self.file)
            if not color:
                content = self._color_by_level(content, level)
            else:
                content = colored(content, color)
            print(content, end=end, flush=True, file=sys.stdout)

    def bar(self, char: str = "-", **kwargs) -> None:
        """Print bar: "-" * self.width"""
        bar_width = self.width or 79
        self.print(char * bar_width, end="\n", **kwargs)

    def dprint(self, content: str, **kwargs) -> None:
        """Alias to self.print(content, report_level=ReportLevel.DEBUG)"""
        self.print(content, level=ReportLevel.DEBUG, **kwargs)

    def print_header(
        self, header: str, bar_char="-", level=ReportLevel.USER, **kwargs
    ) -> None:
        """Print something like below
        -----------------------------------------
        header_str
        -----------------------------------------
        """
        self.print("\n", level=level, **kwargs)
        self.bar(char=bar_char, level=level)
        self.print(header, level=level, **kwargs)
        self.bar(char=bar_char, level=level)

    async def _save_artifacts(self, last_event) -> None:
        """Save current state to file artifacts."""
        if not self.handler:
            return

        state = await self.handler.ctx.store.get_state()  # type: ignore
        save_global_run_artifacts(
            env=state.env_id,
            metadata={
                "task": state.task,
                "task_name": state.task_name,
                "input_file_paths": getattr(state, "input_file_paths", []),
                "action_count": state.action_count,
                "dumps": state.dumps,
            },
            chat=state.chat_history,
            tokens=state.token_sum,
            last_event=last_event,
            text_artifacts={
                ".plan_prompt.txt": state.plan_prompt_txt,
            },
            json_artifacts={
                ".plan.json": state.current_plan,
                ".completed_subtasks.json": state.completed_subtasks,
            },
        )

    async def handle_event(
        self, ev: Event, handler: WorkflowHandler, state: dict[str, str]
    ) -> None:
        """Primary logic for handling events. Direct use of this function is not
        recommended. Use 'attach_workflow'.

        Parameters
        ----------
        ev
            llama_index Event to handle
        handler
            llama_index handler. Used when 'InputRequiredEvent' is given
        state
            global state dict maintained between handle_event call.

        Returns
        -------
        None
        """
        if self.report_level >= _debug:
            ev_cls_name = ev.__class__.__name__
            if not ev_cls_name.endswith("Stream"):
                self.print_header(ev_cls_name, level=_debug)

        if isinstance(ev, StopEvent):
            self.dprint("Stop workflow")

        elif isinstance(ev, AgentStream):
            # TODO: llama_index/core/agent/workflow/function_agent.py line 58
            # function_agent uses both stream and agent output, thereby we have
            # duplicated outputs here
            if self.handle_stream:
                self.print(ev.delta, end="")

        elif isinstance(ev, AgentInput):
            last_msg = ev.input[-1]
            self.dprint(
                f"(Last chat msg) {last_msg.role.value}:\n {last_msg.content}"
            )
            self.print(f"Full input: {ev.input}", level=_trace)

        elif isinstance(ev, AgentOutput):
            if ev.response.content:  # Final output when wf is done
                self.print(ev.response.content)
            if ev.tool_calls:
                tools = [call.tool_name for call in ev.tool_calls]
                self.dprint(f"Planning to use tools: {tools}")
            self.print(f"Raw output: {ev.raw}", level=_trace)

        elif isinstance(ev, ToolCallResult):  # ToolCall subclass
            self.print(f"Tool end: {ev.tool_name}")
            self.dprint(f"output: {ev.tool_output}")

        elif isinstance(ev, evt.SimpleToolCallResultEvent):  # ToolCall subclass
            self.print(f"Tool end: {ev.tool_name}")
            self.dprint(f"output: {ev.tool_output}")
            self.dprint(f"is_error: {ev.is_error}")

        elif isinstance(ev, ToolCall):
            self.print(f"Tool call: {ev.tool_name}")
            self.print(f"Tool id: {ev.tool_id}", level=_trace)
            self.dprint(f"Tool arguments: {ev.tool_kwargs}")

        elif isinstance(ev, PlanStream):  # custom event of plan.py
            if self.handle_stream:
                last_task = (
                    "..."
                    if "plan_stream_last_task" not in state
                    else state["plan_stream_last_task"]
                )
                try:
                    last_task = ev.partial_plan.subtasks[-1].name  # type: ignore
                    state["plan_stream_last_task"] = last_task
                except (AttributeError, TypeError, IndexError) as e:
                    pass
                msg = f"Planning: {last_task}"
                self._spinner.spin(msg)

        elif isinstance(ev, StartTasks):  # custom event of plan.py
            plan = ev.plan
            env_id = ev.env_id
            task_name = ev.task_name
            self.dprint(f"Env id: {env_id}")
            if hasattr(plan, "subtasks"):
                self.print_header(
                    f"{task_name}, total {len(plan.subtasks)} subtasks"
                )
            else:  # PlanOutline
                self.print_header(f"{task_name}, strategy: {plan}")

        elif isinstance(ev, StartSubtask):  # custom event of plan.py
            subtask = ev.subtask
            self.print(f"The '{subtask.name}' task started with {subtask.agent}")

        elif isinstance(ev, SubtaskSuccess):  # custom event of plan.py
            subtask = ev.subtask
            self.print(f"'{subtask.name}' done.")

        elif isinstance(ev, SubtaskRetryCheck):  # custom event of plan.py
            subtask = ev.subtask
            reason = ev.reason
            self.print(f"'{subtask.name}' failed due to :{reason}, check retry")

        elif isinstance(ev, SubtaskFail):  # custom event of plan.py
            subtask = ev.subtask
            reason = ev.reason
            self.print(f"'{subtask.name}' (total) failed due to :{reason}.")

        elif isinstance(ev, SubtasksAllDone):  # custom event of plan.py
            self.print("All subtasks are done.")

        elif isinstance(ev, TaskComplete):
            self.print("Task success")

        elif isinstance(ev, TaskFail):
            self.print("Task fail")

        elif isinstance(ev, evt.InputRequiredWithStepEvent):
            response = self.human_response(ev.prefix)
            handler.ctx.send_event(
                evt.HumanResponsedWithStepEvent(response=response, step=ev.step)
            )

        elif isinstance(ev, InputRequiredEvent):
            response = self.human_response(ev.prefix)
            handler.ctx.send_event(HumanResponseEvent(response=response))

        else:
            self.dprint(f"Unknown event: {ev}")

        await self._save_artifacts(ev)

    async def attach_workflow(self, handler: WorkflowHandler):
        """Attach workflow handler to process event.

        Parameters
        ----------
        handler
            llamaindex workflow_handler obtained from .run()
        """
        # some internal state for better cosmetics, eg) last_event_name
        state = {}

        current_agent = None
        self.handler = handler

        async for ev in handler.stream_events():
            if (
                hasattr(ev, "current_agent_name")
                and ev.current_agent_name != current_agent
            ):  # TODO: sepcific to llama index multi-agent-workflow
                current_agent = ev.current_agent_name
                self.print_header(f"Agent: {current_agent}", bar_char="=")
            await self.handle_event(ev, handler, state=state)
        ret = await handler
        return ret

    def human_response(self, prefix: str) -> str:
        """Read user input
        Parameters
        ----------
        prefix
            prefix str before self.prompt.

        Returns
        -------
        user_input
            user input string read from console
        """
        try:
            user_input = input(prefix + self.prompt)
        except KeyboardInterrupt:
            return "CLI: Keyboard interruped"  # ?
        except EOFError:
            user_input = ""  # ?
        return user_input
