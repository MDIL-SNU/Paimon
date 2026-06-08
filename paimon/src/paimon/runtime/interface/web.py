"""Paimon interface for web-launched runs (non-interactive)."""

import os.path as osp
import asyncio
import json
from asyncio import StreamWriter
from typing import Any

from workflows.handler import WorkflowHandler
from llama_index.core.workflow import Event
from llama_index.core.agent.workflow import (
    AgentInput,
    AgentOutput,
    ToolCall,
    ToolCallResult,
)

import paimon.workflow.events as evt
from paimon.models import PlanState
from paimon.agent.paimon_agent import PaimonAgent
from paimon.world import Environment
from paimon.util.artifacts import save_global_run_artifacts
from paimon.util.log import debug, info


def _build_event_payload(ev: Event, extra: dict[str, Any] | None = None) -> dict:
    """Build event payload from event instance and optional extra data."""
    payload = {"name": type(ev).__name__}
    if extra:
        payload.update(extra)
    return payload


class WebInterface:
    """Interface for web-launched agent runs with socket communication."""

    def __init__(
        self,
        writer: StreamWriter,
        agent: PaimonAgent,
        env: Environment,
        config: dict[str, Any],
    ):
        self.writer = writer
        self.agent = agent
        self.env = env
        self.config = config
        debug("[web] WebInterface initialized")

    async def send_event(self, event_data: dict) -> None:
        """Send JSON event to socket."""
        line = json.dumps(event_data) + "\n"
        self.writer.write(line.encode("utf-8"))
        await self.writer.drain()

    def _save_artifacts(self, last_event: dict) -> None:
        """Save current state to file artifacts."""
        self.env.sys_run(
            command="cp .agent_memory.json ../.chat.json",
            sub_wd="01_working_directory",
        )
        save_global_run_artifacts(
            env=self.env,
            metadata={
                "action_count": 0,  # TODO
                "previous_response_id": getattr(
                    self.agent.llm, "_previous_response_id"
                ),
                **self.config,
            },
            tokens=self.agent.token_used,
            last_event=last_event,
        )

    async def attach_workflow(self, handler: WorkflowHandler) -> Any:
        """Attach workflow handler and process events.

        Parameters
        ----------
        handler
            llama-index workflow handler from .run()

        Returns
        -------
        result
            Workflow result (AgentOutput)
        """
        info("[web] Workflow attached")

        async for ev in handler.stream_events():
            await self._handle_event(ev)

        result = await handler
        info("[web] Workflow completed")
        return result

    async def _handle_event(self, ev: Event) -> None:
        """Handle workflow events."""
        debug(f"[web] Event: {type(ev).__name__}")

        if isinstance(ev, ToolCall):
            extra = {"tool": ev.tool_name, "tool_kwargs": ev.tool_kwargs}
            await self._send_and_save(ev, extra)

        elif isinstance(ev, ToolCallResult):
            extra = {"tool": ev.tool_name, "success": not ev.tool_output.is_error}
            await self._send_and_save(ev, extra)

        elif isinstance(ev, AgentOutput):
            content = ev.response.content or ""
            has_tool_calls = len(ev.tool_calls) > 0
            extra = {
                "content": content if not has_tool_calls else "",
                "has_tool_calls": has_tool_calls,
            }
            payload = _build_event_payload(ev, extra)
            await self.send_event({"type": "event", **payload})
            # Only save artifacts for final output (no more tool calls)
            if not has_tool_calls:
                self._save_artifacts(_build_event_payload(ev, {"content": content}))

        elif isinstance(ev, AgentInput):
            extra = {"agent": ev.current_agent_name}
            payload = _build_event_payload(ev, extra)
            await self.send_event({"type": "event", **payload})

    async def _send_and_save(
        self, ev: Event, extra: dict[str, Any] | None = None
    ) -> None:
        """Send event to socket and save artifacts."""
        payload = _build_event_payload(ev, extra)
        await self.send_event({"type": "event", **payload})
        self._save_artifacts(payload)


class MultiAgentWebInterface:
    """Interface for multi-agent workflows with user interaction support."""

    def __init__(
        self,
        socket_path: str,
        env: Environment,
        config: dict[str, Any],
    ):
        self.socket_path = socket_path
        self.env = env
        self.config = config
        self.writer = None
        self.reader = None
        self.handler = None
        debug("[web] MultiAgentWebInterface initialized")

    async def send_event(self, event_data: dict) -> None:
        """Send JSON event to socket."""
        assert self.reader is not None and self.writer is not None
        line = json.dumps(event_data) + "\n"
        self.writer.write(line.encode("utf-8"))
        await self.writer.drain()

    async def _open_connection(self):
        assert self.reader is None and self.writer is None
        debug("[web interface] Open socket.")
        reader, writer = await asyncio.open_unix_connection(self.socket_path)
        self.reader = reader
        self.writer = writer

    async def _close_connection(self):
        assert self.reader is not None and self.writer is not None
        debug("[web interface] Close socket.")
        self.writer.close()
        await self.writer.wait_closed()
        self.reader = None
        self.writer = None

    async def get_input(self) -> tuple[str, list[str]]:
        """Read user input from socket. Return tuple of user message and list of user
        uploaded file names if present.
        """
        if not self.reader:
            await self._open_connection()
        assert self.reader is not None and self.writer is not None

        data = await self.reader.read()
        save_global_run_artifacts(
            env=self.env,
            last_event={"name": "HumanResponseEvent"},
        )

        request = json.loads(data.decode("utf-8"))
        debug(f"[web] {self.env.id}: Received: {request}")
        assert "type" in request, "invalid payload"

        assert request["type"] == "user_msg"
        if "user_msg" not in request or not isinstance(request["user_msg"], str):
            raise ValueError("invalid payload")
        user_msg = request["user_msg"]

        # Transfer mid-chat uploaded files
        mid_chat_files = request.get("files", [])
        user_provided_fnames = []
        for file_path in mid_chat_files:
            info(f"[interface] {self.env.id}: Uploading mid-chat file: {file_path}")
            await self.env.put(file_path, sub_wd="external_files")
            fname = osp.basename(file_path)
            user_provided_fnames.append(fname)

        return user_msg, user_provided_fnames

    async def _save_artifacts(self, last_event: Any) -> None:
        """Save current state to file artifacts."""
        if not self.handler:
            debug("[interface] save artifacts is called without handler")
            return

        state: PlanState = await self.handler.ctx.store.get_state()  # type: ignore
        save_global_run_artifacts(
            env=self.env,
            metadata={
                "task": state.task,
                "task_name": state.task_name,
                "action_count": state.action_count,
                "dumps": state.dumps,
                "agent": "Orchestrator",
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

    async def attach_workflow(self, handler: WorkflowHandler) -> Any:
        """Attach workflow and process events until InputRequired or completion.

        Returns
        -------
        result : str | Any
            "waiting_for_user" if paused for input, else workflow result
        ctx : Context | None
            Workflow context for injecting HumanResponseEvent
        """
        info("[web] Multi-agent workflow attached")
        self.handler = handler

        async for ev in self.handler.stream_events():
            await self._handle_event(ev)

        result = await self.handler
        info("[web] Multi-agent workflow completed")
        return result

    async def _send_and_save(
        self, ev: Event, extra: dict[str, Any] | None = None
    ) -> None:
        """Send event to socket and save artifacts."""
        payload = _build_event_payload(ev, extra)
        await self.send_event({"type": "event", **payload})
        await self._save_artifacts(payload)

    async def _handle_event(self, ev: Event) -> None:
        """Handle workflow events."""
        # Event is consumed by: paimon_web/web/static/js/chat.js
        debug(f"[web] Multi-agent event: {type(ev).__name__}")

        if isinstance(ev, evt.InputRequiredWithStepEvent):
            assert self.handler
            extra = {"question": ev.prefix}
            await self._send_and_save(ev, extra)
            await self.send_event({"type": "done"})
            await self._close_connection()
            debug("[web] input required event artifacts saved")
            user_msg, user_files = await self.get_input()
            await self._save_artifacts(
                {"name": "user_msg", "content": user_msg, "files": user_files}
            )
            resp_ev = evt.HumanResponsedWithStepEvent(
                response=user_msg, step=ev.step, files=user_files
            )
            debug("[web] Send human response event")
            self.handler.ctx.send_event(resp_ev)
            return

        elif isinstance(ev, ToolCall):
            ev.tool_kwargs.pop("ctx", None)
            extra = {"tool": ev.tool_name, "tool_kwargs": ev.tool_kwargs}
            await self._send_and_save(ev, extra)

        elif isinstance(ev, evt.SimpleToolCallResultEvent):
            extra = {"tool": ev.tool_name, "success": not ev.is_error}
            await self._send_and_save(ev, extra)

        elif isinstance(ev, evt.StartTasks):
            extra = {"env_id": ev.env_id, "task_name": ev.task_name}
            await self._send_and_save(ev, extra)

        elif isinstance(ev, evt.StartSubtask):
            extra = {"subtask_name": ev.subtask.name, "agent": ev.subtask.agent}
            await self._send_and_save(ev, extra)

        elif isinstance(ev, evt.SubtaskSuccess):
            extra = {"subtask_name": ev.subtask.name}
            await self._send_and_save(ev, extra)

        elif isinstance(ev, evt.SubtaskFail):
            extra = {"subtask_name": ev.subtask.name, "reason": ev.message}
            await self._send_and_save(ev, extra)

        elif isinstance(ev, evt.TaskComplete):
            extra = {"report": ev.complete_task.report}
            await self._send_and_save(ev, extra)

        elif isinstance(ev, evt.TaskFail):
            extra = {"excuse": ev.excuse}
            await self._send_and_save(ev, extra)

