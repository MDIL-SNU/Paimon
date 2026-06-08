"""Extract reusable few-shot examples from successful trajectories.

Each function uses run_agent_pipeline with structured Pydantic tool output.
All functions are pure -- no side effects beyond LLM calls.
"""

import json
from pathlib import Path

from pydantic import BaseModel
from llama_index.core.llms import ChatMessage
from llama_index.core.memory import ChatMemoryBuffer

from paimon.llm import get_llm, run_agent_pipeline
from paimon.models import Subtask
from paimon.util.chat import chat_hist_to_str
from paimon.util.tool_factory import create_model_tool

from .models import SubtaskExample, TrajectoryExamples


# -- Helpers ------------------------------------------------------------------


def _load_agent_memory(memory_path: Path) -> list[ChatMessage]:
    with open(memory_path) as f:
        dct = json.load(f)
    return ChatMemoryBuffer.from_dict(dct).get_all()


def _list_non_hidden_files(directory: Path) -> list[str]:
    return sorted(
        p.name
        for p in directory.iterdir()
        if p.is_file() and not p.name.startswith(".")
    )


# -- LLM extraction tools (Pydantic models used as structured output) --------


class _SubtaskSummary(BaseModel):
    description: str
    summary: str
    script_filenames: list[str]


# -- Extraction functions -----------------------------------------------------


async def extract_subtask_summary(
    llm_class: str,
    subtask: Subtask,
    chat_messages: list[ChatMessage],
    available_files: list[str],
) -> _SubtaskSummary:
    """Extract purpose, inputs/outputs, trial summary, and working scripts
    from a single subtask.
    """
    traj_str = chat_hist_to_str(chat_messages)
    file_list = "\n".join(f"- {f}" for f in available_files)
    subtask_json = subtask.model_dump_json(indent=2)

    prompt = f"""\
Summarize one subtask from an atomistic simulation workflow.

Omit task-specific values (atom counts, energies, compositions). Keep operationally transferable numbers (timescales, timesteps, convergence durations, cutoffs).

<subtask_definition>
{subtask_json}
</subtask_definition>

<agent_conversation>
{traj_str}
</agent_conversation>

<files_in_directory>
{file_list}
</files_in_directory>

- description: One-sentence high-level summary of what this subtask does.
- summary: A short paragraph in plain sentences (no labels like "Purpose:" or "Why:"). Cover in order: (1) why this subtask exists in the workflow, (2) what it consumes and produces, (3) what was attempted, what failed, and what fixed it. Keep operationally useful numbers (e.g. "500 ps sufficed for density convergence"). Use generic names for task-specific data (e.g. "molecular template xyz files" not specific filenames).
- script_filenames: Script(s) from the file list that contributed to the final output. Exclude scripts that were only part of trial-and-error and superseded. Exclude data/logs/outputs. Empty list if none."""

    llm = get_llm(llm_class, metadata={"role": "episodic"})
    tool = create_model_tool(_SubtaskSummary)
    user_msg = ChatMessage(content=prompt, role="user")

    _, tool_calls, _ = await run_agent_pipeline(
        llm=llm,
        chat_history=[user_msg],
        tools=[tool],
        tool_required=True,
        agent_name="episodic_extract_subtask",
        metadata={"role": "episodic"},
    )
    return _SubtaskSummary(**tool_calls[0].tool_kwargs)


# -- Orchestrator -------------------------------------------------------------


async def extract_trajectory(
    traj_dir: Path,
    llm_class: str = "fast_reasoning",
) -> TrajectoryExamples:
    """Extract all few-shot examples from a single successful trajectory.

    Parameters
    ----------
    traj_dir
        Path to a trajectory directory
    llm_class
        LLM class string (resolved via get_llm per call)

    Returns
    -------
    TrajectoryExamples
    """
    traj_dir = Path(traj_dir)

    with open(traj_dir / ".plan.json") as f:
        plan_data = json.load(f)

    subtask_dicts = plan_data["subtasks"]
    subtask_examples: list[SubtaskExample] = []

    for task_number, st_dict in enumerate(subtask_dicts, start=1):
        subtask = Subtask(**st_dict)

        prefix = f"{task_number:02d}_"
        matches = [
            d
            for d in sorted(traj_dir.iterdir())
            if d.is_dir() and d.name.startswith(prefix)
        ]
        assert len(matches) == 1, (
            f"Expected 1 dir with prefix '{prefix}' in {traj_dir}, got {len(matches)}"
        )
        sub_dir = matches[0]

        memory_path = sub_dir / ".agent_memory.json"
        assert memory_path.exists(), f"Missing agent memory: {memory_path}"

        chat_messages = _load_agent_memory(memory_path)
        available_files = _list_non_hidden_files(sub_dir)

        result = await extract_subtask_summary(
            llm_class, subtask, chat_messages, available_files
        )

        working_scripts: dict[str, str] = {}
        for fname in result.script_filenames:
            script_path = sub_dir / fname
            assert script_path.exists(), (
                f"LLM selected non-existent script: {script_path}"
            )
            working_scripts[fname] = script_path.read_text()

        subtask_examples.append(
            SubtaskExample(
                subtask_name=subtask.name,
                task_number=task_number,
                agent=subtask.agent,
                description=result.description,
                summary=result.summary,
                working_scripts=working_scripts,
            )
        )

    return TrajectoryExamples(subtask_examples=subtask_examples)
