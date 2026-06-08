"""
Collection of complex prompts that include some logic and templates
"""

from paimon import cfg
from paimon.world.environment import Environment
from paimon.workflow.plan_viz import plan_to_llm_friendly_str
from paimon.knowledge import get_knowledge
from paimon.episodic.format import get_fewshot_prompt, get_flow_hints_for_expert
import paimon.models as mdl
import paimon.workflow.events as evt


INIT_USER_PROMPT = """\
<user_message>
{user_query}
</user_message>
<external_files>
{external_files}
</external_files>
"""


def generate_initial_user_prompt(
    env: Environment,
    user_query: str,
    user_requested_values: list[mdl.Value],
    expert_knowledge: str | None = None,
) -> str:
    external_files_ls = env.list_working_directory("external_files")

    prompt = INIT_USER_PROMPT.format(
        user_query=user_query, external_files=external_files_ls
    )

    if user_requested_values:
        prompt += "<user_requests>\n"
        for val in user_requested_values:
            prompt += f"- {val}\n"
        prompt += "</user_requests>"

    if expert_knowledge:
        ek_txt = get_knowledge(expert_knowledge)
        prompt += f"<expert_knowledge>\n{ek_txt}</expert_knowledge>"
        expert_knowledge_name = expert_knowledge.split("/")[-1]
        if cfg.use_episodic and (
            hints := get_flow_hints_for_expert(expert_knowledge_name)
        ):
            prompt += "\n" + hints

    return prompt


AGENT_PROMPT = """\
<plan status>
{plan_status}
</plan status>

<assigned task>
{task_name}

<instruction>
{instruction}
</instruction>

<required output files>
{output_files}
</required output files>

<required output values>
{output_values}
</required output values>

<dependency context>
{dependency_context}
</dependency context>
</assigned task>

<your working directory>
{working_directory}
</your working directory>
"""  # noqa: E501


def generate_agent_prompt(
    plan: mdl.Plan,
    global_task: str,
    target_subtask: mdl.SubtaskWithDir,
    environment: Environment,
    prompt_template: str = AGENT_PROMPT,
) -> str:
    # TODO: currently not used as we're printing working directory instead
    plan_status = plan_to_llm_friendly_str(plan, target_subtask.name)

    required_output_files_block = []
    for output_file in target_subtask.output_files:
        fname = output_file.filename
        if output_file.replicates == 1:
            required_output_files_block.append(f"- `{fname}`")
        else:
            required_output_files_block.append(
                f"- {', '.join(output_file.enumerate())}"
            )
    required_output_files_block = "\n".join(required_output_files_block)
    if not required_output_files_block:
        required_output_files_block = "No required output files"

    required_values_block = []
    for v in target_subtask.output_values:
        required_values_block.append(f'- "{v.name}" ({v.unit})')
    required_values_block = "\n".join(required_values_block)
    if not required_values_block:
        required_values_block = "No required output values"

    dependency_context = ""
    for dep in target_subtask.dependencies:
        dep_subtask = plan.get_subtask(dep)
        assert isinstance(dep_subtask, mdl.SubtaskWithDir)

        ls_output = environment.list_working_directory(dep_subtask.sub_wd)
        fu_summary = dep_subtask.reported_file_usage_summary

        reported_values = ""
        if len(dep_subtask.output_values) > 0:
            assert dep_subtask.reported_output_values
            for ov in dep_subtask.output_values:
                val = dep_subtask.reported_output_values[ov.name]
                reported_values += f"\n- {ov.name} ({ov.unit}) = {val}"
        dependency_context += f"""\
<{dep_subtask.task_number}. {dep}>
<files>
{ls_output}
</files>
<file usage summary>
{fu_summary}

Use `../{dep_subtask.sub_wd}/{{filename}}` to accses
</file usage summary>
<reported values>
{reported_values}
</reported values>
</{dep_subtask.task_number}. {dep}>
"""
    if len(dependency_context) == 0:
        dependency_context = "No dependencies"

    prompt = prompt_template.format(
        plan_status=plan_status,
        task_name=target_subtask.name,
        instruction=target_subtask.instruction,
        output_files=required_output_files_block,
        output_values=required_values_block,
        dependency_context=dependency_context,
        working_directory=environment.list_working_directory(target_subtask.sub_wd),
    )

    if not cfg.use_episodic:
        return prompt

    for eid in target_subtask.example_ids:
        expert_knowledge, rest = eid.split("/", 1)
        trajectory_id, task_number_str = rest.split(":")
        if block := get_fewshot_prompt(
            expert_knowledge, trajectory_id, int(task_number_str)
        ):
            prompt += block

    return prompt


SUBTASK_REPORT_PROMPT = """\
<subtask_report>

<name>{name}</name>
<status>{status}</status>
<agent>{agent}</agent>

<report>
{report}
</report>

<output_values>
{output_values}
</output_values>

<structure_summary>
{structure_summary}
</structure_summary>

</subtask_report>
"""


HARD_FAILURE_COMMON = """\
<name>{name}</name>
<status>Fail - {status}</status>
<agent>{agent}</agent>

<important>
** ask_agent is not available **
The task will be discarded automatically.
</important>

<general guide>
- When generating the next subtask, explicitly instruct the agent on conditions requiring early abort_task instead of forcing continuation.

- Note: Repeated occurrences of these failure with the same kind of subtask states indicate that the current task may be fundamentally incompatible with system capabilities. You may abort the entire task if this pattern persists.
</general guide>
"""  # noqa: E501


HARD_FAILURE_MAX_ITERATION = """\
<detail>
The agent failed to make progress and exhausted its step budget.

Planner guidance:
- Reassess the clarity and granularity of the subtask.
- Consider restructuring or simplifying the subtask.
</detail>
"""


HARD_FAILURE_CRITIC_MAX_ITERATION = """\
<detail>
The agent’s outputs repeatedly failed critic review.

Planner guidance:
- Refine the subtask specification to reduce ambiguity.
- Reassess the clarity and granularity of the subtask.
- Consider restructuring or simplifying the subtask.

<critic message>
{critic_message}
</critic message>
</detail>
"""


HARD_FAILURE_CRITIC_MALICIOUS = """\
<detail>
The critic flagged the output as unsafe or invalid without allowing retries.

Planner guidance:
- Consider abort.
- Treat as a severe failure. Recreate the subtask.
- If needed, shift to a safer fallback approach.

<critic message>
{critic_message}
</critic message>
</detail>
"""


HARD_FAILURE_UNKOWN = """\
This should not happen. Abort your task.
"""


def generate_hard_failure_prompt(event: evt.SubtaskFail) -> str:
    prev_subtask = event.subtask
    reason = event.reason
    head = HARD_FAILURE_COMMON.format(
        name=prev_subtask.name, agent=prev_subtask.agent, status=str(reason)
    )
    if reason == mdl.SubtaskFailReason.MAX_ITERATION:
        body = HARD_FAILURE_MAX_ITERATION
    elif reason == mdl.SubtaskFailReason.CRITIC_MAX_ITERATION:
        body = HARD_FAILURE_CRITIC_MAX_ITERATION.format(critic_message=event.message)
    elif reason == mdl.SubtaskFailReason.CRITIC_MALICIOUS:
        body = HARD_FAILURE_CRITIC_MALICIOUS.format(critic_message=event.message)
    else:
        body = HARD_FAILURE_UNKOWN

    return head + "\n" + body


def generate_subtask_report(
    event: evt.SubtaskSuccess | evt.SubtaskFail,
    prompt_template: str = SUBTASK_REPORT_PROMPT,
) -> str:
    if (
        isinstance(event, evt.SubtaskFail)
        and event.reason != mdl.SubtaskFailReason.ABORT
    ):
        return generate_hard_failure_prompt(event)

    prev_subtask = event.subtask
    if prev_subtask.reported_output_values:
        output_values = ""
        submitted = prev_subtask.reported_output_values
        for val in prev_subtask.output_values:
            output_values += f"\n- {val} = {submitted[val.name]}"
        output_values = output_values.strip()
    else:
        output_values = "No output values were reported."

    if prev_subtask.reported_output_files_summary:
        structure_summary = ""
        for summary in prev_subtask.reported_output_files_summary.values():
            structure_summary += summary + "\n"
        structure_summary = structure_summary.strip()
    else:
        structure_summary = "No structure summary"

    subtask_output = prompt_template.format(
        name=prev_subtask.name,
        agent=prev_subtask.agent,
        status="Success" if isinstance(event, evt.SubtaskSuccess) else "Fail",
        report=prev_subtask.reported_message_to_planner,
        output_values=output_values,
        structure_summary=structure_summary,
    )

    if isinstance(event, evt.SubtaskFail):
        subtask_output += """\
<system>
This failed subtask and its agent will be discarded once you create the next subtask. 
Before that, you can ask to this agent. Only question-asking is permitted with this agent; 
to request additional instructions, a new subtask with must be created.
</system>
"""
        return subtask_output

    return subtask_output
