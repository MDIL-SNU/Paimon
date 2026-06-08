from typing import TYPE_CHECKING
import os.path as osp

from paimon.world import get_env, Environment
if TYPE_CHECKING:
    from paimon.models import Plan


def plan_to_mermaid_str(plan: "Plan") -> str:
    """
    Generate a Mermaid diagram representing the plan as a DAG.
    
    Args:
        plan: The Plan object to visualize
        
    Returns:
        str: Mermaid diagram code for the plan
    """
    mermaid_code = []
    mermaid_code.append("graph TD")
    
    # Add all nodes first
    for i, subtask in enumerate(plan.subtasks, 1):
        node_id = f"task{i}"
        # Use actual line break for Mermaid
        label = f"{i}: {subtask.name}<br>({subtask.agent})"
        mermaid_code.append(f"    {node_id}[\"{label}\"]")
    
    # Add edges for dependencies
    task_id_map = {subtask.name: i for i, subtask in enumerate(plan.subtasks, 1)}
    for subtask in plan.subtasks:
        to_node = f"task{task_id_map[subtask.name]}"
        for dep in subtask.dependencies:
            from_node = f"task{task_id_map[dep]}"
            mermaid_code.append(f"    {from_node} --> {to_node}")
    
    return "\n".join(mermaid_code)


def plan_to_llm_friendly_str(plan: "Plan", notate_task: str | None = None) -> str:
    mermaid_code = []
    mermaid_code.append("Plan")
    
    notate_task_found = False

    # Add all nodes first
    for i, subtask in enumerate(plan.subtasks, 1):
        node_id = f"{i}"
        # Use actual line break for Mermaid
        label = f"\"{subtask.name}\" ({subtask.agent})"
        if subtask.name == notate_task:
            notate_task_found = True
            label = label + " **Your task**"
        mermaid_code.append(f"    {node_id}. {label}")

    if notate_task and not notate_task_found:
        raise ValueError(f"{notate_task} not fonud")
    
    # Add edges for dependencies
    dependency_yes_flag = False
    mermaid_code.append("Dependencies")
    task_id_map = {subtask.name: i for i, subtask in enumerate(plan.subtasks, 1)}
    for subtask in plan.subtasks:
        to_node = f"task{task_id_map[subtask.name]}"
        if subtask.name == notate_task:
            to_node = "*" + to_node + "*"
        for dep in subtask.dependencies:
            from_node = f"task{task_id_map[dep]}"
            if subtask.name == dep:
                from_node = "*" + from_node + "*"
            mermaid_code.append(f"    {from_node} --> {to_node}")
            dependency_yes_flag = True
    if not dependency_yes_flag:
        mermaid_code.append("    No dependencies")
    
    return "\n".join(mermaid_code)


def save_plan_viz(plan: "Plan", env: str | Environment) -> None:
    if isinstance(env, str):
        env = get_env(env)

    env.write_file(plan_to_mermaid_str(plan), ".plan.mmd")
