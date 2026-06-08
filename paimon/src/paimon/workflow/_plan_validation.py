from paimon.models import Plan, PlanOutline


def verify_and_process_plan(
    plan: Plan | PlanOutline, allowed_agent_names: list[str]
) -> None:
    """
    Verify the plan and return processed subtasks with IDs and step numbers.

    Args:
        plan: The Plan object to verify
        allowed_agent_names: List of allowed agent names

    Returns:
        None

    Raises:
        ValueError: If any validation rule is violated
    """
    subtasks = plan.subtasks if isinstance(plan, Plan) else plan.outline

    # Validate dependency names
    all_subtask_names = {subtask.name for subtask in subtasks}
    for subtask in subtasks:
        for dep in subtask.dependencies:
            if dep not in all_subtask_names:
                raise ValueError(
                    f"Dependency '{dep}' in subtask '{subtask.name}' does not exist"
                )

    # Validate no duplicate output file names
    all_output_files = {}

    if isinstance(plan, Plan):
        for subtask in plan.subtasks:
            for output_file in subtask.output_files:
                if output_file.filename in all_output_files:
                    raise ValueError(
                        f"Duplicate output file name '{output_file.filename}' "
                        f"in subtask '{subtask.name}'"
                    )
                all_output_files[output_file.filename] = output_file

    # Validate agent names
    for subtask in subtasks:
        if subtask.agent not in allowed_agent_names:
            raise ValueError(
                f"Agent name '{subtask.agent}' in subtask "
                f"'{subtask.name}' is not allowed"
            )

    # Validate all outputs are used except for final tasks
    subtasks_with_dependents = set()
    for subtask in subtasks:
        for dep in subtask.dependencies:
            subtasks_with_dependents.add(dep)
