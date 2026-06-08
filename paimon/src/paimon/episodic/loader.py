"""Load episodic examples via knowledge library.

Folder structure under knowledge/episodic/ encodes the mapping:

    knowledge/episodic/{expert_knowledge}/{trajectory_id}.json
"""

from paimon.knowledge.library import get_knowledge_json, list_knowledge_json

from .models import SubtaskExample, TrajectoryExamples


def load_all_for_expert(
    expert_knowledge: str,
) -> dict[str, TrajectoryExamples]:
    """Load all trajectory examples linked to an expert knowledge.

    Returns
    -------
    dict mapping trajectory_id to TrajectoryExamples.
    Empty dict if no links exist.
    """
    traj_ids = list_knowledge_json(f"episodic/{expert_knowledge}")
    result = {}
    for traj_id in traj_ids:
        raw = get_knowledge_json(f"episodic/{expert_knowledge}/{traj_id}")
        result[traj_id] = TrajectoryExamples.model_validate(raw)
    return result


def resolve_example(
    expert_knowledge: str,
    trajectory_id: str,
    task_number: int,
) -> SubtaskExample:
    """Load a single subtask example by its explicit coordinates."""
    raw = get_knowledge_json(f"episodic/{expert_knowledge}/{trajectory_id}")
    traj = TrajectoryExamples.model_validate(raw)
    for ex in traj.subtask_examples:
        if ex.task_number == task_number:
            return ex
    raise ValueError(
        f"No example with task_number={task_number} in trajectory {trajectory_id}"
    )
