from pydantic import BaseModel


class SubtaskExample(BaseModel):
    subtask_name: str
    task_number: int
    agent: str
    description: str
    summary: str
    working_scripts: dict[str, str]


class TrajectoryExamples(BaseModel):
    subtask_examples: list[SubtaskExample]
