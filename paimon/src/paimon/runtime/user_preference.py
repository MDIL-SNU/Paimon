from copy import deepcopy
from dataclasses import dataclass, field


@dataclass
class UserPreference:
    key: str
    prompt: str
    overrides: dict = field(default_factory=dict)


_PREFERENCES: dict[str, UserPreference] = {
    "computational_researcher": UserPreference(
        key="computational_researcher",
        prompt="""\
The user is a hands-on computational materials scientist. They want strict \
control over the technical details. Before executing tasks, always ask the user \
to verify low-level simulation parameters (e.g., force field selection, \
ensembles, timestep, k-points, convergence criteria). Wait for explicit \
approval on these technical setups before proceeding with the actual calculation.\
""",
        overrides={
            "request_user_input_tool_budget": 10**18,
            "forbid_request_user_input_tool": False,
            "max_action": 1000,
            "stay_alive_after_completion": True,
        },
    ),
    "computational_pi": UserPreference(
        key="computational_pi",
        prompt="""\
The user is a computational principal investigator (PI). They understand the \
overall simulation workflow but do not want to be bothered with low-level \
parameters. Make autonomous decisions on simulation parameters (force fields, \
timesteps, etc.). Only pause to interact and request approval at major \
simulation milestones (e.g., 'Geometry optimization complete. Ready to start \
MD production run'). Report the workflow progress, not the granular parameters.\
""",
        overrides={
            "request_user_input_tool_budget": 5,
            "forbid_request_user_input_tool": False,
            "max_action": 200,
            "stay_alive_after_completion": True,
        },
    ),
    "experimentalist": UserPreference(
        key="experimentalist",
        prompt="""\
The user is an experimentalist with no background in computational simulation. \
They only care about scientific concepts, experimental conditions, and final \
material properties. Make fully autonomous decisions on ALL computational \
methodologies, workflows, and parameters. Never mention simulation jargon \
(e.g., NVT/NPT, force fields, convergence). Do not ask any technical questions \
during the execution phase. Only output the final scientific results based on \
the initial interactive task definition.\
""",
        overrides={
            "request_user_input_tool_budget": 2,
            "forbid_request_user_input_tool": False,
            "max_action": 200,
            "stay_alive_after_completion": True,
        },
    ),
    "autonomous": UserPreference(
        key="autonomous",
        prompt="""\
The user wants a fully autonomous, fire-and-forget experience. Proceed with \
the entire scientific and computational workflow based solely on the initial \
conversation. Make all technical and workflow decisions independently. Do not pause \
to ask for permission, clarification, or review at any point. Complete all \
tasks autonomously and output the final results.
""",
        overrides={
            "request_user_input_tool_budget": 0,
            "forbid_request_user_input_tool": False,
            "max_action": 40,
            "stay_alive_after_completion": False,
        },
    ),
}


def get_preferences() -> dict[str, UserPreference]:
    return deepcopy(_PREFERENCES)
