import os
import os.path as osp
import asyncio
import traceback

from llama_index.core.llms import ChatMessage
from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAIResponses
from llama_index.core.program.function_program import get_function_tool
from pydantic import BaseModel, Field

from paimon.models import SubtaskAgentState, TaskStatus
from paimon.agent import ToolSystemError
from paimon.util.log import debug, debug_var, debug_assert
from paimon.util.context import get_env_with_sub_wd


ASE_RELAXATION_STAGE_EXTRACTOR_SYSTEM_PROMPT = """\
You are a strict information extractor. You will be given an **INPUT** block containing a task description for a computational workflow. Your job is to read that INPUT **verbatim** and answer three specific questions with a structured JSON object. Do **not** infer beyond the text; base your answers only on what's explicitly stated.

## Questions to Answer

1. **Q1**: Is the agent, when executing this instruction, required to perform a (loose) structural relaxation/minimization as an actual step of this task? Judge this by what the instruction directs the agent to DO in this task, not by what it mentions, describes, or references. Exclude relaxation/minimization that is explicitly directed at individual Li, PF6, DEC, DMC, PC, or LiPF6 species/components.
   * Output a boolean: `true` or `false`.

2. **Q2**: If and only if Q1 is `true`: Is the **relaxation criterion** specified as **“maximum force < 2.0 eV/Å (Angstrom)”** (equivalently “maximum force below 2.0 eV/Å”)?
   * Output a boolean: `true` or `false`.
   Additional rule: If Q1 is false, then Q2 must be false.

3. **Q3**: If and only if Q2 is `true`: What is the **name of the output structure file** that should be produced by the loose relaxation?
   * Output the exact filename as a string. If not applicable, output "Not applicable".

## Extraction Rules

* Treat “loose relaxation” as synonymous with “loose geometry relaxation” / “loose relaxation of packed structure”.
* For Q2, accept phrasings like “maximum force < 2.0 eV/Å”, “maximum force below 2.0 eV/Å”, or “max force < 2.0 eV/Å”. Minor wording differences are okay if the numeric threshold and units match.
* Units: Å == Angstrom. The threshold must be **2.0 eV/Å**. If a different value or no value is given, Q2 must be `false`.
* For Q3, return **the exact filename** as written in the INPUT (e.g., `relaxed_loose.extxyz`). Do not invent paths or alter case. If Q2 is `false`, return `"Not applicable"`.

## Example

Given an INPUT that states:

* “Perform a loose geometry relaxation… Relax until maximum force < 2.0 eV/Å… Output relaxed structure as relaxed_loose.extxyz.”

Your answers should look like:
```answers
{
  "q1": true,
  "q2": true,
  "q3": "relaxed_loose.extxyz"
}
```

Now read the INPUT and produce the output.
"""  # noqa: E501


class ASE_relaxation_extractor_schema(BaseModel):
    """Answers"""

    q1: bool = Field(..., description="Q1. loose relaxation task")
    q2: bool = Field(False, description="Q2. maximum force below 2.0 eV/Angstrom")
    q3: str = Field(
        "Not applicable", description="Q3. filename of the relaxed structure file"
    )


_ase_relaxation_extractor_tool = get_function_tool(ASE_relaxation_extractor_schema)
_ase_relaxation_extractor_tool.metadata.name = "Answers"

ase_relaxation_extractor = OpenAIResponses(
    model="gpt-5-mini",
    store=True,
    track_previous_responses=False,
    reasoning_options={"effort": "medium"},
)

# Src reference files pth prefix
BENCHMARK_DIR = os.getenv("BENCHMARK_DIR")
if BENCHMARK_DIR is None:
    raise RuntimeError("BENCHMARK_DIR environment variable is required")

# global state variable; this is okay as we launch different python process
# for each tast cases every time
ASE_STAGE_CHECKED = False

MOLECULE_NAME = os.getenv("SKIP_SLURM_MOLECULE", "DEC")
PATH_PRE_MOL = osp.join(BENCHMARK_DIR, f"{MOLECULE_NAME}_LAMMPS")


# TODO: add DMC, PC composition checks
# TODO: decide whether/how to add cell size check
async def _ase_relax_stage_check(
    ctx: Context, agent_state: SubtaskAgentState
) -> None:
    """Given context, check whether this is the ASE relaxation stage.
    and if it does, perform checks whether the stage is done correctly
    and if it is done correctly, copy-paste reference structures it became
    consistent with to later lammps trajectory files
    """
    global ASE_STAGE_CHECKED

    env, sub_wd, venv_name = await get_env_with_sub_wd(ctx)

    if ASE_STAGE_CHECKED:
        return

    if agent_state.agent_name == "LAMMPS agent" and not ASE_STAGE_CHECKED:
        content = "[ASE_RELAX_CHECK] relax in lammps"
        env.write_file(content, "./ASE_CHECK_FAIL")
        raise ToolSystemError(f"[ASE_RELAX_JUDGE] ASE_CHECK_FAIL {content}")

    input_instruction = "INPUT:\n" + (await ctx.store.get("user_msg_str"))

    llm_input: list[ChatMessage] = [
        ChatMessage(
            role="system", content=ASE_RELAXATION_STAGE_EXTRACTOR_SYSTEM_PROMPT
        ),
        ChatMessage(role="user", content=input_instruction),
    ]

    resp = ase_relaxation_extractor.chat_with_tools(
        [_ase_relaxation_extractor_tool],
        chat_history=llm_input,
        allow_parallel_tool_calls=False,
        tool_required=True,
        metadata={
            "role": "ase_step_judge",
            "env_id": env.id,
        },
    )
    tool_calls = ase_relaxation_extractor.get_tool_calls_from_response(
        resp, error_on_no_tool_call=True
    )
    assert len(tool_calls) == 1, "more than one tool call"
    assert tool_calls[0].tool_name == "Answers"
    r: ASE_relaxation_extractor_schema = _ase_relaxation_extractor_tool.fn(
        **tool_calls[0].tool_kwargs
    )

    env.write_json(r, filename=".ase_relaxation_extractor.json", sub_wd=sub_wd)

    if not r.q1:
        if r.q2:  # violates rule
            env.write_file(
                "[ASE_RELAX_JUDGE] [POOR_JUDGE] q1 F but q2 T", "./POOR_JUDGE"
            )
            raise ToolSystemError("[POOR_LLM_JUDGE]")
        return

    if r.q1 and not r.q2:
        content = "[ASE_RELAX_CHECK] not 2.0 relax"
        env.write_file(content, "./ASE_CHECK_FAIL")
        raise ToolSystemError(f"[ASE_RELAX_JUDGE] ASE_CHECK_FAIL {content}")

    # logically always true
    assert r.q1 and r.q2, "???"

    # scenario: planner did not specified output file but ASE did it correctly
    # => can not be found from the instruction
    # I suspect it would be extremly rarely happens, but if it happens, we could
    # feed all the scratchpad as an input to the judge
    if not env.file_exists(r.q3, sub_wd=sub_wd):
        env.write_file("[ASE_RELAX_JUDGE] [EDGE_CASE] file not exist", "./EDGE_CASE")
        raise ToolSystemError("[EDGE_CASE]")

    if not r.q3.endswith(".extxyz"):
        env.write_file("[ASE_RELAX_JUDGE] not_extxyz", "./EDGE_CASE")
        raise ToolSystemError("[ASE_RELAX_JUDGE] EDGE_CASE (not_extxyz)")

    target_fname = r.q3

    # checklist
    # 1. all PBC true
    # 2. composition (based on molecule type)
    # 3. fmax < 2.0 (sevennet-0 d3)
    # 4. li-li > 2.7 Angstrom (originally 3.0, but it is after the relaxation)
    # 5. cell size (how?)
    debug(MOLECULE_NAME)
    _check_fn_body = """\
import numpy as np
from ase.io import read
from ase.geometry import get_distances
from sevenn.calculator import SevenNetD3Calculator
from collections import Counter

BENCHMARK_DIR = k["BENCHMARK_DIR"]
filepath = k["filepath"]
molecule = k["molecule"]

ref_atoms = read(f"{BENCHMARK_DIR}/{molecule}_LAMMPS/_structure_step1/ase_relaxed.extxyz")
ref_composition = dict(Counter(ref_atoms.get_chemical_symbols()))
ret = []

fmax_threshold = 2.0
min_li_li = 2.7

try:
    a = read(filepath)
except Exception as e:
    return f"[FAIL] 0: cannot read file ({e})"
if isinstance(a, list):
    a = a[0]

if not (a.pbc is not None and np.all(a.pbc)):  # 1
    ret.append(f"[FAIL] 1: PBC must be True in all three directions.")
comp = {}
for s in a.get_chemical_symbols():
    comp[s] = comp.get(s, 0) + 1
if comp != ref_composition:  # 2
    ret.append(f"[FAIL] 2: Composition mismatch. Actual {comp} != Ref {ref_composition}.")
b = a.copy()
b.calc = SevenNetD3Calculator("7net-0", device="cuda")
fmax = (
    float(np.max(np.linalg.norm(b.get_forces(apply_constraint=False), axis=1)))
    if len(b) > 0
    else 0.0
)
print(fmax)
if not np.isfinite(fmax) or fmax >= fmax_threshold:  # 3
    ret.append(f"[FAIL] 3: fmax={fmax:.3f} eV/Å ≥ {fmax_threshold:.3f} eV/Å.")
li = [i for i, s in enumerate(a.get_chemical_symbols()) if s == "Li"]
if len(li) >= 2:  # 4
    mins = []
    for i, ii in enumerate(li[:-1]):
        jj = li[i + 1 :]
        _, d = get_distances(
            a.positions[ii], a.positions[jj], cell=a.cell, pbc=a.pbc
        )
        mins.append(np.min(d))
    m = float(np.min(mins))
    if m <= min_li_li:
        ret.append(f"[FAIL] 4: min(Li–Li)={m:.3f} Å ≤ {min_li_li:.3f} Å.")
else:
    ret.append(f"[FAIL] X: Number of Li < 2")

L = a.cell.lengths()  # 5
L_ref = ref_atoms.cell.lengths()

tol = 0.10  # 10% tolerance

rel_errors = np.abs(L - L_ref) / L_ref
if np.any(rel_errors > tol):
    ret.append(f"[FAIL] 5: Cell length mismatch. Actual {L} vs Ref {L_ref}, rel_error={rel_errors}.")

if len(ret) == 0:
    return "PASS"
else:
    return "&&".join(ret)
"""
    try:
        ret = await env.python_call(
            _check_fn_body,
            timeout=360,
            sub_wd=sub_wd,
            func_kwargs={"BENCHMARK_DIR": BENCHMARK_DIR, "filepath": target_fname, "molecule": MOLECULE_NAME},
            venv_name=venv_name,
        )
    except Exception as e:
        content = "[ASE_RELAX_CHECK][BUG_IN_SCRIPT]" + str(e)
        env.write_file(content, "./BUG")
        raise ToolSystemError(f"[ASE_RELAX_CHECK][BUG_IN_SCRIPT] {e}")

    assert isinstance(ret, str)
    if ret != "PASS":
        content = "[ASE_RELAX_CHECK] " + ret
        env.write_file(content, "./ASE_CHECK_FAIL")
        raise ToolSystemError(f"[ASE_RELAX_JUDGE] ASE_CHECK_FAIL {content}")

    # Check all passed - copy paste necessary files
    env.sys_run(f"mv {target_fname} .ori.extxyz", sub_wd=sub_wd)
    env.sys_run(
        f"cp {PATH_PRE_MOL}/_structure_step1/ase_relaxed.extxyz {target_fname}",
        sub_wd=sub_wd,
    )
    ASE_STAGE_CHECKED = True
    return


TARGET_TOOL = "complete_task"
def wrap_tool(tool: FunctionTool, agent_name: str):
    if not (
        tool.metadata.get_name() == "complete_task"
        and agent_name in ("ASE agent", "LAMMPS agent")
    ):
        return tool

    f = tool.async_fn

    async def wrapper(*args, **kwargs):
        assert "ctx" in kwargs

        ret = await f(*args, **kwargs)

        ctx: Context = kwargs["ctx"]
        agent_state: SubtaskAgentState = await ctx.store.get("agent_state")

        if agent_state.task_status == TaskStatus.SUCCESS:
            try:
                await _ase_relax_stage_check(ctx, agent_state)
            except ToolSystemError:
                raise
            except asyncio.CancelledError:
                raise
            except Exception as e:
                bt = traceback.format_exc()
                debug(f"[ASE_RELAX_JUDGE][BUG][UNKNOWN] TRACEBACK:\n {bt}")
                raise ToolSystemError(f"[ASE_RELAX_JUDGE][BUG][UNKNOWN] {e}") from e

        return ret

    debug("[LE_ASE_GATE] applied")
    tool._async_fn = wrapper
    return tool
