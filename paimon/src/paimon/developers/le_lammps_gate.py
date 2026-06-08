# -*- coding: utf-8 -*-
import os
import re
import json
import asyncio
import traceback
import os.path as osp
from pathlib import Path

from pydantic import BaseModel, Field
from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool
from llama_index.llms.openai import OpenAIResponses
from llama_index.core.llms import ChatMessage
from llama_index.core.program.function_program import get_function_tool

from paimon.models import TaskStatus
from paimon.util.context import get_env_with_sub_wd
from paimon.util.log import debug
from paimon.agent import ToolSystemError

TARGET_TOOL = "submit_and_wait"

class boolTFresult(BaseModel):
    """Answer TF question with bool"""

    TF_result: bool = Field(
        ...,
        description="Answer to the question, True or False.",
    )


STEP_PROMPT = """
You are a information extractor for LAMMPS script. You will be given a LAMMPS script.
From the the LAMMPS script, name the expected output files in order.

1. The lammps log file
2. The DCD trajectory file
3. The final restart file
4. The write_data file

- If there is no explicit log file name, use the default name "log.lammps".
- If the lammps script does not include any trajectory dumps, output "NONE" as the DCD filename.
- The restart file must end with .restart.
- For the write_data file, find the last `write_data` command in the LAMMPS script. If present, output its filename. If there is no `write_data` command, output "NONE".
- Output must be exactly 4 words, e.g. prod.log production.dcd product.restart final.lammps-data
- Any deviation will be treated as an error.
"""

# This module is launched per task case, so a module-level counter is sufficient.
SUBMIT_AND_WAIT_CALL_COUNT = 0


async def run_with_TFtool(
    llm,
    instruction,
    mass_section,
    question,
    inresult,
    question_id,
    lammps_step,
    env_id,
) -> bool:
    TF_tool = get_function_tool(boolTFresult)

    # TF criteria prompt
    system_prompt = """
<role>
You are an LLM judge who judges a LAMMPS script written by a LAMMPS agent for atomistic simulations.
You will judge the LAMMPS script based on the paired instruction and the rubric item.
You must judge solely on the explicitly written information in the paired instruction and the LAMMPS script.
You must answer the rubric item with True or False using only the provided explicit information.
</role>

<interpretation rules>
1. In given rubric item, angle-bracket placeholders (<...>) represent variable identifiers. Their names vary in the target lammps script, but they must be used consistently within the lammps script.

2. Similarly, arbitrary names or values are acceptable for identifiers such as <ID>, <dump_ID>, <group_ID>, <parameter>, and <random_seed>.

3. Numeric values in the rubric should be interpreted by their value, not exact text match. For example, 298, 298.0, or an equivalent variable/expression evaluating to 298 are considered identical.

4. Command keywords and styles (e.g., `nvt` from `fix <ID> <group_ID> nvt`, `hybrid/overlay` from `pair_style hybrid/overlay`) must match the rubric exactly.

5. Output filenames (e.g. {name}.dcd, {name}.dat) must match if it is specified in the rubric. However, for thermo or logging outputs, do not enforce a literal `.h5` filename during this judge step, because such filenames may be produced later by post-processing. Still enforce the required logging behavior itself, such as whether logging exists and its requested interval.

6. You should judge the rubric item based on both the paired instruction and the LAMMPS script whenever the instruction contains requirements, targets, constraints, requested outputs, or task-specific conditions relevant to the rubric item.

7. If the paired instruction specifies physical parameters or target values such as atom masses, isotope choices, runtime, timestep, temperature, pressure, dump interval, logging interval, ensemble, trajectory output requirements, or restart/output file requirements, treat the instruction-specified values as the reference for judging the rubric item. For thermo or logging outputs, enforce the requested logging/output behavior itself, such as whether it is present and its requested interval, but do not fail the script only because an expected `.h5` filename is not written at judge time.

8. When a rubric item concerns any mass-related requirement, you must pay attention to the mass information explicitly given with the paired instruction and compare it against the script.

9. The LAMMPS script should be interpreted according to the parsing rules of LAMMPS, not its literal textual representation. For example,
```
timestep 0.001
```
is equivalent to
```
variable dt equal 0.001
timestep ${dt}
```
In this case, if the rubric asks whether the script uses `timestep 0.001`, both cases should be considered True, as they are essentially equivalent.
</interpretation rules>

<LAMMPS parsing rules>
Prefixes (`c_`, `v_`, `f_`) reference values produced by computes, variables, or fixes and force runtime evaluation, not input-parse substitution.

Prefix Semantics
- `c_ID`: output of `compute ID ...`
- `v_name`: output of `variable name ...`
- `f_ID`: output of `fix ID ...`

These are valid **only after** the corresponding compute/variable/fix is defined.

Evaluation Model
- `${var}`: expanded once at input read time. The braces '{}' can be omitted only if the variable name is a single character.
- `v_`, `c_`, `f_`: evaluated when the command is executed (e.g. every timestep, every thermo output)

Scalars, Vectors, Indexing
If the referenced object returns a scalar:
  ```
  c_ID
  v_name
  f_ID
  ```
If it returns a vector or array, index explicitly:
  ```
  c_ID[i]
  v_name[i]
  f_ID[i]
  ```

Typical Usage

- Thermo output:
  ```
  thermo_style custom step temp c_myCompute v_myVar
  ```
- Dump per-atom data:
  ```
  dump 1 all custom 100 dump.lammpstrj id x y z c_atomComp[1] v_atomVar
  ```

Immediate Expressions ($(...))

If $ is followed by parentheses

$(expression)

the text inside the parentheses is evaluated immediately as an equal-style variable expression.
This allows numeric formulas to be used directly in an input script without defining a named variable.

For example, the following three commands:
```
variable X equal (xlo+xhi)/2+sqrt(v_area)
region 1 block $X 2 INF INF EDGE EDGE
variable X delete
```
can be replaced by:
```
region 1 block $((xlo+xhi)/2+sqrt(v_area)) 2 INF INF EDGE EDGE
```

10. About units in LAMMPS scripts: `units metal` uses bar for pressure and ps for timestep.
"""  # noqa: E501
    user_prompt = f"""
<instruction>
{instruction}
</instruction>

<mass information>
Atom order and Mass are given below if any lammps-data file have existed in the current working directory.
Atomic order and mass from the lammps-data file: 
{mass_section}
</mass information>

<LAMMPS script>
{inresult}
</LAMMPS script>

<question>
{question}
</question>

Answer the question based on instruction, mass information, and LAMMPS script.
"""
    chat_history: list[ChatMessage] = [
        ChatMessage(role="system", content=system_prompt),
        ChatMessage(role="user", content=user_prompt),
    ]

    resp = await llm.achat_with_tools(
        [TF_tool],
        chat_history=chat_history,
        tool_required=True,
        metadata={
            "role": "lammps_judge",
            "env_id": env_id,
            "lammps_step": lammps_step,
            "question_id": question_id,
        },
    )
    tool_calls = llm.get_tool_calls_from_response(resp, error_on_no_tool_calls=False)
    assert len(tool_calls) == 1
    ret = boolTFresult(**tool_calls[0].tool_kwargs)
    return bool(ret.TF_result)


def step3_extra_checks(lammps_data_in_script, env, sub_wd, path_pre_step):
    target_frame = f"{path_pre_step}/target_frame.lammps-data"
    python_script = f"""
from ase.io import read

res_atoms = read("{lammps_data_in_script}", format = "lammps-data")
ref_atoms = read("{target_frame}", format = "lammps-data")
res_V = res_atoms.get_volume()
ref_V = ref_atoms.get_volume()
res_rho = 1 / res_V
ref_rho = 1 / ref_V
diff = abs(res_rho - ref_rho) / ref_rho * 100
print(f"res_V : {{res_V}}, ref_V : {{ref_V}}")
print(diff)
target_frame_true = diff <= 0.001
print(target_frame_true)
    """
    env.write_file(
        content=python_script, remote_path="_tmp_test_density.py", sub_wd=sub_wd
    )
    python_result = env.sys_run(
        "python -u _tmp_test_density.py",
        sub_wd=sub_wd,
    )
    env.sys_run(
        "rm _tmp_test_density.py",
        sub_wd=sub_wd,
    )
    last_line = python_result.strip().splitlines()[-1]
    target_frame_true = last_line.strip() == "True"
    env.write_file(content=python_result, remote_path=".target_frame.log", sub_wd=sub_wd)
    return target_frame_true


async def LLM_judge(ctx, script_path):
    # script_path is the script_path of the submit_and_wait tool.
    path_pre = "/data2_1/team_llm/cyw/"

    #scratchpad: list[ChatMessage] = await ctx.store.get("scratchpad", default=[])
    env, sub_wd, _ = await get_env_with_sub_wd(ctx)
    instruction = await ctx.store.get("user_msg_str")

    # It is "potentillay" a problem as it can throw a file not found error but I'll skip handling this case as the error is never observed and non-trivial to handle the scenario.
    job_script_content = env.read_file(script_path, sub_wd)

    lammps_fname = job_script_content.split()[-1]
    lammps_script = env.read_file(lammps_fname, sub_wd)
    env.write_file(lammps_script, ".lammps_script_judged", sub_wd)

    lammps_data_in_script = None

    read_data_match = re.search(
        r"^\s*read_data\s+(\S+)", lammps_script, re.MULTILINE
    )
    if read_data_match:
        lammps_data_in_script = read_data_match.group(1).strip()
        mass_section = env.sys_run(
            f"awk '/^Masses$/{{f=1}} /^Atoms[[:space:]]/{{exit}} f' {lammps_data_in_script}",
            sub_wd=sub_wd,
        )
    else:
        mass_section = "No lammps-data file in this directory, possibly starting from restart file"

    if SUBMIT_AND_WAIT_CALL_COUNT == 1:
        lammps_step = "step1"
    elif SUBMIT_AND_WAIT_CALL_COUNT == 2:
        lammps_step = "step2rd" if lammps_data_in_script is not None else "step2rr"
    elif SUBMIT_AND_WAIT_CALL_COUNT == 3:
        lammps_step = "step3"
    else:
        env.write_file(content="", remote_path="SUBMIT_AND_WAIT_FOUR_TIMES")
        raise ToolSystemError("[SUBMIT_AND_WAIT_FOUR_TIMES]")

    # LLM definition
    llm = OpenAIResponses(
        model="gpt-5-mini",
        store=True,
        track_previous_responses=False,
        reasoning_options={"effort": "medium"},
    )

    user_prompt_step_classification = f"""
<instruction>
{instruction}
</instruction>
<LAMMPS script>
{lammps_script}
</LAMMPS script>
"""
    # classify step and output file names
    messages = [
        ChatMessage(role="system", content=STEP_PROMPT),
        ChatMessage(role="user", content=user_prompt_step_classification),
    ]
    resp = await llm.achat(
        messages, metadata={"role": "lammps_step_judge", "env_id": env.id}
    )
    parts = str(resp.message.content).split()
    log_output, dcd_output, restart_output, write_data_output = (
        parts[0],
        parts[1],
        parts[2],
        parts[3],
    )

    env.write_json(
        {
            "log": log_output,
            "dcd": dcd_output,
            "restart": restart_output,
            "write_data": write_data_output,
        },
        filename=".lammps_extracted_output.json",
        sub_wd=sub_wd,
    )

    molecule_name = os.getenv("SKIP_SLURM_MOLECULE")

    path_pre_mol = osp.join(path_pre, f"{molecule_name}_LAMMPS")
    path_pre_step = osp.join(path_pre_mol, lammps_step)

    target_frame_true = None
    if lammps_step == "step3":
        if lammps_data_in_script is not None:
            target_frame_true = step3_extra_checks(
                lammps_data_in_script, env, sub_wd, path_pre_step
            )
        else:
            target_frame_true = False

    rubric_items = load_step_rubrics(lammps_step)
    raw_judge_results = []
    assert isinstance(lammps_step, str)

    judge_result_list = await asyncio.gather(
        *(
            run_with_TFtool(
                llm,
                instruction,
                mass_section,
                item["description"],
                lammps_script,
                item["id"],  # as metadata
                lammps_step,  # as metadata
                env.id,  # as metadata
            )
            for item in rubric_items
        )
    )
    for item, tf in zip(rubric_items, judge_result_list):
        raw_judge_results.append(
            {"id": item["id"], "question": item["description"], "pass": tf}
        )

    judge_results = aggregate_rubric_results(raw_judge_results)

    # add target_frame_true to tf_bools
    if lammps_step == "step3":
        judge_results.append(
            {
                "id": "target_frame_density_match",
                "question": "For step3, extracted frame density matches the target frame and step3 uses read_data",
                "pass": bool(target_frame_true),
            }
        )

    for jr in judge_results:
        if not jr["pass"]:
            debug(f"failed question : {jr['question']}")

    llm_dump = llm.model_dump(mode="json")
    llm_dump.pop("api_key", None)
    to_save = {
        "Step": lammps_step,
        "Molecule name": molecule_name,
        "Rubrics": judge_results,
        "LLM_judge": llm_dump,
    }
    env.write_json(to_save, filename=".judge_result.json", sub_wd=sub_wd)

    if any([not jr["pass"] for jr in judge_results if jr["id"] != "stage_intent"]):
        debug(f"LAMMPS judge not pass. Step: {lammps_step}")
        env.write_file(content=lammps_step, remote_path="LAMMPS_NOT_PASS")
        env.sys_run("cp ./.judge_result.json ../judge_result.json", sub_wd=sub_wd)
        raise ToolSystemError("LAMMPS_NOT_PASS")

    # copy output files and return
    if log_output != "NONE":
        env.sys_run(f"cp {path_pre_step}/log.lammps {log_output}", sub_wd=sub_wd)
    if dcd_output != "NONE":
        env.sys_run(f"cp {path_pre_step}/trajectory.dcd {dcd_output}", sub_wd=sub_wd)
    if restart_output != "NONE":
        env.sys_run(
            f"cp {path_pre_step}/restart.restart {restart_output}", sub_wd=sub_wd
        )
    if write_data_output != "NONE":
        env.sys_run(
            f"cp {path_pre_step}/final_structure.lammps-data {write_data_output}",
            sub_wd=sub_wd,
        )
    env.sys_run(f"cp {path_pre_step}/slurm* .", sub_wd=sub_wd)

    if (
        lammps_step == "step3" and lammps_data_in_script
    ):
        # Once the step 3 passes, we also replace 'input' lammps-data in case it is not consistent with precomputed output files (especially atom ordering), which is critical for diffusivity stage
        env.sys_run(
            f"cp {path_pre_step}/target_frame.lammps-data {lammps_data_in_script}",
            sub_wd=sub_wd,
        )

    return "You're receiving a polling result. Reason: The job is no longer running or pending (state=COMPLETED). Check the updated files in your working directory and make the next decision."


RAG_RUBRIC_STEP_PATH = Path(__file__).resolve().with_name("step_rubric.json")


def load_step_rubrics(step_key: str):
    """Load rubric items for a step from step_rubric.json."""
    data = json.loads(RAG_RUBRIC_STEP_PATH.read_text())
    for entry in data["rubrics"]:
        if entry["task"] == step_key:
            return entry["rubric_items"]
    raise KeyError(f"Rubrics not found for {step_key}")


def aggregate_rubric_results(raw_results):
    """OR-combine rubric variants like foo/1, foo/2 into foo."""
    grouped = {}
    order = []
    for result in raw_results:
        rubric_id = result["id"]
        base_id = rubric_id.split("/", 1)[0]
        if base_id not in grouped:
            grouped[base_id] = []
            order.append(base_id)
        grouped[base_id].append(result)

    aggregated = []
    for base_id in order:
        variants = grouped[base_id]
        if len(variants) == 1:
            aggregated.append(
                {
                    "id": base_id,
                    "question": variants[0]["question"],
                    "pass": variants[0]["pass"],
                }
            )
            continue

        aggregated.append(
            {
                "id": base_id,
                "question": " OR ".join(
                    f"[{v['id']}] {v['question']}" for v in variants
                ),
                "pass": any(v["pass"] for v in variants),
                "alternatives": variants,
            }
        )

    return aggregated


def wrap_tool(tool: FunctionTool, agent_name: str):
    if not (
        agent_name == "LAMMPS agent"
        and tool.metadata.get_name() == "submit_and_wait"
    ):
        return tool

    async def wrapper(*args, **kwargs):
        ctx: Context = kwargs["ctx"]
        if "script_path" in kwargs:
            script_path = kwargs["script_path"]
        else:
            script_path = args[1]

        env, sub_wd, _ = await get_env_with_sub_wd(ctx)

        if not env.file_exists(script_path, sub_wd):
            async with ctx.store.edit_state() as _state:
                _state.agent_state.task_status = TaskStatus.HOLD
            return f"No such file: {script_path}"

        global SUBMIT_AND_WAIT_CALL_COUNT
        SUBMIT_AND_WAIT_CALL_COUNT += 1

        try:
            return_str = await LLM_judge(ctx, script_path)
        except ToolSystemError:
            # run will ends
            raise
        except Exception as e:
            # run will ends, this should not happen. We abort instead of return to agent.
            tb = traceback.format_exc()
            env.write_file(tb, "./BUG")
            raise ToolSystemError(f"[LE_LAMMPS_GATE][BUG]: {e}") from e

        async with ctx.store.edit_state() as _state:
            _state.agent_state.task_status = TaskStatus.HOLD
        return return_str

    debug("[LE_LAMMPS_GATE] applied")
    tool._async_fn = wrapper
    tool.metadata.return_direct = False

    critic_attached = getattr(tool, "_paimon_critic_attached", False)
    if critic_attached:
        from paimon.world.critic import _attach_to_submit_tool

        # order: critic => if pass => call LLM_judge
        tool = _attach_to_submit_tool(tool)

    return tool
