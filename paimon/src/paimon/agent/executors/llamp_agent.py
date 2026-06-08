import json
import traceback
from io import StringIO
import os
import os.path as osp
import re
import time

import requests  # HTTP requests run locally
from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context

import ase.io
from pymatgen.core import Structure
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

from paimon import cfg
from paimon.knowledge.library import get_knowledge
from paimon.util.context import get_env_with_sub_wd
from paimon.world.common_tools import run_python_tool, write_file_tool, run_bash_tool
from paimon.agent.agent_config import AgentConfig
from paimon.util.log import debug


def _saved_structure_to_atoms_txt(structure_fname):
    # return Atoms if successful
    root = cfg.llamp_config.server_working_dir

    structure = Structure.from_file(osp.join(root, structure_fname))
    sga = SpacegroupAnalyzer(structure, symprec=1e-3)
    structure = sga.get_conventional_standard_structure()
    atoms = structure.to_ase_atoms()

    # Embed provenance from filename (mp-{id}-{formula}-sg{n}.json)
    m = re.match(r"(mp-\d+)-(.+)-sg(\d+)\.json$", osp.basename(structure_fname))
    if m:
        atoms.info["material_id"] = m.group(1)
        atoms.info["formula"] = m.group(2)
        atoms.info["space_group_number"] = int(m.group(3))
    atoms.info["space_group_symbol"] = sga.get_space_group_symbol()

    tmp = StringIO()
    ase.io.write(tmp, atoms, format="extxyz")
    return tmp.getvalue()


def _parse(response_text: str) -> dict:
    chat_id_match = re.search(r"\[chat_id\](\S+)", response_text)
    chat_id = chat_id_match.group(1) if chat_id_match else None
    action_pattern = r"Action:\s*```(?:json)?\s*(\{.*?\})\s*```"
    action_matches = re.findall(action_pattern, response_text, re.DOTALL)
    actions = []
    for match in action_matches:
        try:
            action_data = json.loads(match)
            actions.append(action_data)
        except json.JSONDecodeError:
            actions.append({"raw": match, "parse_error": True})

    structure_files = re.findall(r"(mp-\d+-[\w]+-sg\d+\.json)", response_text)
    structure_files = list(dict.fromkeys(structure_files))

    final_answer = next(
        (
            a.get("action_input")
            for a in actions
            if a.get("action") == "Final Answer"
        ),
        "",
    )
    return {
        "chat_id": chat_id,
        "actions": actions,
        "final_answer": final_answer,
        "structure_files": structure_files,
        "raw_response": response_text,
    }


async def query_llamp(
    ctx: Context,
    query: str,
) -> str:
    env, sub_wd, venv_name = await get_env_with_sub_wd(ctx)

    debug(f"[LLaMP Agent] Starting query: {query[:100]}...")

    # Get llamp configuration
    llamp_url = cfg.llamp_config.service_url
    openai_key = cfg.llamp_config.openai_api_key or os.environ.get("OPENAI_API_KEY")
    mp_key = cfg.llamp_config.mp_api_key or os.environ.get("MP_API_KEY")
    # Use longer timeout - llamp queries can take time
    timeout = cfg.llamp_config.timeout

    debug(f"[LLaMP Agent] llamp URL: {llamp_url}, timeout: {timeout}s")

    assert openai_key, "Error: OPENAI_API_KEY not configured."
    assert mp_key, "Error: MP_API_KEY not configured."

    chat_id = f"paimon_{env.id}_{int(time.time())}"

    payload = {
        "text": query,
        "OpenAiAPIKey": openai_key,
        "mpAPIKey": mp_key,
        "chat_id": chat_id,
    }

    try:
        debug(f"[LLaMP Agent] Sending HTTP POST to {llamp_url}/api/chat")
        response = requests.post(
            f"{llamp_url}/api/chat", json=payload, timeout=timeout
        )
        response.raise_for_status()

        # Parse streaming response
        result_text = response.text
        debug(f"[LLaMP Agent] Received response: {len(result_text)} characters")
        resp_parsed = _parse(result_text)
    except requests.exceptions.HTTPError as e:
        debug(f"HTTP error: {e.response.status_code}")
        debug(f"Response: {e.response.text}")
        resp_parsed = _parse(result_text)
    except requests.exceptions.Timeout:
        error_msg = "Error: request timed out. Simplify your query."
        debug(f"[LLaMP Agent] {error_msg}")
        return error_msg
    except requests.exceptions.RequestException as e:
        debug(f"[LLaMP Agent] Request exception: {e}")
        raise e

    response_filename = f"llamp_response_{int(time.time())}.json"
    env.write_json(resp_parsed, filename=response_filename, sub_wd=sub_wd)

    stct_retrieved_info = []
    if resp_parsed["structure_files"]:
        for fname in resp_parsed["structure_files"]:
            try:
                extxyz_fname = fname.replace(".json", ".extxyz")
                atoms_str = _saved_structure_to_atoms_txt(fname)
                env.write_file(atoms_str, remote_path=extxyz_fname, sub_wd=sub_wd)
                stct_retrieved_info.append(f"{extxyz_fname} is saved")
            except FileNotFoundError as _:
                stct_retrieved_info.append(f"{fname} is mentioned but not found")
            except Exception as e:
                print("Error detacted while parsing structure file:")
                traceback.print_exc()
                stct_retrieved_info.append(f"Failed to retrieve mentioned {fname}")
    if not stct_retrieved_info:
        stct_retrieved_info.append("No structure is retrieved from LLaMP")
    stct_retrieved_info_txt = "\n".join(stct_retrieved_info)

    return f"""\
<LLaMP response>
{resp_parsed["final_answer"]}
</LLaMP response>
<LLaMP structure retrieval log>
{stct_retrieved_info_txt}
</LLaMP structure retrieval log>
"""


query_llamp_tool = FunctionTool.from_defaults(
    name="query_llamp",
    description="""\
Query the Materials Project database via llamp agent. 
Accepts natural language queries for material properties, structures, thermodynamics, electronic structure, etc.
For structure request, it is saved as as extxyz files.
You MUST NOT specify any representation of the structure (conventional cell or primitive cell) and the file format in the query.
""",  # noqa: E501
    async_fn=query_llamp,
)


def config() -> AgentConfig:
    system_prompt = """\
{{common}}
{{tips}}
"""

    description = """\
<general>
This agent queries Materials Project database for properties, structures, and thermodynamic data via API service and saves resulting structure if requested.
Crystal structure initialization should begin with this agent to ensure data provenance and reproducibility.
This agent is strictly limited to retrieval of information or structure and does not perform any form of structure construction, modification, surface generation, or geometry manipulation.
</general>

<instruction_requirements>
- Clearly specify the target material or property to retrieve
- Provide search criteria such as chemical formula, constituent elements, or simple constraints
- Requests must be limited to bulk material lookup and data retrieval
</instruction_requirements>

<typical_outputs>
- Bulk crystal structure files retrieved directly from Materials Project (e.g., .extxyz)
- Material property summaries in text or structured (JSON) form
- Materials Project material IDs
</typical_outputs>

<non_capabilities>
- Slab generation or surface construction
- Supercell creation or atom count modification
- Structure relaxation, reconstruction, or synthesis
- Any geometry manipulation beyond direct retrieval
</non_capabilities>
"""  # noqa: E501

    # TODO: solve this
    name = "LLaMP (Materials Project) agent"
    canonical_name = "llamp_agent"

    return AgentConfig(
        name=name,
        description=description,
        tools=[
            query_llamp_tool,
            write_file_tool,
            run_python_tool,
            run_bash_tool,
        ],
        system_prompt=system_prompt.replace("{{common}}", "{{common}}").replace(
            "{{tips}}", get_knowledge(f"agents/{canonical_name}/tips")
        ),
        task_types=[
            "Structure Generation",
            "Preparation",
        ],
        auxiliary=True,
    )
