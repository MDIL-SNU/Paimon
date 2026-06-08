import os
import os.path as osp
import tempfile
from typing import Annotated, Literal

from llama_index.core.tools import FunctionTool
from llama_index.core.workflow import Context

from paimon import cfg
from paimon.models import PlanState, SubtaskAgentState
from paimon.knowledge.library import get_knowledge
from paimon.episodic.format import get_flow_hints_for_expert
from paimon.extraction import extract_methodology
from paimon.llm import is_true as is_true_llm
from paimon.rag.docs import ExpertRAGSystem
from paimon.rag.docs.models import DocEntry
from paimon.util.context import get_env_with_sub_wd, get_state
from paimon.util.log import debug


_expert_rag: ExpertRAGSystem | None = None


async def _get_expert_rag() -> ExpertRAGSystem:
    """Get or initialize expert RAG system."""
    global _expert_rag
    if _expert_rag is None:
        _expert_rag = ExpertRAGSystem(force_rebuild=False)
        await _expert_rag.build_index()
    return _expert_rag


async def retrieve_umlip_knowledge(
    umlip_family: Literal["SevenNet", "MACE"],
) -> str:
    return get_knowledge(f"forcefield/{umlip_family.lower()}/planner.txt")


retrieve_umlip_knowledge_tool = FunctionTool.from_defaults(
    name="retrieve_umlip_knowledge",
    description="""\
Retrieves verified expert knowledge and operational guidelines for Universal Machine Learning Interatomic Potentials (u-MLIPs). 
This tool provides critical information on pre-trained models, including their specific capabilities and limitations. 
You MUST invoke this tool before planning or executing any tasks that using u-MLIP to ground your decisions in verified expert guidelines. 
Currently, only "SevenNet" and "MACE" are supported.
""",  # noqa: E501
    async_fn=retrieve_umlip_knowledge,
)


async def retrieve_expert_knowledge(
    ctx: Context,
    query: Annotated[str, "Query describing what simulation is needed"],
) -> str:
    env, sub_wd, _ = await get_env_with_sub_wd(ctx)
    rag = await _get_expert_rag()
    state = await get_state(ctx)

    retrieve_expert_knowledge_hist = state.auxiliary_data.get(
        "retrieve_expert_knowledge_hist", []
    )

    result: DocEntry | None = await rag.retrieve(query, top_k=1)

    env.append_json(
        key="expert_retrieve_event",
        value={"query": query, "selected": result.title if result else "None"},
        filename=".rag_events.json",
        sub_wd=sub_wd,
    )

    if result is None:
        title = "general"
        content = get_knowledge("expert/general.txt")
    else:
        prompt = f"""\
Decision policy:
1) Mark true when the knowledge directly matches the task.
2) Mark true when the knowledge is from the same simulation domain and can be reasonably adapted
   (e.g., related target property, same material family, same simulation workflow with parameter adjustments).
3) Mark false when adaptation would be a stretch across different physical domains or would likely mislead setup.
4) Mark false when only generic buzzword overlap exists without practical transferability.

<knowledge>
{result.content}
</knowledge>
<task>
{query}
</task>

Based on the policy, is the knowledge relevant enough to the task?
"""  # noqa: E501

        if not (await is_true_llm(prompt)):
            title = "general"
            content = get_knowledge("expert/general.txt")
        else:
            title = result.title
            content = result.content

    # Three cases: duplicates, not found or not relevant (title==general), found
    if title in retrieve_expert_knowledge_hist:
        return f"""\
The search for '{query}' retrieved the same document (Title: {title}) already provided.
No additional expert knowledge matches this query better in the current database.
"""  # noqa: E501

    if title == "general":
        system_message = """\
The specific simulation recipe for your query was not found.
The content provided is a General Reference Guide for atomistic simulations.
Use these best practices to advise the user, but do not fabricate specific parameters or potentials not mentioned in the text.
"""  # noqa: E501
    else:
        system_message = """\
The content provided is the most relevant human-verified simulation protocol found for this query.
It may be closely related rather than an exact match.
Prioritize applicable parameters/workflows and reconcile them with the user's request.
"""  # noqa: E501

    retrieve_expert_knowledge_hist.append(title)

    state.auxiliary_data["retrieve_expert_knowledge_hist"] = (
        retrieve_expert_knowledge_hist
    )
    if isinstance(state, PlanState):
        await ctx.store.set_state(state)  # type: ignore
    elif isinstance(state, SubtaskAgentState):
        await ctx.store.set("agent_state", state)

    output = (
        f'<expert_knowledge title="{title}">\n'
        f"{content}\n"
        f"</expert_knowledge>"
    )

    # Attach flow hints from past successful runs if available
    if cfg.use_episodic and result and title != "general":
        if hints := get_flow_hints_for_expert(result.doc_id):
            output += "\n" + hints

    output += f"<system>{system_message}</system>"
    return output


retrieve_expert_knowledge_tool = FunctionTool.from_defaults(
    name="retrieve_expert_knowledge",
    description="""\
Retrieve the most relevant domain expert knowledge for simulation guidance and best practices.

This tool only covers simulation procedures and methodological best practices.
It does NOT provide general domain knowledge or theoretical explanations.

Use only for specific, well-defined simulation contexts.
Queries should be concrete translatable into a simulation recipe.
Avoid general or vague questions.
""",
    async_fn=retrieve_expert_knowledge,
)


def _download_remote_file(env, filename: str, sub_wd: str) -> str:
    """Download a remote file to a local tempfile.

    Returns local temp path. Caller is responsible for cleanup.
    Returns empty string on failure.
    """
    if not env.file_exists(filename, sub_wd):
        return ""
    wd = env.get_sub_wd_path(sub_wd)
    remote_path = osp.join(wd, filename)
    suffix = osp.splitext(filename)[1] or ".bin"
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix)
    local_path = tmp.name
    tmp.close()
    try:
        debug(f"[env:get] {remote_path} to {local_path}")
        env._conn.get(remote_path, local=local_path)
    except Exception:
        if os.path.exists(local_path):
            os.remove(local_path)
        return ""
    return local_path


async def extract_paper_methodology(
    ctx: Context,
    paper_filename: Annotated[str, "PDF filename in external_files folder"],
    task: Annotated[str, "Focused description of what methodology to extract"],
    supporting_info_filename: Annotated[
        str | None, "Optional supplementary PDF filename in external_files"
    ] = None,
) -> str:
    env, sub_wd, _ = await get_env_with_sub_wd(ctx)

    tmp_paths: list[str] = []
    try:
        paper_path = _download_remote_file(env, paper_filename, "external_files")
        if not paper_path:
            return f"""\
'{paper_filename}' not found or failed to download from external_files.
Check the filename and ensure the PDF has been uploaded.
"""
        tmp_paths.append(paper_path)

        supporting_info_path: str | None = None
        if supporting_info_filename:
            supporting_info_path = _download_remote_file(
                env, supporting_info_filename, "external_files"
            )
            if not supporting_info_path:
                return f"'{supporting_info_filename}' not found or failed to download from external_files."  # noqa: E501
            tmp_paths.append(supporting_info_path)

        debug(f"[extract_paper] starting extraction for {paper_filename}")  # noqa: E501
        result = extract_methodology(
            paper_path, task, supporting_info_path=supporting_info_path
        )
    except Exception as e:
        debug(f"[extract_paper] failed: {type(e).__name__}")
        return f"""\
Extraction failed: {type(e).__name__}: {e}
This may be caused by a network issue or an LLM API error.
Inform the user about this failure.
"""
    finally:
        for p in tmp_paths:
            if os.path.exists(p):
                os.remove(p)

    debug(f"[extract_paper] done for {paper_filename}")

    if env:
        env.append_json(
            key="expert_extract_event",
            value={"paper": paper_filename, "task": task},
            filename=".rag_events.json",
            sub_wd=sub_wd,
        )

    system_message = """\
This is a simulation protocol extracted from a user-provided research paper.
Treat it with the same authority as retrieve_expert_knowledge results.
"""
    return (
        f"<expert_knowledge title="
        f'"Extracted from: {paper_filename}">\n'
        f"{result.protocol}\n"
        f"</expert_knowledge>"
        f"<system>{system_message}</system>"
    )


extract_paper_methodology_tool = FunctionTool.from_defaults(
    name="extract_paper_methodology",
    description="""\
Extract a simulation protocol from a research paper PDF uploaded to external_files.
Provide a focused task description of which methodology to extract from the paper.
Returns a self-contained Markdown protocol.
This tool makes multiple LLM calls and may take 1-2 minutes.
The paper PDF must exist in external_files.
At most one another PDF file can be entered as an SI for the paper.
""",
    async_fn=extract_paper_methodology,
)
