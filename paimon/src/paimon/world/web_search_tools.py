from llama_index.core.llms import ChatMessage
from llama_index.core.tools import FunctionTool
from tavily import TavilyClient

from paimon import cfg
import paimon.llm
from paimon.llm import run_agent_pipeline
from paimon.util.log import debug


_client: TavilyClient | None = None

def get_tavily_client() -> TavilyClient:
    api_key = cfg.web_search_config.tavily_api_key
    global _client
    if _client is None:
        _client = TavilyClient(api_key=api_key)
    return _client


def _make_llm_context(results: list[dict], max_chars: int = 50000) -> str:
    """Build bounded context for LLM reasoning from all Tavily hits."""
    chunks: list[str] = []
    total = 0
    for i, item in enumerate(results, 1):
        title = (item.get("title") or "").strip()
        url = (item.get("url") or "").strip()
        content = (item.get("raw_content") or item.get("content") or "").strip()
        if not content:
            continue
        block = f"[Result {i}]\nTitle: {title}\nURL: {url}\nContent:\n{content}\n"
        if total + len(block) > max_chars:
            remain = max_chars - total
            if remain > 200:
                chunks.append(block[:remain])
            break
        chunks.append(block)
        total += len(block)
    return "\n".join(chunks) if chunks else "No usable page content found."


async def _extract_all_and_summarize(
    query: str,
    required_output: str,
    llm_context: str,
) -> str:
    llm = paimon.llm.get_llm(
        "fast_reasoning",
        metadata={"role": "web_search_extract"},
    )
    sys_prompt = """
You are a high-fidelity web extraction assistant for atomistic simulation.
Use all provided search results.
Answer the query directly from the provided search context.
Treat required_output as the strict contract for what to include.
Stay concise and include concrete facts with source-aware grounding.
Adapt output to query type instead of forcing a fixed template.
Response policy by query type:
- If the user asks for specific values/properties, return value(s) with unit and conditions (temperature/pressure/phase/model assumptions when available), plus references.
- If the user asks for code/workflow, return code and practical usage steps, plus references.
- For code answers, include provenance/context: module or package name, coding intent (what the code is trying to do), and whether it comes from official docs, GitHub, a paper supplement, blog, or another source type.
- When version information is available for code-related dependencies/tools/APIs, state versions explicitly.
- If the user asks for both, include both values and code/usage in one coherent answer.
- For every major claim, provide supporting references centered on source provenance: document/article/paper type, title, publisher/journal/site, and date when available.
- URLs are optional and secondary; do not rely on URL-only references.
- If key information is missing or conflicting, state that explicitly.
- Prioritize required_output when shaping the final response.
Preservation rules:
- When code is requested, preserve technical details exactly.
- Keep code blocks verbatim when code is requested.
- Preserve package/library names, versions, units, and key conditions.
- Do not replace code with intent-only summaries.
 """  # noqa: E501
    inp = [
        ChatMessage(role="system", content=sys_prompt),
        ChatMessage(
            role="user",
            content=(
                f"Query:\n{query}\n\n"
                f"Required output:\n{required_output}\n\n"
                f"Search context:\n{llm_context}"
            ),
        ),
    ]
    resp, _, _ = await run_agent_pipeline(
        llm=llm,
        chat_history=inp,
        agent_name="web_search",
        metadata={"role": "web_search_extract"},
    )
    return str(resp.message.content)


async def web_search(
    query: str,
    required_output: str = "Essential answer with concrete values/conditions and references.",
) -> str:
    """Search via Tavily and answer using query + explicit output contract."""
    if not query.strip():
        return "[Error] Query must not be empty."
    if not required_output.strip():
        return "[Error] required_output must not be empty."

    conf = cfg.web_search_config
    assert conf.tavily_api_key, "TAVILY_API_KEY is not configured."

    try:
        data = get_tavily_client().search(
            query=query,
            search_depth=conf.search_depth,
            max_results=conf.max_results,
            include_answer=False,
            include_raw_content=True,
            timeout=60,
        )
    except Exception as e:
        return f"[Error] Web search request failed: {type(e).__name__}: {e}"

    results = data.get("results", [])
    search_llm_context = _make_llm_context(results)
    return await _extract_all_and_summarize(
        query=query,
        required_output=required_output,
        llm_context=search_llm_context,
    )


web_search_tool = FunctionTool.from_defaults(
    name="web_search",
    description="""External web lookup for atomistic simulation related questions.

Use policy:
- Use this tool as a last resort, only after trying other relevant information retrieval tools first.

Inputs:
- query: concise retrieval query that includes all essential terms (core keywords/entities/conditions).
- required_output: explicit extraction contract that specifies exactly which information to return and how to structure it (e.g., target values, units, conditions, provenance, code/usage/version details).
This tool retrieves web evidence using query, then shapes the final response according to required_output.

Output expectation:
- Return what required_output asks for.
- For values/properties: include value, unit, and conditions when available.
- For code/workflow: include code, usage, module/package, intent, and versions when available.
- References should emphasize provenance (source type/title/publisher or journal/date); URL is optional.
- Explicitly note missing or conflicting information.
""",  # noqa: E501
    async_fn=web_search,
)
