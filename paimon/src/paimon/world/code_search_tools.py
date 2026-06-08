"""Research tools for code extraction and introspection.

Tool specs ported from CASCADE direct_tools/research_tools.py.
Provides FunctionTool wrappers for use with llama-index agents.
"""
import asyncio
import json
import site
import sysconfig
from typing import Any

from llama_index.core.workflow import Context
from llama_index.core.tools import FunctionTool

from paimon.rag.code_search.chroma_store import get_store
from paimon.rag.code_search.crawler import (
    create_crawler,
    close_crawler,
    extract_code_single_page,
    extract_code_smart_crawl,
    validate_and_normalize_url,
)
from paimon.rag.code_search.local_indexer import query_package_code
from paimon.rag.format import format_code_search_results
from paimon.util.context import get_env_with_sub_wd
from paimon.world.remote_python import pycall_quick_introspect

# Compute site-packages hint for docstrings
try:
    _purelib = sysconfig.get_paths().get("purelib") or ""
    _sitepkgs: list[str] = []
    try:
        _sitepkgs = site.getsitepackages() or []
    except Exception:
        _sitepkgs = []
    SITE_PACKAGES_HINT = _purelib or (_sitepkgs[0] if _sitepkgs else "<site-packages>")
except Exception:
    SITE_PACKAGES_HINT = "<site-packages>"

# Module-level crawler instance
_crawler = None


async def _get_crawler():
    """Get or create the shared crawler instance."""
    global _crawler
    if _crawler is None:
        _crawler = await create_crawler()
    return _crawler


async def _reset_crawler():
    """Reset the crawler instance."""
    global _crawler
    if _crawler is not None:
        await close_crawler(_crawler)
        _crawler = None


async def extract_code_from_url_impl(
    url: str | None = None,
    urls: list[str] | None = None,
) -> str:
    EXTRACTION_TIMEOUT = 100

    try:
        async def extract_with_timeout() -> str:
            return await _extract_code_internal(url, urls)

        try:
            result = await asyncio.wait_for(
                extract_with_timeout(), timeout=EXTRACTION_TIMEOUT
            )
            return result
        except asyncio.TimeoutError:
            timeout_message: dict[str, Any] = {
                "success": False,
                "error": f"Extraction timed out after {EXTRACTION_TIMEOUT} seconds",
                "error_type": "timeout",
                "suggestion": "This webpage may not exist, or extraction is taking "
                "too long. Please try a different URL.",
                "url": url if url else (urls[0] if urls else "unknown"),
                "extracted_code": [],
                "code_blocks_found": 0,
            }

            if urls:
                timeout_message["urls"] = urls
                timeout_message["per_url_results"] = [
                    {
                        "success": False,
                        "url": u,
                        "error": f"Extraction timed out after {EXTRACTION_TIMEOUT} seconds",
                        "error_type": "timeout",
                    }
                    for u in urls
                ]
                timeout_message["summary"] = {
                    "total_unique_urls": len(urls),
                    "successful_extractions": 0,
                    "timeout_urls": len(urls),
                    "total_code_blocks_found": 0,
                }

            return json.dumps(timeout_message, indent=2)

    except Exception as e:
        return json.dumps(
            {"success": False, "error": str(e), "error_type": "exception"}, indent=2
        )


async def _extract_code_internal(
    url: str | None = None, urls: list[str] | None = None
) -> str:
    """Internal implementation of extract_code_from_url."""
    try:
        store = get_store()
        crawler = await _get_crawler()

        async def process_one_url(target_url: str) -> dict[str, Any]:
            """Process a single URL with caching and fallback strategy."""
            target_url, is_valid = validate_and_normalize_url(target_url)
            if not is_valid:
                return {
                    "success": False,
                    "url": target_url,
                    "error": "Invalid URL provided",
                }

            # 1. Check cache first
            if await store.check_exists(target_url):
                cached_code_blocks = await store.get(target_url)
                if cached_code_blocks:
                    for block in cached_code_blocks:
                        block["source_url"] = target_url
                    return {
                        "success": True,
                        "url": target_url,
                        "code_blocks_found": len(cached_code_blocks),
                        "extracted_code": cached_code_blocks,
                        "extraction_method": "cached",
                        "cached": True,
                    }

            # 2. Try single page extraction
            single_page_result = await extract_code_single_page(crawler, target_url)
            try:
                result_data = json.loads(single_page_result)
                if result_data.get("success", False):
                    extracted_code = result_data.get("extracted_code", [])
                    for block in extracted_code:
                        block["source_url"] = target_url

                    if result_data.get("code_blocks_found", 0) > 0:
                        await store.save(
                            target_url,
                            extracted_code,
                            "single_page",
                        )
                        result_data["cached"] = False
                        return result_data
            except (json.JSONDecodeError, KeyError):
                pass

            # 3. Fallback to smart crawl
            smart_result = await extract_code_smart_crawl(crawler, target_url)
            try:
                smart_data = json.loads(smart_result)
                extracted_code = smart_data.get("extracted_code", [])
                for block in extracted_code:
                    block["source_url"] = target_url

                await store.save(
                    target_url,
                    extracted_code,
                    "smart_crawl",
                )
                smart_data["cached"] = False
                return smart_data
            except Exception as e:
                return {
                    "success": False,
                    "url": target_url,
                    "error": f"Both single page and smart crawl failed: {str(e)}",
                }

        # Handle multiple URLs
        if urls is not None and len(urls) > 0:
            if len(urls) == 1:
                result = await process_one_url(urls[0])
                return json.dumps(result, indent=2)

            # Remove duplicates while preserving order
            unique_urls: list[str] = []
            seen_urls: set[str] = set()
            duplicate_urls: list[str] = []
            for u in urls:
                if u in seen_urls:
                    duplicate_urls.append(u)
                else:
                    unique_urls.append(u)
                    seen_urls.add(u)

            if len(unique_urls) == 1:
                result = await process_one_url(unique_urls[0])
                return json.dumps(result, indent=2)

            # Process multiple unique URLs
            all_results: list[dict[str, Any]] = []
            all_code_blocks: list[dict[str, Any]] = []
            cached_count = 0
            extracted_count = 0

            for u in unique_urls:
                result = await process_one_url(u)
                all_results.append(result)
                if result.get("success") and result.get("extracted_code"):
                    all_code_blocks.extend(result["extracted_code"])
                    if result.get("cached", False):
                        cached_count += 1
                    else:
                        extracted_count += 1

            summary_info: dict[str, Any] = {
                "cached_urls": cached_count,
                "newly_extracted_urls": extracted_count,
                "total_unique_urls": len(unique_urls),
                "total_original_urls": len(urls),
                "duplicate_urls_skipped": len(duplicate_urls),
            }
            if duplicate_urls:
                summary_info["duplicate_urls"] = duplicate_urls

            return json.dumps(
                {
                    "success": True,
                    "urls": urls,
                    "unique_urls": unique_urls,
                    "total_code_blocks_found": len(all_code_blocks),
                    "all_extracted_code": all_code_blocks,
                    "per_url_results": all_results,
                    "summary": summary_info,
                },
                indent=2,
            )

        # Handle single URL
        elif url is not None:
            result = await process_one_url(url)
            return json.dumps(result, indent=2)

        # No URL provided
        else:
            return json.dumps(
                {"success": False, "error": "No url or urls provided"}, indent=2
            )

    except Exception as e:
        return json.dumps({"success": False, "error": str(e)}, indent=2)


async def retrieve_extracted_code_impl(query: str, match_count: int = 5) -> str:
    try:
        store = get_store()
        results = await store.search(query=query, top_k=match_count)

        formatted_results: list[dict[str, Any]] = []
        for result in results:
            formatted_result = {
                "source_url": result.get("source_url"),
                "code": result.get("code"),
                "summary": result.get("summary"),
                "context_before": result.get("context_before"),
                "context_after": result.get("context_after"),
                "type": result.get("type"),
                "language": result.get("language"),
                "index": result.get("index"),
                "similarity_score": result.get("similarity_score"),
            }
            formatted_results.append(formatted_result)

        return json.dumps(
            {
                "success": True,
                "query": query,
                "search_mode": "vector",
                "results": formatted_results,
                "count": len(formatted_results),
            },
            indent=2,
        )
    except Exception as e:
        return json.dumps(
            {"success": False, "query": query, "error": str(e)}, indent=2
        )


async def search_package_code_impl(
    pkg_name: str,
    query: str,
    match_count: int = 5,
) -> str:
    try:
        results = await query_package_code(
            pkg_name=pkg_name,
            query=query,
            top_k=match_count,
        )

        if not results:
            return f"No code examples found in {pkg_name} package for query: {query}"

        # Format results for agent consumption (similar to LAMMPS RAG)
        formatted_text = format_code_search_results(
            results=results,
            include_context=True,
            include_source=True,
            max_results=match_count,
            max_chars=15000,
        )

        return formatted_text

    except Exception as e:
        error_msg = str(e)
        # Check if it's a missing collection error
        if "not found" in error_msg.lower() or "does not exist" in error_msg.lower():
            return (
                f"ERROR: Package '{pkg_name}' has not been indexed yet.\n\n"
                f"To build the index, run:\n"
                f"  python -m paimon.rag.code_search.cli build {pkg_name} <html_root>\n"
            )
        return f"ERROR searching {pkg_name} package: {error_msg}"


async def quick_introspect_impl(
    ctx: Context,
    code_content: str | None = None,
    class_hint: str | None = None,
    method_hint: str | None = None,
    package_path: str | None = None,
    function_hint: str | None = None,
    module_hint: str | None = None,
    repo_hint: str | None = None,
    max_suggestions: int = 10,
    no_imports: bool = False,
) -> str:
    env, sub_wd, venv_name = await get_env_with_sub_wd(ctx)
    return await pycall_quick_introspect(
        env=env,
        code_content=code_content,
        class_hint=class_hint,
        method_hint=method_hint,
        package_path=package_path,
        function_hint=function_hint,
        module_hint=module_hint,
        repo_hint=repo_hint,
        max_suggestions=max_suggestions,
        no_imports=no_imports,
        sub_wd=sub_wd,
        venv_name=venv_name,
    )


async def runtime_probe_snippet_impl(snippet: str) -> str:
    SNIPPETS: dict[str, str] = {
        "try_get_key": (
            "from simulation_util.introspection import try_get_key\n\n"
            "# Usage at the KeyError site: replace mapping['k'] with "
            "try_get_key(mapping, 'k')\n"
        ),
        "try_get_attr": (
            "from simulation_util.introspection import try_get_attr\n\n"
            "# Usage at the AttributeError site: replace obj.attr with "
            "try_get_attr(obj, 'attr')\n"
        ),
    }

    key = (snippet or "").strip()
    if key not in SNIPPETS:
        return json.dumps(
            {
                "success": False,
                "report": "Invalid snippet. Use one of: try_get_key, try_get_attr",
            },
            indent=2,
        )

    headers = {
        "try_get_key": (
            "Usage at the KeyError site: replace mapping['k'] with "
            "try_get_key(mapping, 'k').\n"
            "**You MUST paste this snippet right after your imports.**\n"
        ),
        "try_get_attr": (
            "Usage at the AttributeError site: replace obj.attr with "
            "try_get_attr(obj, 'attr').\n"
            "**You MUST paste this snippet right after your imports.**\n"
        ),
    }

    header = headers[key]
    return json.dumps(
        {"success": True, "report": header + "\n" + SNIPPETS[key]}, indent=2
    )


# Create FunctionTool instances


extract_code_from_url_tool = FunctionTool.from_defaults(
    name="extract_code_from_url",
    description="""Extract code examples and commands from one or more URLs for immediate use.

This tool uses a caching strategy: first checks if code has already been extracted
from the URL(s), and if not, performs extraction and stores the results for future use.

Extraction strategy:
1. Check cache first
2. Try single page extraction with optimized strategy:
   - HTML extraction only for ReadTheDocs/Sphinx
   - Markdown extraction for other content types
   - Special handling for Jupyter notebooks (JSON extraction)
3. If no code blocks found, fallback to smart crawl

If a list of URLs is provided, all will be processed and results merged.
Each code block in the result will include a 'source_url' key indicating its origin.

Args:
    url: Single URL to extract code from
    urls: List of URLs to extract code from

Returns:
    JSON string with extracted code examples, commands, and content summary.
    Key fields to focus on:
    - 'code': The extracted code content
    - 'context_before': Text content before the code block
    - 'context_after': Text content after the code block
    - 'source_url': URL where the code was extracted from
""",  # noqa: E501
    async_fn=extract_code_from_url_impl,
)


retrieve_extracted_code_tool = FunctionTool.from_defaults(
    name="retrieve_extracted_code",
    description="""Retrieve extracted code blocks relevant to the query.

This tool searches the extracted_code table for code blocks relevant to the query
and returns the matching examples with their summaries and context.
You can call this tool multiple times with different queries to get more relevant results.

Args:
    query: The search query
    match_count: Maximum number of results to return (default: 5)

Returns:
    JSON string with the search results including code, context, and similarity scores.
""",  # noqa: E501
    async_fn=retrieve_extracted_code_impl,
)


search_package_code_tool = FunctionTool.from_defaults(
    name="search_package_code",
    description="""Retrieve code blocks relevant to query from pre-built vector store.

This tool searches the extracted_code table for code blocks relevant to the query
and returns the matching examples with their summaries and context. It queries pre-built vector stores for specific packages (e.g., ase)
You can call this tool multiple times with different queries to get more relevant results.

Use this tool when you need code examples from well-known packages that have been
pre-indexed.

Args:
    pkg_name: Package name (e.g., "ase")
    query: Search query describing the code you need
    match_count: Maximum number of results to return (default: 5)

Returns:
    Formatted text with code examples, context, and relevance scores.
""",  # noqa: E501
    async_fn=search_package_code_impl,
)


quick_introspect_tool = FunctionTool.from_defaults(
    name="quick_introspect",
    description=f"""
    Fast, static-first introspection for fixing import/class/method/function related errors.
    Uses Jedi for static discovery first (no side effects), then runtime import/inspect fallback.
    Returns a human/agent-readable report string with suggested import lines (if no-imports is not set), class methods (if method_hint/class_hint and repo_hint or package_path are provided), and functions (if function_hint/module_hint and repo_hint or package_path are provided).
    Parameter relationships:
    - repo_hint vs package_path: mutually exclusive; pass at most one.
    - module_hint must be used together with function_hint.
    - If class_hint or method_hint is provided, one of repo_hint or package_path is required.
    - If function_hint is provided, one of repo_hint or package_path is required.
    Notes:
    - You can provide fuzzy hints for class_hint, method_hint, function_hint, module_hint (but repo_hint need to be exact) if you cannot provide exact hints.
    - It is recommended to use code_content to provide the code content for import diagnostics.
    - repo_hint is the top-level import module name (may differ from pip distribution name). If repo_hint cannot be imported, provide package_path instead.
    - PATH HANDLING for package_path:
      * Absolute path: used as-is.
      * Relative path: resolved against the active environment's site-packages root {SITE_PACKAGES_HINT}.
        For example, passing "pydantic" will be tried as {SITE_PACKAGES_HINT}/pydantic.
      * If unsure about the absolute path, use your check_package_version tool (if you have this tool) to obtain it.
      * If repo_hint fails to import, try providing an absolute or relative package_path (relative path starts from {SITE_PACKAGES_HINT}).
    - Function vs Method:
      * Use method_hint when the target is a class member (instance/class method). Do not use function_hint for class/instance methods. **Most of the time, you should use method_hint**
      * Use function_hint only for top-level (module-level) functions; optionally add module_hint to narrow
      * Heuristics: analyze the call-site pattern — calls like SomeClass.method(...)/obj.method(...) → method_hint; calls like package.module.function(...), module.function(...), function(...) → function_hint
    - method_hint can be provided without class_hint to trigger a repo-wide search (noisy but useful, but often it is more recommended to add fuzzy or exact class_hint to narrow down the search).
    - To silence import diagnostics if you think it is too noisy for your use case, set no_imports=true and reuse this tool to introspect the code again.
    - max_suggestions is the maximum number of suggestions to return. If not provided, it will return all suggestions. You can set it to a smaller number to reduce the noise if needed.
    - Set env QI_DEBUG_ENGINE=1 to see whether Jedi or runtime fallback is used. But normally you don't need to set this.
""",  # noqa: E501
    async_fn=quick_introspect_impl,
)


runtime_probe_snippet_tool = FunctionTool.from_defaults(
    name="runtime_probe_snippet",
    description="""\
Return a ready-to-paste Python probe snippet for targeted debugging of runtime errors.
    snippet: one of "try_get_key", "try_get_attr".
    - try_get_key: Use at a KeyError site. Replace mapping['k'] with try_get_key(mapping, 'k').
      If the key exists, it prints an OK message (either the earlier error was at a different site and you
      should probe the right line and re-run, or you are now probing a different key than the one that caused the earlier KeyError and have successfully
      debugged it). If missing, it prints available keys and error context to guide a fix.
    - try_get_attr: Use at an AttributeError site. Replace obj.attr with try_get_attr(obj, 'attr').
      If the attribute exists, it prints an OK message (either the earlier error was at a different site and
      you should probe the right line and re-run, or you are now probing a different attribute than the one that caused the earlier AttributeError and have
      successfully debugged it). If missing, it prints public attributes, similarity suggestions and error context to guide a fix.
    The returned report contains a short usage header followed by the snippet code. You MUST Paste the snippet into
    your current script right after your imports and follow the usage note to replace the failing access.
""",  # noqa: E501
    async_fn=runtime_probe_snippet_impl,
)


def get_research_tools() -> list[FunctionTool]:
    """Return all research tools for agent use."""
    return [
        extract_code_from_url_tool,
        retrieve_extracted_code_tool,
        search_package_code_tool,
        quick_introspect_tool,
        runtime_probe_snippet_tool,
    ]
