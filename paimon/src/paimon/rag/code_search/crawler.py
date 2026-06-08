"""Web crawler for fetching HTML documentation.

Ported from CASCADE research_mcp.py.
Uses crawl4ai for async web crawling with browser automation.
"""

from __future__ import annotations

import json
import re
from typing import Any
from urllib.parse import urldefrag, urlparse
from xml.etree import ElementTree

import requests
from crawl4ai import (
    AsyncWebCrawler,
    BrowserConfig,
    CacheMode,
    CrawlerRunConfig,
    MemoryAdaptiveDispatcher,
)

from paimon.rag.code_search.extractors import (
    detect_content_type_and_source,
    extract_code_blocks,
    extract_command_examples,
    extract_jupyter_notebook_cells,
    extract_markdown_code_blocks,
    extract_readthedocs_code_blocks,
    extract_smart_context_after,
    extract_smart_context_before,
)


def is_sitemap(url: str) -> bool:
    """Check if a URL is a sitemap."""
    return url.endswith("sitemap.xml") or "sitemap" in urlparse(url).path


def is_txt(url: str) -> bool:
    """Check if a URL is a text file."""
    return url.endswith(".txt")


def parse_sitemap(sitemap_url: str) -> list[str]:
    """Parse a sitemap and extract URLs."""
    resp = requests.get(sitemap_url, timeout=30)
    urls: list[str] = []
    if resp.status_code == 200:
        try:
            tree = ElementTree.fromstring(resp.content)
            urls = [loc.text for loc in tree.findall(".//{*}loc") if loc.text]
        except Exception:
            pass
    return urls


def github_blob_to_raw(url: str) -> str:
    """Convert a GitHub blob URL to the corresponding raw URL if applicable.

    For example:
    https://github.com/user/repo/blob/branch/path/to/file.md
    ->
    https://raw.githubusercontent.com/user/repo/branch/path/to/file.md

    If the URL is not a GitHub blob URL, return it unchanged.
    """
    if not url:
        return url

    # Check if already converted
    if "raw.githubusercontent.com" in url:
        return url

    m = re.match(r"https://github.com/([^/]+)/([^/]+)/blob/([^#?]+)", url)
    if m:
        user, repo, path = m.groups()
        return f"https://raw.githubusercontent.com/{user}/{repo}/{path}"
    return url


def validate_and_normalize_url(url: str) -> tuple[str, bool]:
    """Validate and normalize a URL for processing.

    Handles:
    - view-source: URLs (converts to regular URLs)
    - GitHub blob URLs (converts to raw URLs)
    - Regular URLs (validates and returns as-is)

    Returns:
        Tuple of (normalized_url, is_valid)
    """
    if not url:
        return url, False

    try:
        # Handle view-source: URLs
        if url.startswith("view-source:"):
            # Remove view-source: prefix
            url = url[12:]  # len('view-source:') = 12

        # Basic URL validation
        parsed = urlparse(url)
        if not parsed.scheme or not parsed.netloc:
            return url, False

        # Normalize the URL (GitHub blob to raw, etc.)
        normalized = github_blob_to_raw(url)
        return normalized, True
    except Exception:
        return url, False


def smart_chunk_markdown(text: str, chunk_size: int = 5000) -> list[str]:
    """Split text into chunks, respecting code blocks and paragraphs."""
    chunks: list[str] = []
    start = 0
    text_length = len(text)

    while start < text_length:
        # Calculate end position
        end = start + chunk_size

        # If we're at the end of the text, just take what's left
        if end >= text_length:
            chunks.append(text[start:].strip())
            break

        # Try to find a code block boundary first (```)
        chunk = text[start:end]
        code_block = chunk.rfind("```")
        if code_block != -1 and code_block > chunk_size * 0.3:
            end = start + code_block
        # If no code block, try to break at a paragraph
        elif "\n\n" in chunk:
            # Find the last paragraph break
            last_break = chunk.rfind("\n\n")
            if last_break > chunk_size * 0.3:  # Only break if we're past 30%
                end = start + last_break
        # If no paragraph break, try to break at a sentence
        elif ". " in chunk:
            # Find the last sentence break
            last_period = chunk.rfind(". ")
            if last_period > chunk_size * 0.3:  # Only break if we're past 30%
                end = start + last_period + 1

        # Extract chunk and clean it up
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)

        # Move start position for next chunk
        start = end

    return chunks


async def crawl_batch(
    crawler: AsyncWebCrawler, urls: list[str], max_concurrent: int = 10
) -> list[dict[str, Any]]:
    """Batch crawl multiple URLs in parallel.

    Args:
        crawler: AsyncWebCrawler instance
        urls: List of URLs to crawl
        max_concurrent: Maximum number of concurrent browser sessions

    Returns:
        List of dictionaries with URL, markdown content, and success status
    """
    crawl_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS, stream=False, verbose=False
    )

    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent,
    )

    try:
        results = await crawler.arun_many(
            urls=urls, config=crawl_config, dispatcher=dispatcher
        )
        return [
            {"url": r.url, "markdown": r.markdown, "success": True}
            for r in results
            if r.success and r.markdown
        ]
    except Exception:
        return []
    finally:
        # Ensure dispatcher is properly cleaned up
        try:
            await dispatcher.close()
        except Exception:
            pass


async def crawl_recursive_internal_links(
    crawler: AsyncWebCrawler,
    start_urls: list[str],
    max_depth: int = 3,
    max_concurrent: int = 10,
) -> list[dict[str, Any]]:
    """Recursively crawl internal links from start URLs up to a maximum depth.

    Args:
        crawler: AsyncWebCrawler instance
        start_urls: List of starting URLs
        max_depth: Maximum recursion depth
        max_concurrent: Maximum number of concurrent browser sessions

    Returns:
        List of dictionaries with URL and markdown content
    """
    run_config = CrawlerRunConfig(
        cache_mode=CacheMode.BYPASS, stream=False, verbose=False
    )

    dispatcher = MemoryAdaptiveDispatcher(
        memory_threshold_percent=70.0,
        check_interval=1.0,
        max_session_permit=max_concurrent,
    )

    visited: set[str] = set()

    def normalize_url(url: str) -> str:
        return urldefrag(url)[0]

    current_urls = {normalize_url(u) for u in start_urls}
    results_all: list[dict[str, Any]] = []

    for _depth in range(max_depth):
        urls_to_crawl = [
            normalize_url(url)
            for url in current_urls
            if normalize_url(url) not in visited
        ]
        if not urls_to_crawl:
            break

        results = await crawler.arun_many(
            urls=urls_to_crawl, config=run_config, dispatcher=dispatcher
        )

        next_level_urls: set[str] = set()
        for result in results:
            norm_url = normalize_url(result.url)
            visited.add(norm_url)

            if result.success and result.markdown:
                results_all.append(
                    {"url": result.url, "markdown": result.markdown, "success": True}
                )

                for link in result.links.get("internal", []):
                    next_url = normalize_url(link["href"])
                    if next_url not in visited:
                        next_level_urls.add(next_url)

        current_urls = next_level_urls

    return results_all


async def crawl_markdown_file(
    crawler: AsyncWebCrawler, url: str
) -> list[dict[str, Any]]:
    """Crawl a .txt or markdown file.

    Args:
        crawler: AsyncWebCrawler instance
        url: URL of the file

    Returns:
        List of dictionaries with URL, markdown content, and success status
    """
    try:
        crawl_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS, stream=False, verbose=False
        )

        result = await crawler.arun(url=url, config=crawl_config)
        if result.success:
            # Try to get content, prefer markdown then text
            content = result.markdown or result.text or ""
            if content:
                return [{"url": url, "markdown": content, "success": True}]
            else:
                return [
                    {
                        "url": url,
                        "markdown": "",
                        "success": False,
                        "error": "No content found",
                    }
                ]
        else:
            return [
                {
                    "url": url,
                    "markdown": "",
                    "success": False,
                    "error": result.error_message or "Unknown error",
                }
            ]
    except Exception as e:
        return [{"url": url, "markdown": "", "success": False, "error": str(e)}]


async def extract_code_single_page(crawler: AsyncWebCrawler, url: str) -> str:
    """Extract code from a single webpage without following any links.

    Optimized extraction strategy:
    1. HTML extraction only for ReadTheDocs/Sphinx documentation
    2. Markdown extraction for other content types
    3. Special handling for Jupyter notebooks (JSON extraction with raw URL fallback)
    4. No deduplication to preserve all code blocks

    Returns:
        JSON string with extraction results
    """
    try:
        original_url = url
        url, is_valid = validate_and_normalize_url(url)
        if not is_valid:
            return json.dumps(
                {
                    "success": False,
                    "url": original_url,
                    "crawl_method": "single_page",
                    "error": "Invalid URL provided for single page extraction",
                    "extracted_code": [],
                },
                indent=2,
            )

        # Configure crawler
        run_config = CrawlerRunConfig(
            cache_mode=CacheMode.BYPASS, stream=False, verbose=False
        )

        # Fetch the page
        result = await crawler.arun(url=url, config=run_config)

        # Check if crawl was successful and we have content
        if not result.success:
            return json.dumps(
                {
                    "success": False,
                    "url": original_url,
                    "crawl_method": "single_page",
                    "error": "Failed to crawl page",
                    "error_details": getattr(result, "error_message", "Unknown error"),
                    "extracted_code": [],
                },
                indent=2,
            )

        if not hasattr(result, "html") and not hasattr(result, "markdown"):
            return json.dumps(
                {
                    "success": False,
                    "url": original_url,
                    "crawl_method": "single_page",
                    "error": "No content available (neither HTML nor Markdown)",
                    "extracted_code": [],
                },
                indent=2,
            )

        code_blocks: list[dict[str, Any]] = []
        extraction_method = "none"
        doc_system = "unknown"

        # Strategy: HTML only for ReadTheDocs/Sphinx, otherwise use markdown
        # For Jupyter notebooks, prioritize JSON extraction
        # 1) HTML path only for ReadTheDocs/Sphinx
        if hasattr(result, "html") and result.html:
            try:
                # Detect documentation system
                content_type, detected_doc_system, has_code = (
                    detect_content_type_and_source(result.html, url)
                )
                doc_system = detected_doc_system

                # Only use HTML extraction for ReadTheDocs/Sphinx
                if detected_doc_system in ["readthedocs", "sphinx"]:
                    code_blocks = extract_readthedocs_code_blocks(
                        result.html, min_length=3
                    )
                    extraction_method = f"html_{detected_doc_system}"
                else:
                    code_blocks = []
            except Exception:
                code_blocks = []

        # 2) Markdown path for everything else
        if not code_blocks and hasattr(result, "markdown") and result.markdown:
            try:
                md = result.markdown
                extraction_method = "markdown"

                # Check if this is Jupyter notebook content
                is_ipynb_url = url.lower().endswith(".ipynb")
                looks_like_json_notebook = (
                    md.strip().startswith("{") and '"cells"' in md[:2000]
                )

                if is_ipynb_url or looks_like_json_notebook:
                    # Try direct JSON extraction from markdown
                    if looks_like_json_notebook:
                        code_blocks = extract_jupyter_notebook_cells(md, min_length=3)
                        doc_system = "jupyter"
                        extraction_method = "markdown_jupyter_json"
                    else:
                        # It's an .ipynb URL but markdown is not JSON (GitHub page)
                        raw_url = github_blob_to_raw(url)
                        # Try crawler first (may not return JSON text)
                        raw_result = await crawler.arun(url=raw_url, config=run_config)
                        raw_content = getattr(raw_result, "text", None) or getattr(
                            raw_result, "markdown", None
                        )
                        raw_content = raw_content or ""

                        if not (
                            raw_content
                            and raw_content.strip().startswith("{")
                            and '"cells"' in raw_content[:2000]
                        ):
                            # Fallback to direct HTTP request
                            try:
                                resp = requests.get(raw_url, timeout=15)
                                if resp.status_code == 200:
                                    raw_content = resp.text
                                else:
                                    raw_content = ""
                            except Exception:
                                raw_content = ""

                        if (
                            raw_content
                            and raw_content.strip().startswith("{")
                            and '"cells"' in raw_content[:2000]
                        ):
                            code_blocks = extract_jupyter_notebook_cells(
                                raw_content, min_length=3
                            )
                            doc_system = "jupyter"
                            extraction_method = "raw_jupyter_json"
                        else:
                            code_blocks = []

                # If still nothing, extract markdown code blocks + commands
                if not code_blocks:
                    md_blocks = extract_markdown_code_blocks(md, min_length=3)
                    command_blocks = extract_command_examples(md, min_length=3)

                    # Filter overly long command blocks
                    command_blocks = [
                        b for b in command_blocks if len(b.get("code", "")) < 5000
                    ]

                    # Merge unique command blocks
                    if command_blocks:
                        existing_codes = {
                            block.get("code", "").strip() for block in md_blocks
                        }
                        for cmd_block in command_blocks:
                            if cmd_block.get("code", "").strip() not in existing_codes:
                                md_blocks.append(cmd_block)

                    code_blocks = md_blocks
                    doc_system = "unknown"
                    extraction_method = "markdown_generic"

            except Exception:
                code_blocks = []

        # Keep all code blocks (no deduplication)
        unique_code_blocks = code_blocks

        # Format extracted code blocks
        all_extracted_code: list[dict[str, Any]] = []
        for i, block in enumerate(unique_code_blocks):
            # Enhance context if missing
            context_before = block.get("context_before", "")
            context_after = block.get("context_after", "")

            if (
                (not context_before or not context_after)
                and hasattr(result, "markdown")
                and result.markdown
            ):
                code_snippet = block.get("code", "")[:50]
                position = result.markdown.find(code_snippet)
                if position > 0:
                    context_before = extract_smart_context_before(
                        result.markdown, position, max_chars=1000
                    )
                    end_position = position + len(block.get("code", ""))
                    context_after = extract_smart_context_after(
                        result.markdown, end_position, max_chars=1000
                    )

            # Build code block info
            code_info: dict[str, Any] = {
                "index": i + 1,
                "url": original_url,
                "type": block.get("type", "code"),
                "language": block.get("language", "unknown"),
                "code": block.get("code", ""),
                "context_before": context_before[:1000],
                "context_after": context_after[:1000],
                "extraction_method": extraction_method,
                "doc_system": doc_system,
                "summary": "",  # No LLM summary generation
            }

            # Add optional metadata
            if block.get("title"):
                code_info["title"] = block["title"]
            if block.get("description"):
                code_info["description"] = block["description"]

            all_extracted_code.append(code_info)

        # Return results
        return json.dumps(
            {
                "success": True,
                "url": original_url,
                "processed_url": url,
                "crawl_method": "single_page",
                "extraction_method": extraction_method,
                "doc_system": doc_system,
                "content_length": (
                    len(result.html)
                    if hasattr(result, "html") and result.html
                    else len(result.markdown)
                    if hasattr(result, "markdown") and result.markdown
                    else 0
                ),
                "code_blocks_found": len(all_extracted_code),
                "extracted_code": all_extracted_code,
                "message": (
                    f"Found {len(all_extracted_code)} code blocks"
                    if all_extracted_code
                    else "No code blocks found in page"
                ),
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps(
            {
                "success": False,
                "url": url,
                "crawl_method": "single_page",
                "error": str(e),
                "extracted_code": [],
            },
            indent=2,
        )


async def extract_code_smart_crawl(crawler: AsyncWebCrawler, url: str) -> str:
    """Extract code using smart crawl for complex sites."""
    try:
        # Store original URL
        original_url = url

        # Automatically convert GitHub blob URLs to raw URLs for better extraction
        url, is_valid = validate_and_normalize_url(url)
        if not is_valid:
            return json.dumps(
                {
                    "success": False,
                    "url": original_url,
                    "error": "Invalid URL provided for smart crawl",
                    "extracted_code": [],
                },
                indent=2,
            )

        crawl_results: list[dict[str, Any]] = []
        if is_txt(url):
            # For text files, use simple crawl
            crawl_results = await crawl_markdown_file(crawler, url)
        elif is_sitemap(url):
            # For sitemaps, extract URLs and crawl in parallel
            sitemap_urls = parse_sitemap(url)
            crawl_results = await crawl_batch(
                crawler, sitemap_urls[:5]
            )  # Limit to first 5 URLs
        else:
            # For regular webpages, crawl recursively
            crawl_results = await crawl_recursive_internal_links(
                crawler, [url], max_depth=2, max_concurrent=5
            )

        # Extract code from all crawled results
        all_extracted_code: list[dict[str, Any]] = []
        total_content_length = 0

        for result in crawl_results:
            if result.get("success") and result.get("markdown"):
                # Pass URL context to extraction
                result_url = result.get("url", url)
                code_blocks = extract_code_blocks(
                    result["markdown"], min_length=3, url=result_url
                )
                total_content_length += len(result["markdown"])

                for block in code_blocks:
                    code_info = {
                        "index": len(all_extracted_code) + 1,
                        "url": original_url,  # Use the original URL
                        "type": block.get("type", "unknown"),
                        "language": block.get("language", "unknown"),
                        "code": block["code"],
                        "context_before": block.get("context_before", "")[:1000],
                        "context_after": block.get("context_after", "")[:1000],
                        "summary": "",  # No LLM summary generation
                    }
                    all_extracted_code.append(code_info)

        return json.dumps(
            {
                "success": True,
                "url": original_url,
                "processed_url": url,
                "crawl_method": "smart_crawl",
                "content_length": total_content_length,
                "code_blocks_found": len(all_extracted_code),
                "extracted_code": all_extracted_code,
            },
            indent=2,
        )

    except Exception as e:
        return json.dumps(
            {
                "success": False,
                "url": original_url,
                "error": str(e),
                "extracted_code": [],
            },
            indent=2,
        )


async def create_crawler() -> AsyncWebCrawler:
    """Create and initialize a new AsyncWebCrawler instance."""
    browser_config = BrowserConfig(headless=True, verbose=False)
    crawler = AsyncWebCrawler(config=browser_config)
    await crawler.start()
    return crawler


async def close_crawler(crawler: AsyncWebCrawler) -> None:
    """Close an AsyncWebCrawler instance."""
    try:
        await crawler.close()
    except Exception:
        pass


__all__ = [
    "is_sitemap",
    "is_txt",
    "parse_sitemap",
    "github_blob_to_raw",
    "validate_and_normalize_url",
    "smart_chunk_markdown",
    "crawl_batch",
    "crawl_recursive_internal_links",
    "crawl_markdown_file",
    "extract_code_single_page",
    "extract_code_smart_crawl",
    "create_crawler",
    "close_crawler",
]
