"""Fetch a web page and convert to LLM-friendly markdown."""

import asyncio
import re

from crawl4ai import AsyncWebCrawler


def crawl(url: str) -> str:
    """Fetch URL and return cleaned markdown text."""
    return asyncio.run(_crawl_and_clean(url))


async def _crawl_and_clean(url: str) -> str:
    """Fetch URL with crawl4ai and return cleaned markdown."""
    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
    md = result.markdown
    # collapse runs of 3+ blank lines into 2
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()
