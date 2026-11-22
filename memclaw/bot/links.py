"""Link summaries: find the URLs in a message, fetch each page and summarize it."""

from __future__ import annotations

import re
from urllib.parse import urlparse

from bs4 import BeautifulSoup
from curl_cffi.requests import AsyncSession
from loguru import logger

from ..openai_client import OpenAIClient


class LinkProcessor:
    """Extract URLs from text, fetch page content, and generate summaries."""

    def __init__(self, openai: OpenAIClient):
        self.openai = openai
        self._url_re = re.compile(r"https?://[^\s<>\"')\]]+", re.IGNORECASE)

    def extract_urls(self, text: str) -> list[str]:
        urls = self._url_re.findall(text)
        seen: set[str] = set()
        cleaned: list[str] = []
        for url in urls:
            url = url.rstrip(".,;:!?")
            try:
                parsed = urlparse(url)
                if parsed.scheme and parsed.netloc and url not in seen:
                    seen.add(url)
                    cleaned.append(url)
            except Exception:
                continue
        return cleaned

    async def fetch_content(self, url: str, timeout: float = 10.0) -> str | None:
        try:
            async with AsyncSession() as session:
                response = await session.get(
                    url,
                    timeout=timeout,
                    allow_redirects=True,
                    impersonate="chrome120",
                )
                response.raise_for_status()

                if "text/html" not in response.headers.get("content-type", ""):
                    return None

                return self._extract_text(response.text)
        except Exception as exc:
            logger.warning("fetch_content failed for {url}: {exc}", url=url, exc=exc)
            return None

    @staticmethod
    def _extract_text(html: str, max_chars: int = 5000) -> str:
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "nav", "footer", "header", "aside"]):
            tag.decompose()

        main = (
            soup.find("article")
            or soup.find("main")
            or soup.find(class_=re.compile(r"content|article|post", re.I))
            or soup.find("body")
        )
        text = (
            main.get_text(separator=" ", strip=True)
            if main
            else soup.get_text(separator=" ", strip=True)
        )
        text = re.sub(r"\s+", " ", text).strip()
        return text[:max_chars]

    async def summarize(self, content: str, url: str) -> str:
        try:
            return await self.openai.chat([
                {
                    "role": "system",
                    "content": (
                        "Summarize this web content concisely. "
                        "Focus on main points. Keep it under 50 tokens."
                    ),
                },
                {"role": "user", "content": f"Summarize content from {url}:\n\n{content}"},
            ])
        except Exception as e:
            return f"Could not summarize: {e}"

    async def process_links(self, text: str) -> list[dict]:
        """Extract URLs, fetch content, summarize. Returns [{url, summary}]."""
        urls = self.extract_urls(text)
        results: list[dict] = []

        for url in urls:
            content = await self.fetch_content(url)
            summary = await self.summarize(content, url) if content else None
            results.append({"url": url, "summary": summary})

        return results
