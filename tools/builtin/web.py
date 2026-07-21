"""Built-in web tools: fetch a URL and search the web."""

from __future__ import annotations

import html
import json
import re
import urllib.parse
import urllib.request

from tools.decorator import tool

_UA = "Mozilla/5.0 (compatible; auto-researcher/1.0)"
_DEFAULT_TIMEOUT = 15


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode entities."""
    text = re.sub(r"<script[^>]*>.*?</script>", "", text, flags=re.S | re.I)
    text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.S | re.I)
    text = re.sub(r"<[^>]+>", "", text)
    text = html.unescape(text)
    text = re.sub(r"\s{3,}", "\n\n", text)
    return text.strip()


@tool(name="web_fetch", description="Fetch a URL and return its text content", kind="action")
def web_fetch(workspace: str, url: str, max_chars: int = 32000) -> dict:
    """Fetch a web page or file and return the text content.

    Strips HTML tags; returns raw content for plain text / JSON / code files.

    Args:
        workspace: auto-injected — do NOT pass.
        url: Full URL to fetch (must start with http:// or https://).
        max_chars: Maximum characters to return (default 32000).

    Returns:
        {"url": str, "content": str, "content_type": str, "chars": int}
    """
    if not url.startswith(("http://", "https://")):
        return {"error": "URL must start with http:// or https://"}

    req = urllib.request.Request(url, headers={"User-Agent": _UA})
    try:
        with urllib.request.urlopen(req, timeout=_DEFAULT_TIMEOUT) as resp:
            content_type = resp.headers.get("Content-Type", "")
            raw = resp.read(max_chars * 4)  # read a bit extra before trimming
    except Exception as exc:
        return {"error": f"fetch failed: {exc}"}

    # Decode
    charset = "utf-8"
    if "charset=" in content_type:
        charset = content_type.split("charset=")[-1].split(";")[0].strip() or "utf-8"

    try:
        text = raw.decode(charset, errors="replace")
    except (LookupError, UnicodeDecodeError):
        text = raw.decode("utf-8", errors="replace")

    # Strip HTML for HTML responses
    ct_lower = content_type.lower()
    if "html" in ct_lower:
        text = _strip_html(text)

    text = text[:max_chars]
    return {
        "url": url,
        "content": text,
        "content_type": content_type,
        "chars": len(text),
    }


@tool(name="web_search", description="Search the web and return result snippets", kind="action")
def web_search(workspace: str, query: str, max_results: int = 8) -> dict:
    """Search the web using DuckDuckGo and return titles, URLs, and snippets.

    Args:
        workspace: auto-injected — do NOT pass.
        query: Search query string.
        max_results: Number of results to return (default 8, max 20).

    Returns:
        {"query": str, "results": [{"title": str, "url": str, "snippet": str}]}
    """
    max_results = min(max_results, 20)

    # ── Try DuckDuckGo Lite (HTML) for real search results ─────────────────
    try:
        encoded = urllib.parse.urlencode({"q": query})
        url = f"https://lite.duckduckgo.com/lite/?{encoded}"
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=_DEFAULT_TIMEOUT) as resp:
            body = resp.read(131072).decode("utf-8", errors="replace")
        results = _parse_ddg_lite(body, max_results)
        if results:
            return {"query": query, "results": results, "source": "duckduckgo"}
    except Exception:
        pass

    # ── Fallback: DuckDuckGo JSON (instant answers / related topics) ────────
    try:
        params = urllib.parse.urlencode({
            "q": query, "format": "json",
            "no_html": "1", "skip_disambig": "1",
        })
        url = f"https://api.duckduckgo.com/?{params}"
        req = urllib.request.Request(url, headers={"User-Agent": _UA})
        with urllib.request.urlopen(req, timeout=_DEFAULT_TIMEOUT) as resp:
            data = json.loads(resp.read())
        results = _parse_ddg_json(data, max_results)
        return {"query": query, "results": results, "source": "duckduckgo_instant"}
    except Exception as exc:
        return {"query": query, "results": [], "error": f"search failed: {exc}"}


# ── Parsers ─────────────────────────────────────────────────────────────────────

def _parse_ddg_lite(html_body: str, max_results: int) -> list[dict]:
    """Parse DuckDuckGo Lite HTML into result list."""
    results = []
    # DDG Lite results look like:
    #   <a class="result-link" href="URL">TITLE</a>
    #   <td class="result-snippet">SNIPPET</td>
    link_re = re.compile(r'<a[^>]+class="result-link"[^>]*href="([^"]+)"[^>]*>(.*?)</a>', re.S)
    snip_re = re.compile(r'class="result-snippet"[^>]*>(.*?)</td>', re.S)

    links = link_re.findall(html_body)
    snips = snip_re.findall(html_body)

    for i, (url, title) in enumerate(links[:max_results]):
        snippet = snips[i] if i < len(snips) else ""
        results.append({
            "title": _strip_html(title).strip(),
            "url": html.unescape(url),
            "snippet": _strip_html(snippet).strip()[:300],
        })
    return results


def _parse_ddg_json(data: dict, max_results: int) -> list[dict]:
    """Parse DuckDuckGo JSON instant-answer API response."""
    results = []

    if data.get("AbstractText"):
        results.append({
            "title": data.get("Heading", ""),
            "url": data.get("AbstractURL", ""),
            "snippet": data["AbstractText"][:400],
        })

    for topic in data.get("RelatedTopics", [])[:max_results]:
        if not isinstance(topic, dict) or "Text" not in topic:
            continue
        text = topic.get("Text", "")
        first_url = topic.get("FirstURL", "")
        title = text.split(" - ")[0] if " - " in text else text[:60]
        results.append({
            "title": title.strip(),
            "url": first_url,
            "snippet": text[:300],
        })
        if len(results) >= max_results:
            break

    return results
