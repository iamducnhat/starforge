from __future__ import annotations

import html
import math
import re
from typing import Any
from urllib.parse import parse_qs, quote_plus, urldefrag, urljoin, urlparse
from urllib.request import Request, urlopen


def _http_get(url: str, timeout: int = 15) -> str:
    req = Request(url, headers={"User-Agent": "LocalCodingAssistant/1.0"})
    with urlopen(
        req, timeout=timeout
    ) as resp:  # nosec B310 - controlled URL construction
        data = resp.read()
    return data.decode("utf-8", errors="ignore")


def _strip_tags(text: str) -> str:
    text = re.sub(r"<[^>]+>", " ", text)
    text = html.unescape(text)
    return re.sub(r"\s+", " ", text).strip()


def _decode_duckduckgo_redirect(url: str) -> str:
    parsed = urlparse(url)
    if parsed.path.startswith("/l/"):
        q = parse_qs(parsed.query)
        if "uddg" in q and q["uddg"]:
            return q["uddg"][0]
    return url


def _is_http_url(url: str) -> bool:
    scheme = urlparse(url).scheme.lower()
    return scheme in {"http", "https"}


def _extract_page_text(html_text: str, max_chars: int = 2200) -> str:
    if not html_text:
        return ""
    cleaned = re.sub(
        r"<(script|style|noscript|svg|header|footer|nav|form|iframe)[^>]*>.*?</\1>",
        " ",
        html_text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    text = _strip_tags(cleaned)
    if not text:
        return ""
    return text[:max_chars].strip()


def _extract_title(html_text: str) -> str:
    if not html_text:
        return ""
    m = re.search(
        r"<title[^>]*>(.*?)</title>", html_text, flags=re.IGNORECASE | re.DOTALL
    )
    if not m:
        return ""
    return _strip_tags(m.group(1)).strip()


def _canonicalize_url(url: str, base_url: str | None = None) -> str:
    u = url.strip()
    if not u:
        return ""
    if base_url:
        u = urljoin(base_url, u)
    u, _frag = urldefrag(u)
    parsed = urlparse(u)
    if parsed.scheme.lower() not in {"http", "https"}:
        return ""
    if not parsed.netloc:
        return ""
    return u


def _extract_links(html_text: str, base_url: str, max_links: int = 200) -> list[str]:
    if not html_text:
        return []
    href_pattern = re.compile(r'<a[^>]*href=["\'](.*?)["\']', re.IGNORECASE | re.DOTALL)
    out: list[str] = []
    seen: set[str] = set()
    for href in href_pattern.findall(html_text):
        url = _canonicalize_url(href, base_url=base_url)
        if not url or url in seen:
            continue
        seen.add(url)
        out.append(url)
        if len(out) >= max_links:
            break
    return out


def _complex_query(query: str) -> bool:
    q = query.lower()
    markers = (
        "latest",
        "current",
        "today",
        "news",
        "research",
        "official",
        "compare",
        "comprehensive",
        "best",
        "guide",
        "overview",
        "explain",
    )
    return any(m in q for m in markers) or len(q.split()) >= 8


def _resolve_profile(
    query: str,
    level: str = "auto",
    max_results: int | None = None,
    fetch_top_pages: int | None = None,
    page_timeout: int | None = None,
) -> dict[str, int | str]:
    level_raw = (level or "auto").strip().lower()
    if level_raw not in {"auto", "quick", "balanced", "deep"}:
        level_raw = "auto"

    if level_raw == "auto":
        level_used = "deep" if _complex_query(query) else "balanced"
    else:
        level_used = level_raw

    defaults = {
        "quick": {"max_results": 5, "fetch_top_pages": 2, "page_timeout": 8},
        "balanced": {"max_results": 12, "fetch_top_pages": 5, "page_timeout": 10},
        "deep": {"max_results": 25, "fetch_top_pages": 10, "page_timeout": 12},
    }
    profile = defaults[level_used].copy()
    if isinstance(max_results, int):
        profile["max_results"] = max(1, min(max_results, 50))
    if isinstance(fetch_top_pages, int):
        profile["fetch_top_pages"] = max(0, min(fetch_top_pages, 25))
    if isinstance(page_timeout, int):
        profile["page_timeout"] = max(3, min(page_timeout, 30))

    # DuckDuckGo HTML pagination uses `s` offsets; scan enough pages to cover target volume.
    profile["result_pages"] = max(1, min(8, math.ceil(profile["max_results"] / 10)))
    profile["level_requested"] = level_raw
    profile["level_used"] = level_used
    return profile


def _parse_ddg_html_results(page: str) -> list[dict[str, str]]:
    link_pattern = re.compile(
        r'<a[^>]*class="result__a"[^>]*href="(.*?)"[^>]*>(.*?)</a>',
        re.IGNORECASE | re.DOTALL,
    )
    snippet_pattern = re.compile(
        r'<a[^>]*class="result__snippet"[^>]*>(.*?)</a>|<div[^>]*class="result__snippet"[^>]*>(.*?)</div>',
        re.IGNORECASE | re.DOTALL,
    )
    links = link_pattern.findall(page)
    snippets_raw = snippet_pattern.findall(page)
    snippets = [a or b for a, b in snippets_raw]

    out: list[dict[str, str]] = []
    for idx, (href, title_html) in enumerate(links):
        snippet_html = snippets[idx] if idx < len(snippets) else ""
        out.append(
            {
                "title": _strip_tags(title_html),
                "url": _decode_duckduckgo_redirect(href),
                "snippet": _strip_tags(snippet_html),
            }
        )
    return out


def _parse_ddg_lite_results(page: str) -> list[dict[str, str]]:
    # Fallback parser for lite endpoint when HTML endpoint shape changes.
    anchor_pattern = re.compile(
        r'<a[^>]*href="(https?://[^"]+)"[^>]*>(.*?)</a>', re.IGNORECASE | re.DOTALL
    )
    out: list[dict[str, str]] = []
    for href, title_html in anchor_pattern.findall(page):
        out.append(
            {"title": _strip_tags(title_html), "url": href.strip(), "snippet": ""}
        )
    return out


def _search_duckduckgo(
    query: str, max_results: int, result_pages: int, timeout: int
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    results: list[dict[str, Any]] = []
    seen_urls: set[str] = set()
    pages_scanned = 0
    endpoint = "duckduckgo_html"
    had_error = False

    for page_idx in range(result_pages):
        offset = page_idx * 30
        url = f"https://duckduckgo.com/html/?q={quote_plus(query)}&s={offset}"
        try:
            html_page = _http_get(url, timeout=timeout)
            pages_scanned += 1
            page_results = _parse_ddg_html_results(html_page)
            if not page_results:
                break
            for item in page_results:
                page_url = str(item.get("url", "")).strip()
                if not page_url or page_url in seen_urls:
                    continue
                seen_urls.add(page_url)
                results.append(item)
                if len(results) >= max_results:
                    return results, {
                        "engine": endpoint,
                        "pages_scanned": pages_scanned,
                        "had_error": had_error,
                    }
        except Exception:
            had_error = True
            break

    if results:
        return results, {
            "engine": endpoint,
            "pages_scanned": pages_scanned,
            "had_error": had_error,
        }

    # Fallback to lite endpoint.
    endpoint = "duckduckgo_lite"
    try:
        lite_url = f"https://lite.duckduckgo.com/lite/?q={quote_plus(query)}"
        page = _http_get(lite_url, timeout=timeout)
        pages_scanned += 1
        for item in _parse_ddg_lite_results(page):
            page_url = str(item.get("url", "")).strip()
            if not page_url or page_url in seen_urls:
                continue
            seen_urls.add(page_url)
            results.append(item)
            if len(results) >= max_results:
                break
    except Exception:
        had_error = True

    return results, {
        "engine": endpoint,
        "pages_scanned": pages_scanned,
        "had_error": had_error,
    }


def search_web(
    query: str,
    max_results: int | None = None,
    fetch_top_pages: int | None = None,
    page_timeout: int | None = None,
    level: str = "auto",
) -> dict[str, Any]:
    if not query.strip():
        return {
            "query": query,
            "results": [],
            "meta": {"level_requested": level, "level_used": "quick"},
        }

    profile = _resolve_profile(
        query=query,
        level=level,
        max_results=max_results,
        fetch_top_pages=fetch_top_pages,
        page_timeout=page_timeout,
    )

    max_results_eff = int(profile["max_results"])
    fetch_top_pages_eff = int(profile["fetch_top_pages"])
    page_timeout_eff = int(profile["page_timeout"])
    result_pages_eff = int(profile["result_pages"])

    base_results, search_meta = _search_duckduckgo(
        query=query,
        max_results=max_results_eff,
        result_pages=result_pages_eff,
        timeout=page_timeout_eff,
    )

    results: list[dict[str, Any]] = []
    for idx, item in enumerate(base_results[:max_results_eff], start=1):
        results.append(
            {
                "rank": idx,
                "title": str(item.get("title", "")).strip(),
                "url": str(item.get("url", "")).strip(),
                "snippet": str(item.get("snippet", "")).strip(),
            }
        )

    # Fetch and parse top pages so downstream steps use page content, not snippets only.
    for item in results[:fetch_top_pages_eff]:
        page_url = str(item.get("url", "")).strip()
        if not page_url:
            item["page_fetched"] = False
            item["page_error"] = "empty_url"
            continue
        if not _is_http_url(page_url):
            item["page_fetched"] = False
            item["page_error"] = "unsupported_scheme"
            continue
        try:
            page_html = _http_get(page_url, timeout=page_timeout_eff)
            item["page_fetched"] = True
            page_excerpt = _extract_page_text(page_html)
            if len(page_excerpt) < 80:
                page_excerpt = str(item.get("snippet", "")).strip()
            item["page_excerpt"] = page_excerpt
            item["code_snippets"] = extract_code_snippets(
                page_html, max_snippets=4, max_chars=700
            )
        except Exception as e:  # pragma: no cover
            item["page_fetched"] = False
            item["page_error"] = str(e)

    return {
        "query": query,
        "results": results,
        "meta": {
            "engine": search_meta.get("engine", "duckduckgo_html"),
            "pages_scanned": int(search_meta.get("pages_scanned", 0)),
            "level_requested": str(profile["level_requested"]),
            "level_used": str(profile["level_used"]),
            "max_results": max_results_eff,
            "fetch_top_pages": fetch_top_pages_eff,
            "page_timeout": page_timeout_eff,
            "had_search_error": bool(search_meta.get("had_error", False)),
        },
    }


def read_web(
    url: str,
    timeout: int = 12,
    max_chars: int = 12000,
    include_links: bool = True,
    max_links: int = 120,
) -> dict[str, Any]:
    target = _canonicalize_url(url)
    if not target:
        return {"ok": False, "error": "invalid url"}
    try:
        html_text = _http_get(target, timeout=timeout)
        text = _extract_page_text(html_text, max_chars=max_chars)
        result: dict[str, Any] = {
            "ok": True,
            "url": target,
            "title": _extract_title(html_text),
            "text": text,
            "code_snippets": extract_code_snippets(
                html_text, max_snippets=8, max_chars=700
            ),
        }
        if include_links:
            result["links"] = _extract_links(
                html_text, base_url=target, max_links=max_links
            )
        return result
    except Exception as e:
        return {"ok": False, "url": target, "error": str(e)}


def scrape_web(
    start_url: str,
    max_pages: int = 20,
    max_depth: int = 2,
    same_domain_only: bool = True,
    include_external: bool = False,
    timeout: int = 10,
    max_links_per_page: int = 120,
) -> dict[str, Any]:
    root = _canonicalize_url(start_url)
    if not root:
        return {"ok": False, "error": "invalid start_url"}

    root_domain = urlparse(root).netloc.lower()
    max_pages = max(1, min(max_pages, 200))
    max_depth = max(0, min(max_depth, 6))

    queue: list[tuple[str, int]] = [(root, 0)]
    visited: set[str] = set()
    pages: list[dict[str, Any]] = []
    all_links: set[str] = set()

    while queue and len(visited) < max_pages:
        current, depth = queue.pop(0)
        if current in visited:
            continue
        visited.add(current)

        try:
            html_text = _http_get(current, timeout=timeout)
            links = _extract_links(
                html_text, base_url=current, max_links=max_links_per_page
            )
            all_links.update(links)

            page_record = {
                "url": current,
                "depth": depth,
                "title": _extract_title(html_text),
                "excerpt": _extract_page_text(html_text, max_chars=1500),
                "links_count": len(links),
                "links": links,
            }
            pages.append(page_record)

            if depth < max_depth:
                for link in links:
                    link_domain = urlparse(link).netloc.lower()
                    same_domain = link_domain == root_domain
                    if same_domain_only and not same_domain:
                        continue
                    if not include_external and not same_domain:
                        continue
                    if link not in visited:
                        queue.append((link, depth + 1))
        except Exception as e:
            pages.append(
                {
                    "url": current,
                    "depth": depth,
                    "error": str(e),
                    "links_count": 0,
                    "links": [],
                }
            )

    return {
        "ok": True,
        "start_url": root,
        "domain": root_domain,
        "visited_pages": len(visited),
        "pages": pages,
        "unique_links": sorted(all_links),
    }


def extract_code_snippets(
    html_text: str, max_snippets: int = 8, max_chars: int = 700
) -> list[str]:
    if not html_text:
        return []

    patterns = [
        re.compile(
            r"<pre[^>]*>\s*<code[^>]*>(.*?)</code>\s*</pre>", re.IGNORECASE | re.DOTALL
        ),
        re.compile(r"<pre[^>]*>(.*?)</pre>", re.IGNORECASE | re.DOTALL),
        re.compile(r"<code[^>]*>(.*?)</code>", re.IGNORECASE | re.DOTALL),
    ]

    snippets: list[str] = []
    seen = set()

    for pattern in patterns:
        for match in pattern.findall(html_text):
            clean = _strip_tags(match).strip()
            if len(clean) < 20 or clean in seen:
                continue
            seen.add(clean)
            snippets.append(clean[:max_chars])
            if len(snippets) >= max_snippets:
                return snippets

    return snippets
