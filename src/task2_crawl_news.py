"""
Task 2 — Crawl five official IELTS Writing guidance webpages.

Stage 1: requests (+ trafilatura / BeautifulSoup)
Stage 2: Crawl4AI / Playwright fallback when Stage 1 fails
Never fabricate content. Never bypass CAPTCHA/Cloudflare challenges.
"""

from __future__ import annotations

import asyncio
import json
import re
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from .env_utils import load_repo_env

load_repo_env()

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "landing" / "news"
INSPECTION_DIR = PROJECT_ROOT / "data" / "inspection"

ARTICLE_URLS = [
    "https://ielts.idp.com/prepare/article-writing-task-2-why-cant-i-get-a-band-8",
    "https://ielts.idp.com/prepare/article-ielts-writing-task-2-8-steps-to-band-8",
    "https://takeielts.britishcouncil.org/blog/ielts-writing-task-2-tips",
    "https://takeielts.britishcouncil.org/blog/how-to-write-an-english-essay-for-ielts",
    "https://takeielts.britishcouncil.org/take-ielts/prepare/free-ielts-english-practice-tests/writing/academic/task-2",
]

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/122.0.0.0 Safari/537.36"
)
CONNECT_TIMEOUT = 20
READ_TIMEOUT = 60
MIN_CONTENT_CHARS = 500
MAX_RETRIES = 2


def setup_directory():
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    INSPECTION_DIR.mkdir(parents=True, exist_ok=True)


def _source_org_for_url(url: str) -> str:
    host = urlparse(url).netloc.lower()
    if "idp.com" in host:
        return "IDP IELTS"
    if "britishcouncil.org" in host:
        return "British Council"
    return "IELTS"


def _clean_markdown(text: str) -> str:
    if not text:
        return ""
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    drop_patterns = [
        r"(?i)^\s*accept (all )?cookies.*$",
        r"(?i)^\s*manage cookies.*$",
        r"(?i)^\s*sign in\s*$",
        r"(?i)^\s*log in\s*$",
        r"(?i)^\s*create an account.*$",
        r"(?i)^\s*share (this|on).*$",
        r"(?i)^\s*follow us.*$",
        r"(?i)^\s*subscribe.*newsletter.*$",
        r"(?i)^\s*cookie policy.*$",
        r"(?i)^\s*related (articles|content).*$",
    ]
    lines = []
    for line in text.split("\n"):
        if any(re.match(p, line.strip()) for p in drop_patterns):
            continue
        lines.append(line.rstrip())
    text = "\n".join(lines)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _looks_like_blocked(text: str) -> bool:
    t = (text or "").lower()
    markers = [
        "access denied",
        "just a moment",
        "cf-browser-verification",
        "attention required",
        "cloudflare",
        "captcha",
        "enable javascript and cookies",
        "checking your browser",
    ]
    return any(m in t for m in markers)


def validate_crawled_article(article: dict) -> tuple[bool, str]:
    if article.get("status") != "success":
        return False, article.get("error") or "status_not_success"
    title = (article.get("title") or "").strip()
    content = (article.get("content_markdown") or "").strip()
    if not title:
        return False, "empty_title"
    if len(content) < MIN_CONTENT_CHARS:
        return False, f"content_too_short:{len(content)}"
    if _looks_like_blocked(content) or _looks_like_blocked(title):
        return False, "access_denied_or_challenge"
    if content.lower() in {"placeholder", "lorem ipsum"}:
        return False, "placeholder_content"
    # Reject navigation-only pages
    useful = re.sub(r"\s+", " ", content)
    if useful.count(" ") < 40:
        return False, "navigation_only"
    return True, "ok"


def _extract_with_trafilatura(html: str, url: str) -> tuple[str, Optional[str]]:
    try:
        import trafilatura

        extracted = trafilatura.extract(
            html,
            url=url,
            include_comments=False,
            include_tables=True,
            output_format="markdown",
            favor_recall=True,
        )
        meta = trafilatura.extract_metadata(html)
        title = meta.title if meta else None
        return _clean_markdown(extracted or ""), title
    except Exception:
        return "", None


def _extract_with_bs4(html: str, url: str) -> tuple[str, Optional[str]]:
    soup = BeautifulSoup(html, "html.parser")
    for tag in soup(["script", "style", "noscript", "svg", "iframe"]):
        tag.decompose()
    for tag in soup.find_all(["nav", "footer", "aside", "form", "button", "header"]):
        tag.decompose()

    main = None
    for selector in [
        "article",
        "main",
        "[role=main]",
        ".content",
        ".article-content",
        ".node-content",
        ".field--name-body",
    ]:
        main = soup.select_one(selector)
        if main:
            break
    if main is None:
        main = soup.body or soup

    title = None
    if soup.title and soup.title.string:
        title = soup.title.string.strip()
    h1 = main.find("h1") if main else None
    if h1 and h1.get_text(strip=True):
        title = h1.get_text(" ", strip=True)

    parts: list[str] = []
    if title:
        parts.append(f"# {title}")
    for el in main.find_all(["h1", "h2", "h3", "h4", "p", "li", "table", "ul", "ol"]):
        name = el.name
        text = el.get_text(" ", strip=True)
        if not text:
            continue
        if name == "h1":
            continue
        if name in {"h2", "h3", "h4"}:
            parts.append(f"{'#' * int(name[1])} {text}")
        elif name == "li":
            parts.append(f"- {text}")
        elif name == "table":
            rows = []
            for tr in el.find_all("tr"):
                cells = [c.get_text(" ", strip=True) for c in tr.find_all(["th", "td"])]
                if cells:
                    rows.append("| " + " | ".join(cells) + " |")
            if rows:
                parts.append("\n".join(rows))
        elif name in {"ul", "ol"}:
            continue
        else:
            parts.append(text)
    content = _clean_markdown("\n\n".join(parts))
    if content and "Source URL:" not in content:
        content = content + f"\n\n---\nSource URL: {url}\n"
    return content, title


def _crawl_with_requests(url: str) -> dict:
    session = requests.Session()
    headers = {
        "User-Agent": USER_AGENT,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "en-US,en;q=0.9",
        "Cache-Control": "no-cache",
        "Connection": "keep-alive",
    }
    last_exc: Exception | None = None
    resp = None
    for attempt in range(MAX_RETRIES):
        try:
            resp = session.get(
                url,
                headers=headers,
                timeout=(CONNECT_TIMEOUT, READ_TIMEOUT),
                allow_redirects=True,
            )
            if resp.status_code >= 500:
                raise requests.HTTPError(f"server_error:{resp.status_code}")
            resp.raise_for_status()
            ctype = (resp.headers.get("Content-Type") or "").lower()
            if "html" not in ctype and "text" not in ctype:
                raise ValueError(f"non_html_response:{ctype or 'unknown'}")
            break
        except Exception as exc:
            last_exc = exc
            resp = None
            time.sleep(2 ** attempt)
    if resp is None:
        raise last_exc or RuntimeError("request_failed")

    html = resp.text
    if _looks_like_blocked(html):
        raise ValueError("access_denied_or_challenge")

    method = "requests_bs4"
    content, title = _extract_with_trafilatura(html, url)
    if len(content) >= MIN_CONTENT_CHARS:
        method = "requests_trafilatura"
    else:
        content, title_bs = _extract_with_bs4(html, url)
        title = title or title_bs
        method = "requests_bs4"

    article = {
        "url": url,
        "title": title or url,
        "source_org": _source_org_for_url(url),
        "date_crawled": datetime.now(timezone.utc).isoformat(),
        "status": "success",
        "http_status": resp.status_code,
        "crawler_method": method,
        "content_markdown": content if "Source URL:" in content else content + f"\n\n---\nSource URL: {url}\n",
        "error": None,
    }
    ok, reason = validate_crawled_article(article)
    if not ok:
        raise ValueError(reason)
    return article


async def _crawl_with_crawl4ai(url: str) -> dict:
    from crawl4ai import AsyncWebCrawler

    async with AsyncWebCrawler() as crawler:
        result = await crawler.arun(url=url)
        markdown = ""
        if hasattr(result, "markdown") and result.markdown:
            markdown = str(result.markdown)
        elif hasattr(result, "cleaned_html") and result.cleaned_html:
            markdown, _ = _extract_with_bs4(result.cleaned_html, url)
        markdown = _clean_markdown(markdown)
        title = None
        meta = getattr(result, "metadata", None) or {}
        if isinstance(meta, dict):
            title = meta.get("title")
        if not title:
            m = re.search(r"^#\s+(.+)$", markdown, re.MULTILINE)
            title = m.group(1).strip() if m else url
        if "Source URL:" not in markdown:
            markdown = markdown + f"\n\n---\nSource URL: {url}\n"
        article = {
            "url": url,
            "title": title,
            "source_org": _source_org_for_url(url),
            "date_crawled": datetime.now(timezone.utc).isoformat(),
            "status": "success",
            "http_status": 200,
            "crawler_method": "crawl4ai",
            "content_markdown": markdown,
            "error": None,
        }
        ok, reason = validate_crawled_article(article)
        if not ok:
            raise ValueError(reason)
        return article


async def crawl_article(url: str) -> dict:
    """Crawl one URL with multi-stage fallback. Never fabricates content."""
    # Stage 1: requests
    try:
        return await asyncio.to_thread(_crawl_with_requests, url)
    except Exception as req_exc:
        print(f"  [WARN] requests stage failed ({req_exc}); trying Crawl4AI/Playwright")

    # Stage 2: Crawl4AI / Playwright
    try:
        return await asyncio.wait_for(_crawl_with_crawl4ai(url), timeout=60)
    except Exception as crawl_exc:
        return {
            "url": url,
            "title": None,
            "source_org": _source_org_for_url(url),
            "date_crawled": datetime.now(timezone.utc).isoformat(),
            "status": "failed",
            "http_status": None,
            "crawler_method": "failed",
            "content_markdown": "",
            "error": str(crawl_exc)[:500],
        }


def _should_skip_existing_success(path: Path, url: str) -> bool:
    """Keep successful IDP JSON unless invalid."""
    if not path.exists():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if data.get("url") != url:
        return False
    ok, _ = validate_crawled_article(data)
    return ok


async def crawl_all(force: bool = False):
    setup_directory()
    successes = 0
    failures: list[str] = []
    report_rows: list[dict[str, Any]] = []

    for i, url in enumerate(ARTICLE_URLS, 1):
        filename = f"article_{i:02d}.json"
        filepath = DATA_DIR / filename
        print(f"[{i}/{len(ARTICLE_URLS)}] Crawling: {url}")

        if not force and _should_skip_existing_success(filepath, url):
            data = json.loads(filepath.read_text(encoding="utf-8"))
            successes += 1
            print(f"  [OK] Keeping existing valid success: {filepath.name}")
            report_rows.append(
                {
                    "url": url,
                    "source_org": data.get("source_org"),
                    "status": "success",
                    "crawler_method": data.get("crawler_method") or "cached",
                    "http_status": data.get("http_status"),
                    "content_length": len(data.get("content_markdown") or ""),
                    "failure_reason": None,
                    "output_filename": filename,
                    "reused_existing": True,
                }
            )
            continue

        article = await crawl_article(url)
        ok, reason = validate_crawled_article(article)
        if not ok:
            article["status"] = "failed"
            article["error"] = reason
            article["content_markdown"] = ""
            failures.append(url)
            print(f"  [FAIL] {reason}")
        else:
            successes += 1
            print(f"  [OK] Saved success via {article.get('crawler_method')}")

        filepath.write_text(json.dumps(article, ensure_ascii=False, indent=2), encoding="utf-8")
        report_rows.append(
            {
                "url": url,
                "source_org": article.get("source_org"),
                "status": article.get("status"),
                "crawler_method": article.get("crawler_method"),
                "http_status": article.get("http_status"),
                "content_length": len(article.get("content_markdown") or ""),
                "failure_reason": article.get("error"),
                "output_filename": filename,
                "reused_existing": False,
            }
        )

    report = {
        "crawled_at": datetime.now(timezone.utc).isoformat(),
        "successful_web_articles": successes,
        "total_urls": len(ARTICLE_URLS),
        "failed_urls": failures,
        "articles": report_rows,
    }
    out = INSPECTION_DIR / "web_crawl_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")

    print("\n" + "=" * 60)
    print(f"successful web articles: {successes} / {len(ARTICLE_URLS)}")
    if failures:
        print("Failed URLs:")
        for u in failures:
            print(f"  - {u}")
    print(f"[OK] Wrote {out}")


if __name__ == "__main__":
    asyncio.run(crawl_all())
