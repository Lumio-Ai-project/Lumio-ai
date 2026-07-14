import json
import re
from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from pathlib import Path
from xml.etree import ElementTree

from bs4 import BeautifulSoup

from preprocessing.cleaner import clean_text, extract_main_content
from scraper.base import Article, BaseScraper, ScrapeResult
from scraper.http_client import fetch_url_safe

BBC_LINKS_FILE = Path(__file__).resolve().parent.parent / "bbclinks.txt"

BBC_ARTICLE_PATTERNS = (
    re.compile(r"^https://www\.bbc\.com/news/articles/[a-z0-9]+$"),
    re.compile(r"^https://www\.bbc\.com/news/live/[a-z0-9]+$"),
    re.compile(r"^https://www\.bbc\.com/[a-z-]+/article/\d{8}-"),
    re.compile(r"^https://www\.bbc\.com/sport/[a-z-/]+/articles/[a-z0-9]+$"),
    re.compile(r"^https://www\.bbc\.com/future/article/\d{8}-"),
)

BBC_CATEGORY_MAP = {
    "news": "general",
    "sport": "sport",
    "business": "business",
    "technology": "technology",
    "health": "health",
    "culture": "culture",
    "travel": "travel",
    "future": "science",
}


def load_bbc_article_urls(links_file: Path = BBC_LINKS_FILE) -> list[str]:
    if not links_file.exists():
        return []

    urls: list[str] = []
    seen: set[str] = set()

    for line in links_file.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("substring filter"):
            continue
        if not _is_bbc_article_url(line):
            continue
        if line in seen:
            continue
        seen.add(line)
        urls.append(line)

    return urls


def _is_bbc_article_url(url: str) -> bool:
    return any(pattern.match(url) for pattern in BBC_ARTICLE_PATTERNS)


def _infer_bbc_category(url: str) -> str:
    path = url.replace("https://www.bbc.com/", "")
    segment = path.split("/")[0]
    return BBC_CATEGORY_MAP.get(segment, "general")


class BBCScraper(BaseScraper):
    source = "bbc"
    category = "general"

    def __init__(self, links_file: Path | None = None) -> None:
        self.links_file = links_file or BBC_LINKS_FILE

    def scrape(self, limit: int = 50) -> ScrapeResult:
        result = ScrapeResult()
        urls = load_bbc_article_urls(self.links_file)[:limit]

        if not urls:
            result.errors.append(f"No BBC article URLs found in {self.links_file}")
            return result

        for url in urls:
            article, error = self._scrape_article(url)
            if error:
                result.errors.append(error)
                continue
            if article:
                result.articles.append(article)

        return result

    def _scrape_article(self, url: str) -> tuple[Article | None, str | None]:
        html, error = fetch_url_safe(url)
        if error or not html:
            return None, error

        soup = BeautifulSoup(html, "html.parser")
        title = _extract_title(soup)
        content = extract_main_content(soup)
        published_at = _extract_published_at(soup)
        summary = _extract_summary(soup)
        author = _extract_author(soup)

        if not title:
            return None, f"No title found for {url}"
        if len(content) < 120:
            return None, f"Insufficient content for {url}"

        return Article(
            title=title,
            content=content,
            summary=summary,
            category=_infer_bbc_category(url),
            author=author,
            published_at=published_at,
            source=self.source,
            url=url,
            language="en",
        ), None


def _extract_title(soup: BeautifulSoup) -> str | None:
    meta = soup.find("meta", property="og:title")
    if meta and meta.get("content"):
        return clean_text(meta["content"])

    heading = soup.find("h1")
    if heading:
        return clean_text(heading.get_text())

    return None


def _extract_summary(soup: BeautifulSoup) -> str | None:
    meta = soup.find("meta", property="og:description") or soup.find(
        "meta", attrs={"name": "description"}
    )
    if meta and meta.get("content"):
        return clean_text(meta["content"])
    return None


def _extract_author(soup: BeautifulSoup) -> str | None:
    for selector in (
        {"name": "author"},
        {"property": "article:author"},
    ):
        tag = soup.find("meta", attrs=selector)
        if tag and tag.get("content"):
            return clean_text(tag["content"])

    byline = soup.find(attrs={"data-testid": "byline-new"})
    if byline:
        return clean_text(byline.get_text())

    return None


def _extract_published_at(soup: BeautifulSoup) -> datetime:
    for selector in (
        {"property": "article:published_time"},
        {"property": "og:updated_time"},
        {"name": "pubdate"},
    ):
        tag = soup.find("meta", attrs=selector)
        if tag and tag.get("content"):
            parsed = _parse_datetime(tag["content"])
            if parsed:
                return parsed

    time_tag = soup.find("time", attrs={"datetime": True})
    if time_tag:
        parsed = _parse_datetime(time_tag["datetime"])
        if parsed:
            return parsed

    for script in soup.find_all("script", type="application/ld+json"):
        try:
            data = json.loads(script.string or "")
        except json.JSONDecodeError:
            continue
        items = data if isinstance(data, list) else [data]
        for item in items:
            if not isinstance(item, dict):
                continue
            for key in ("datePublished", "dateModified"):
                if key in item:
                    parsed = _parse_datetime(item[key])
                    if parsed:
                        return parsed

    return datetime.now(timezone.utc)


def _parse_datetime(value: str) -> datetime | None:
    if not value:
        return None

    normalized = value.strip()
    if normalized.endswith("Z"):
        normalized = normalized[:-1] + "+00:00"

    try:
        parsed = datetime.fromisoformat(normalized)
        if parsed.tzinfo is None:
            return parsed.replace(tzinfo=timezone.utc)
        return parsed.astimezone(timezone.utc)
    except ValueError:
        pass

    try:
        return parsedate_to_datetime(normalized).astimezone(timezone.utc)
    except (TypeError, ValueError, IndexError):
        return None
