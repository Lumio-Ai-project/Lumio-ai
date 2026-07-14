from datetime import datetime, timezone
from email.utils import parsedate_to_datetime
from xml.etree import ElementTree

from bs4 import BeautifulSoup

from preprocessing.cleaner import clean_html, clean_text, extract_main_content
from scraper.base import Article, BaseScraper, ScrapeResult
from scraper.http_client import fetch_url_safe


class RSSScraper(BaseScraper):
    """Generic RSS feed scraper — fetches each item URL for full article text."""

    feed_url: str
    source: str
    category: str

    def scrape(self, limit: int = 50) -> ScrapeResult:
        result = ScrapeResult()
        feed_xml, error = fetch_url_safe(self.feed_url)
        if error or not feed_xml:
            result.errors.append(error or f"Empty feed: {self.feed_url}")
            return result

        try:
            root = ElementTree.fromstring(feed_xml)
        except ElementTree.ParseError as exc:
            result.errors.append(f"Invalid RSS XML for {self.source}: {exc}")
            return result

        items = root.findall(".//item")[:limit]
        if not items:
            items = root.findall(".//{*}entry")[:limit]

        for item in items:
            article, item_error = self._parse_item(item)
            if item_error:
                result.errors.append(item_error)
                continue
            if article:
                result.articles.append(article)

        return result

    def _parse_item(self, item: ElementTree.Element) -> tuple[Article | None, str | None]:
        title = _first_text(item, "title")
        link = _first_text(item, "link")
        if not link:
            link_el = item.find("{*}link")
            if link_el is not None:
                link = link_el.attrib.get("href", "")

        guid = _first_text(item, "guid") or link
        pub_date_raw = (
            _first_text(item, "pubDate")
            or _first_text(item, "published")
            or _first_text(item, "updated")
        )
        author = _first_text(item, "author") or _first_text(item, "creator")
        summary = _first_text(item, "description") or _first_text(item, "summary")
        content_encoded = _first_text(item, "{http://purl.org/rss/1.0/modules/content/}encoded")

        if not title or not link:
            return None, f"Skipping {self.source} item without title or link"

        published_at = _parse_pub_date(pub_date_raw)
        body = ""

        if content_encoded:
            body = clean_html(content_encoded)
        elif summary:
            body = clean_html(summary)

        if len(body) < 200:
            fetched_body, fetch_error = self._fetch_article_body(link)
            if fetch_error:
                return None, fetch_error
            body = fetched_body

        if len(body) < 120:
            return None, f"Insufficient content for {link}"

        clean_summary = clean_text(summary) if summary else None
        if clean_summary and len(clean_summary) > len(body):
            clean_summary = clean_summary[:280]

        return Article(
            title=clean_text(title),
            content=body,
            summary=clean_summary,
            category=self.category,
            author=clean_text(author) if author else None,
            published_at=published_at,
            source=self.source,
            url=link or guid,
            language="en",
        ), None

    def _fetch_article_body(self, url: str) -> tuple[str, str | None]:
        html, error = fetch_url_safe(url)
        if error or not html:
            return "", error

        soup = BeautifulSoup(html, "html.parser")
        return extract_main_content(soup), None


def _first_text(item: ElementTree.Element, tag: str) -> str | None:
    node = item.find(tag)
    if node is None or not node.text:
        return None
    return node.text.strip()


def _parse_pub_date(value: str | None) -> datetime:
    if not value:
        return datetime.now(timezone.utc)

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
        return datetime.now(timezone.utc)


class TechCrunchScraper(RSSScraper):
    source = "techcrunch"
    category = "technology"
    feed_url = "https://techcrunch.com/feed/"


class VergeScraper(RSSScraper):
    source = "theverge"
    category = "technology"
    feed_url = "https://www.theverge.com/rss/index.xml"


class TNWScraper(RSSScraper):
    source = "tnw"
    category = "technology"
    feed_url = "https://thenextweb.com/feed/"


class VentureBeatScraper(RSSScraper):
    source = "venturebeat"
    category = "technology"
    feed_url = "https://venturebeat.com/feed/"


class ScienceDailyScraper(RSSScraper):
    source = "sciencedaily"
    category = "science"
    feed_url = "https://www.sciencedaily.com/rss/all.xml"


class PhysOrgScraper(RSSScraper):
    source = "physorg"
    category = "science"
    feed_url = "https://phys.org/rss-feed/"


class OpenAINewsScraper(RSSScraper):
    source = "openai"
    category = "ai"
    feed_url = "https://openai.com/news/rss.xml"


class HuggingFaceScraper(RSSScraper):
    source = "huggingface"
    category = "ai"
    feed_url = "https://huggingface.co/blog/feed.xml"


class ReutersScraper(RSSScraper):
    source = "reuters"
    category = "general"
    feed_url = "https://www.reutersagency.com/feed/?taxonomy=best-topics&post_type=best"


class APNewsScraper(RSSScraper):
    source = "apnews"
    category = "general"
    feed_url = "https://rsshub.app/apnews/topics/apf-topnews"
