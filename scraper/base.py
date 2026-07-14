from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Article:
    title: str
    content: str
    category: str
    source: str
    url: str
    published_at: datetime
    language: str = "en"
    summary: Optional[str] = None
    author: Optional[str] = None

    def to_dict(self) -> dict:
        return {
            "title": self.title,
            "content": self.content,
            "summary": self.summary,
            "category": self.category,
            "author": self.author,
            "publishedAt": self.published_at,
            "source": self.source,
            "url": self.url,
            "language": self.language,
        }


@dataclass
class ScrapeResult:
    articles: list[Article] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


class BaseScraper(ABC):
    source: str
    category: str

    @abstractmethod
    def scrape(self, limit: int = 50) -> ScrapeResult:
        """Fetch up to `limit` articles from this source."""
