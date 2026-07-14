from scraper.base import BaseScraper
from scraper.bbc import BBCScraper
from scraper.rss import (
    APNewsScraper,
    HuggingFaceScraper,
    OpenAINewsScraper,
    PhysOrgScraper,
    ReutersScraper,
    ScienceDailyScraper,
    TechCrunchScraper,
    TNWScraper,
    VentureBeatScraper,
    VergeScraper,
)

SCRAPERS: dict[str, type[BaseScraper]] = {
    "bbc": BBCScraper,
    "techcrunch": TechCrunchScraper,
    "theverge": VergeScraper,
    "tnw": TNWScraper,
    "venturebeat": VentureBeatScraper,
    "sciencedaily": ScienceDailyScraper,
    "physorg": PhysOrgScraper,
    "openai": OpenAINewsScraper,
    "huggingface": HuggingFaceScraper,
    "reuters": ReutersScraper,
    "apnews": APNewsScraper,
}


def list_sources() -> list[str]:
    return sorted(SCRAPERS.keys())


def get_scraper(source: str) -> BaseScraper:
    key = source.lower().strip()
    if key not in SCRAPERS:
        available = ", ".join(list_sources())
        raise ValueError(f"Unknown source '{source}'. Available: {available}")
    return SCRAPERS[key]()
