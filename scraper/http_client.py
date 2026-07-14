import time
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

DEFAULT_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; NewsRAGBot/1.0; +https://github.com/news-rag)"
    ),
    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
    "Accept-Language": "en-US,en;q=0.9",
}

REQUEST_DELAY_SECONDS = 1.0
_last_request_at = 0.0


def fetch_url(url: str, timeout: int = 30) -> str:
    global _last_request_at

    elapsed = time.monotonic() - _last_request_at
    if elapsed < REQUEST_DELAY_SECONDS:
        time.sleep(REQUEST_DELAY_SECONDS - elapsed)

    request = Request(url, headers=DEFAULT_HEADERS)
    with urlopen(request, timeout=timeout) as response:
        charset = response.headers.get_content_charset() or "utf-8"
        _last_request_at = time.monotonic()
        return response.read().decode(charset, errors="replace")


def fetch_url_safe(url: str, timeout: int = 30) -> tuple[str | None, str | None]:
    try:
        return fetch_url(url, timeout=timeout), None
    except HTTPError as exc:
        return None, f"HTTP {exc.code} for {url}"
    except URLError as exc:
        return None, f"Network error for {url}: {exc.reason}"
    except Exception as exc:
        return None, f"Failed to fetch {url}: {exc}"
