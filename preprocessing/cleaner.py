import re
from html import unescape

from bs4 import BeautifulSoup, Comment, NavigableString

BOILERPLATE_TAGS = {
    "script",
    "style",
    "nav",
    "footer",
    "header",
    "aside",
    "noscript",
    "iframe",
    "form",
    "button",
    "svg",
}

BOILERPLATE_CLASS_PATTERNS = re.compile(
    r"(cookie|newsletter|subscribe|social|share|comment|advert|promo|related|sidebar|"
    r"navigation|breadcrumb|footer|header|menu|popup|modal|banner)",
    re.IGNORECASE,
)

WHITESPACE_RE = re.compile(r"\s+")


def clean_html(html: str) -> str:
    """Strip HTML tags and boilerplate, returning plain text."""
    if not html or not html.strip():
        return ""

    soup = BeautifulSoup(html, "html.parser")

    for tag in soup.find_all(BOILERPLATE_TAGS):
        tag.decompose()

    for comment in soup.find_all(string=lambda text: isinstance(text, Comment)):
        comment.extract()

    for element in soup.find_all(True):
        if element.attrs is None:
            continue
        classes = " ".join(element.get("class", []))
        element_id = element.get("id", "")
        if BOILERPLATE_CLASS_PATTERNS.search(f"{classes} {element_id}"):
            element.decompose()

    text = soup.get_text(separator="\n")
    return clean_text(text)


def clean_text(text: str) -> str:
    """Normalize whitespace and remove common noise from scraped text."""
    if not text:
        return ""

    text = unescape(text)
    text = text.replace("\xa0", " ")
    text = text.replace("\u200b", "")

    lines: list[str] = []
    for line in text.splitlines():
        line = WHITESPACE_RE.sub(" ", line).strip()
        if not line:
            continue
        if _is_noise_line(line):
            continue
        lines.append(line)

    return "\n\n".join(lines)


def _is_noise_line(line: str) -> bool:
    lower = line.lower()
    if len(line) < 3:
        return True
    if lower in {"share", "comments", "more on this story", "related topics"}:
        return True
    if lower.startswith("image source,") or lower.startswith("video "):
        return True
    if line.isupper() and len(line) < 40:
        return True
    return False


def extract_main_content(soup: BeautifulSoup) -> str:
    """Try article-specific selectors before falling back to full body."""
    selectors = [
        ("article", {}),
        ("div", {"data-component": "text-block"}),
        ("div", {"class": "article-body"}),
        ("div", {"class": "entry-content"}),
        ("div", {"class": "post-content"}),
        ("main", {}),
    ]

    for tag, attrs in selectors:
        nodes = soup.find_all(tag, attrs=attrs) if attrs else soup.find_all(tag)
        if nodes:
            container = soup.new_tag("div")
            for node in nodes:
                container.append(node.__copy__())
            return clean_html(str(container))

    return clean_html(str(soup.body or soup))
