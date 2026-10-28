"""The landing page, site/index.html, is well formed and links only to what the build publishes.

No render reads the landing page, so a broken tag there blanks the published site with every check
passing: a lost </style> once made the browser read the whole page as CSS.
"""

from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
PAGE = ROOT / "site" / "index.html"
VOID = {"meta", "link", "br", "img", "hr", "input"}
BOOKS = ("book-full", "book-practitioner", "book-course", "book-opendp")


class Page(HTMLParser):
    """Track open tags, and collect each link and the text outside head."""

    def __init__(self) -> None:
        super().__init__()
        self.open: list[str] = []
        self.errors: list[str] = []
        self.links: list[str] = []
        self.body_tags: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if "body" in self.open:
            self.body_tags.append(tag)
        if tag == "a":
            self.links += [value for name, value in attrs if name == "href" and value]
        if tag not in VOID:
            self.open.append(tag)

    def handle_endtag(self, tag: str) -> None:
        if not self.open or self.open[-1] != tag:
            self.errors.append(f"</{tag}> closes {self.open[-1:] or 'nothing'}")
            return
        self.open.pop()


def parse() -> Page:
    page = Page()
    page.feed(PAGE.read_text())
    page.close()
    return page


def test_tags_balance() -> None:
    page = parse()
    assert page.errors == []
    assert page.open == []


def test_body_has_content() -> None:
    page = parse()
    assert {"main", "h1", "section", "a"} <= set(page.body_tags)


def test_links_point_to_published_paths() -> None:
    for href in parse().links:
        if href.startswith(("http://", "https://", "mailto:")):
            continue
        top = href.split("/")[0]
        if top in BOOKS:
            assert (ROOT / top / "_quarto.yml").exists(), href
        elif top == "notes":
            assert (ROOT / "notes" / href.split("/")[1] / "_quarto.yml").exists(), href
        else:
            assert href.endswith(".pdf"), href
