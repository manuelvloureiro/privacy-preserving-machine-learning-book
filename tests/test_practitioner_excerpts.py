"""Every code block in the practitioner edition and the OpenDP book is a tested snippet, an excerpt of a tested script,
or output.

- A ```{.python excerpt="name"} block must equal, line for line, the region of a script of the book's kind
  (code/<part>/practitioner_<topic>.py for the practitioner edition, code/<part>/opendp_<topic>.py for the OpenDP
  book) between a `# excerpt: name` line and the next `# end excerpt` line, after removing the region's common
  indentation. The scripts are run and checked by their own tests, so an excerpt can never drift from code that has
  been run.
- A ```python block is a snippet, which tests/test_practitioner_snippets.py runs.
- A ```text block shows output or an unrun library call, and a ```bash block shows a command.
- No block runs much past 20 lines.
"""

import re
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BOOKS = {"book-practitioner": "practitioner_*.py", "book-opendp": "opendp_*.py"}  # book folder: its scripts
FENCE = re.compile(r"^```([^\n]*?)[ \t]*\n(.*?)^```[ \t]*$", re.DOTALL | re.MULTILINE)
EXCERPT_INFO = re.compile(r'^\{\.python\s+excerpt="([\w-]+)"\}$')
REGION = re.compile(r"^[ \t]*# excerpt: ([\w-]+)[ \t]*\n(.*?)^[ \t]*# end excerpt[ \t]*$", re.DOTALL | re.MULTILINE)
PLAIN = {"python", "text", "bash"}
MAX_LINES = 22


def scripts() -> list[Path]:
    return sorted(script for pattern in BOOKS.values() for script in (ROOT / "code").glob(f"*/{pattern}"))


def regions() -> dict[str, tuple[Path, str]]:
    found: dict[str, tuple[Path, str]] = {}
    for script in scripts():
        for m in REGION.finditer(script.read_text()):
            name = m.group(1)
            assert name not in found, f"excerpt {name!r} is defined in {found[name][0]} and {script}"
            found[name] = (script, textwrap.dedent(m.group(2)).rstrip("\n"))
    return found


def blocks() -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
    for book in BOOKS:
        folder = ROOT / book
        for page in sorted([*folder.rglob("*.md"), *folder.rglob("*.qmd")]):
            for i, m in enumerate(FENCE.finditer(page.read_text()), start=1):
                found.append((f"{book}/{page.relative_to(folder)}#{i}", m.group(1), m.group(2).rstrip("\n")))
    return found


@pytest.mark.parametrize(("name", "info", "body"), blocks(), ids=[name for name, _, _ in blocks()])
def test_block_is_snippet_excerpt_or_output(name: str, info: str, body: str) -> None:
    if m := EXCERPT_INFO.match(info):
        known = regions()
        assert m.group(1) in known, f"{name}: no script marks an excerpt named {m.group(1)!r}"
        script, region = known[m.group(1)]
        assert body == region, f"{name}: differs from excerpt {m.group(1)!r} in {script.relative_to(ROOT)}"
    else:
        assert info in PLAIN, f"{name}: a code block must be python, text, bash or a named excerpt, not {info!r}"
    lines = body.count("\n") + 1
    assert lines <= MAX_LINES, f"{name}: {lines} lines; keep a block to about 20 and move the rest to a script"


def test_every_excerpt_region_is_closed() -> None:
    for script in scripts():
        text = script.read_text()
        opened = len(re.findall(r"^[ \t]*# excerpt: ", text, re.MULTILINE))
        closed = len(re.findall(r"^[ \t]*# end excerpt[ \t]*$", text, re.MULTILINE))
        assert opened == closed == len(REGION.findall(text)), f"{script}: unbalanced excerpt markers"
