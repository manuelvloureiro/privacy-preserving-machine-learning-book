"""Every code block in a study note is an excerpt of its tested script, or a command.

A ```{.python excerpt="name"} block in notes/<note>/index.md must equal, line for line, the region of a script in
code/notes/ between a `# excerpt: name` line and the next `# end excerpt` line, after removing the region's common
indentation, the same rule as tests/test_practitioner_excerpts.py applies to the practitioner edition. A ```bash
block shows a command and a ```text block shows output. A raw block (```{=latex}, ```{=html}) is page layout for one
output format, not code, and is not checked.
"""

import re
import textwrap
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
FENCE = re.compile(r"^```([^\n]*?)[ \t]*\n(.*?)^```[ \t]*$", re.DOTALL | re.MULTILINE)
EXCERPT_INFO = re.compile(r'^\{\.python\s+excerpt="([\w-]+)"\}$')
REGION = re.compile(r"^[ \t]*# excerpt: ([\w-]+)[ \t]*\n(.*?)^[ \t]*# end excerpt[ \t]*$", re.DOTALL | re.MULTILINE)
RAW = re.compile(r"^\{=\w+\}$")
MAX_LINES = 22


def regions() -> dict[str, tuple[Path, str]]:
    found: dict[str, tuple[Path, str]] = {}
    for script in sorted((ROOT / "code" / "notes").glob("*.py")):
        for m in REGION.finditer(script.read_text()):
            name = m.group(1)
            assert name not in found, f"excerpt {name!r} is defined in {found[name][0]} and {script}"
            found[name] = (script, textwrap.dedent(m.group(2)).rstrip("\n"))
    return found


def blocks() -> list[tuple[str, str, str]]:
    found: list[tuple[str, str, str]] = []
    for page in sorted((ROOT / "notes").glob("*/index.md")):
        for i, m in enumerate(FENCE.finditer(page.read_text()), start=1):
            found.append((f"{page.parent.name}#{i}", m.group(1), m.group(2).rstrip("\n")))
    return found


@pytest.mark.parametrize(("name", "info", "body"), blocks(), ids=[name for name, _, _ in blocks()])
def test_block_is_excerpt_or_command(name: str, info: str, body: str) -> None:
    if RAW.match(info):
        return
    if m := EXCERPT_INFO.match(info):
        known = regions()
        assert m.group(1) in known, f"{name}: no script in code/notes/ marks an excerpt named {m.group(1)!r}"
        script, region = known[m.group(1)]
        assert body == region, f"{name}: differs from excerpt {m.group(1)!r} in {script.relative_to(ROOT)}"
    else:
        assert info in {"bash", "text"}, f"{name}: a code block must be bash, text or a named excerpt, not {info!r}"
    lines = body.count("\n") + 1
    assert lines <= MAX_LINES, f"{name}: {lines} lines; keep a block to about 20 and leave the rest in the script"


def test_every_excerpt_region_is_closed() -> None:
    for script in (ROOT / "code" / "notes").glob("*.py"):
        text = script.read_text()
        opened = len(re.findall(r"^[ \t]*# excerpt: ", text, re.MULTILINE))
        closed = len(re.findall(r"^[ \t]*# end excerpt[ \t]*$", text, re.MULTILINE))
        assert opened == closed == len(REGION.findall(text)), f"{script}: unbalanced excerpt markers"
