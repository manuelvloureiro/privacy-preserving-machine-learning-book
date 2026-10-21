"""Every Python snippet in the practitioner edition runs on its own and prints what its comments say.

A snippet is a ```python block. It must be self-contained (its imports included), and each line that prints ends
with the expected output as a comment, `print(f"{x:.4f}")  # 3.7306`. It runs from the repository's root. A snippet
that imports a library from the `practitioner` dependency group (torch, Opacus, OpenDP, ...) is marked `libs`.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
BOOK = ROOT / "book-practitioner"
BLOCK = re.compile(r"^```python\n(.*?)^```$", re.DOTALL | re.MULTILINE)
EXPECTED = re.compile(r"^\s*print\(.*\)\s*#\s*(.+?)\s*$")
LIBS = re.compile(
    r"^\s*(?:import|from)\s+(torch|torchvision|opacus|dp_accounting|opendp|diffprivlib|peft|transformers|sklearn|"
    r"scipy|polars)\b",
    re.MULTILINE,
)


def snippets() -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for page in sorted([*BOOK.rglob("*.md"), *BOOK.rglob("*.qmd")]):
        for i, m in enumerate(BLOCK.finditer(page.read_text()), start=1):
            found.append((f"{page.relative_to(BOOK)}#{i}", m.group(1)))
    return found


def cases() -> list[object]:
    return [
        pytest.param(name, code, id=name, marks=[pytest.mark.libs] if LIBS.search(code) else [])
        for name, code in snippets()
    ]


@pytest.mark.parametrize(("name", "code"), cases())
def test_snippet_prints_its_comments(name: str, code: str) -> None:
    expected = [m.group(1) for line in code.splitlines() if (m := EXPECTED.match(line))]
    assert expected, f"{name}: no print line with an expected-output comment"
    result = subprocess.run(
        [sys.executable, "-I", "-W", "ignore", "-c", code], capture_output=True, text=True, cwd=ROOT, timeout=120
    )
    assert result.returncode == 0, f"{name}:\n{result.stderr}"
    assert result.stdout.splitlines() == expected, name
