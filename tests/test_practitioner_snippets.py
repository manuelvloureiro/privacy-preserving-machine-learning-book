"""Every Python snippet in the practitioner edition runs on its own and prints what its comments say.

A snippet is a ```python block. It must be self-contained (standard library only), and each line
that prints ends with the expected output as a comment, `print(f"{x:.4f}")  # 3.7306`.
"""

import re
import subprocess
import sys
from pathlib import Path

import pytest

BOOK = Path(__file__).resolve().parent.parent / "book-practitioner"
BLOCK = re.compile(r"^```python\n(.*?)^```$", re.DOTALL | re.MULTILINE)
EXPECTED = re.compile(r"^\s*print\(.*\)\s*#\s*(.+?)\s*$")


def snippets() -> list[tuple[str, str]]:
    found: list[tuple[str, str]] = []
    for page in sorted([*BOOK.rglob("*.md"), *BOOK.rglob("*.qmd")]):
        for i, m in enumerate(BLOCK.finditer(page.read_text()), start=1):
            found.append((f"{page.relative_to(BOOK)}#{i}", m.group(1)))
    return found


@pytest.mark.parametrize(("name", "code"), snippets(), ids=[name for name, _ in snippets()])
def test_snippet_prints_its_comments(name: str, code: str) -> None:
    expected = [m.group(1) for line in code.splitlines() if (m := EXPECTED.match(line))]
    assert expected, f"{name}: no print line with an expected-output comment"
    result = subprocess.run([sys.executable, "-I", "-c", code], capture_output=True, text=True, check=True)
    assert result.stdout.splitlines() == expected, name
