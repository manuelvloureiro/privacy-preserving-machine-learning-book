"""What every test of a script shares: running it as the book does and reading what it prints."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code"


def run(script: Path, out: Path) -> str:
    """Run a chapter, solutions, appendix, course or note script with `--out`, and return its stdout.

    The script writes its figures into `out`; the test then checks the numbers the page quotes against
    the printed lines, and that the figures the page shows exist.
    """
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout
