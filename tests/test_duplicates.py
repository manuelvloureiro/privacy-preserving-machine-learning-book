"""tools/duplicates.py finds a repeated paragraph, and the books have none it does not allow."""

import subprocess
import sys
import tomllib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "tools"))
import duplicates

PARAGRAPH = (
    "The script runs every query on the sample and prints the scales, the error bars and the true values. "
    "These are the same on every run, but the released values are not, because the noise cannot be seeded, "
    "so the page shows them in one block marked as one run of the script."
)
OTHER = (
    "A retail bank wants to reach customers who earn more than a threshold, and it trains a model on the "
    "records it holds. The privacy setting decides how much noise each training step adds, and the business "
    "goal decides how many of the right customers the campaign must reach to pay for itself."
)


def test_a_copied_paragraph_is_a_repeat_and_a_different_one_is_not() -> None:
    config = tomllib.loads(duplicates.CONFIG.read_text())
    paras = duplicates.paragraphs("a.md", f"# One\n\n{PARAGRAPH}\n\n{OTHER}\n", config["shingle"], config["min_words"])
    paras += duplicates.paragraphs("b.md", f"# Two\n\n{PARAGRAPH}\n", config["shingle"], config["min_words"])
    found = duplicates.repeats(config, paras)
    assert [(a.page, b.page) for _, a, b in found] == [("a.md", "b.md")]


def test_the_books_have_no_repeat_that_is_not_allowed() -> None:
    result = subprocess.run(
        [sys.executable, "tools/duplicates.py", "check"],
        capture_output=True,
        text=True,
        cwd=duplicates.ROOT,
        check=False,
    )
    assert result.returncode == 0, result.stdout[-2000:]
