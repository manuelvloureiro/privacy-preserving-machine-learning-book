"""code/part02-attacks/practitioner_attribute.py runs and prints the numbers that the practitioner edition quotes.

The page is book-practitioner/part02-attacks/03-attribute-inference.md. The target and ART's attack models are
seeded random forests on one thread, so every printed digit is reproducible.
"""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "code" / "part02-attacks" / "practitioner_attribute.py"


@pytest.mark.libs
def test_practitioner_attribute_matches_page() -> None:
    out = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, check=True).stdout
    lines = {line.strip() for line in out.splitlines()}
    for expected in (
        "[Data] Adult without missing values: 45,222 people, 86 inputs; 4,000 in each set",
        "share female: members 0.3180  non-members 0.3330",
        "target: train accuracy 0.9915  test accuracy 0.8353",
        "ART black-box attack: right on members 0.8300 [0.8180, 0.8415]  on non-members 0.8423 [0.8306, 0.8534]",
        "ART black-box attack: advantage -0.0123 [-0.0285, +0.0040]",
        "imputation baseline: right on members 0.8307 [0.8188, 0.8423]  on non-members 0.8417 [0.8301, 0.8529]",
        "imputation baseline: advantage -0.0110 [-0.0272, +0.0052]",
    ):
        assert expected in lines, expected
