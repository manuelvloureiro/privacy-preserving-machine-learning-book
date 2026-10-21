"""code/part02-attacks/practitioner_table_risk.py runs and prints the numbers that the practitioner edition quotes.

The pages are book-practitioner/part02-attacks/10-risk-assessment.md (step 5), 07-linkage-reidentification.md and
08-syntactic-anonymity.md. The counts on Adult are exact, so every line is compared as printed.
"""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "code" / "part02-attacks" / "practitioner_table_risk.py"


@pytest.mark.libs
def test_practitioner_table_risk_matches_pages() -> None:
    out = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, check=True).stdout
    lines = {line.strip() for line in out.splitlines()}
    for expected in (
        "age + sex + race: unique=0.12%  in classes <5=0.75%  min k=1  J/n=0.012",
        "age + sex + race + native-country: unique=3.07%  in classes <5=7.83%  min k=1  J/n=0.060",
        "age + sex + race + native-country + marital-status: unique=5.88%  in classes <5=11.99%  min k=1  J/n=0.100",
        "age + sex + race + native-country + marital-status + education: unique=14.64%  in classes <5=26.83%  min k=1  J/n=0.227",
        "age + sex + race + native-country + marital-status + education + hours-per-week: unique=36.55%  in classes <5=57.05%  min k=1  J/n=0.480",
        "after generalizing: unique=0.94%  in a class with one occupation=1.20%",
        "age + sex + race: unique=0.01%  in classes <5=0.04%  min k=1  J/n=0.002",
        "age + sex + race + native-country: unique=0.02%  in classes <5=0.13%  min k=1  J/n=0.003",
        "age + sex + race + native-country + marital-status: unique=0.25%  in classes <5=0.98%  min k=1  J/n=0.012",
        "age + sex + race + native-country + marital-status + education: unique=0.94%  in classes <5=3.31%  min k=1  J/n=0.032",
        "age + sex + race + native-country + marital-status + education + hours-per-week: unique=10.09%  in classes <5=21.78%  min k=1  J/n=0.175",
        "records in classes with l=1: 1.20%",
        "records in classes of size >=5 with l=1: 0.03%",
        "min l=1  median l over classes=3",
    ):
        assert expected in lines, expected
