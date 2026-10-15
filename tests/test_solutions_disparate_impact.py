"""code/part05-dp-ml/solutions_disparate_impact.py runs and reproduces the numbers in book-exercises/part05-dp-ml/08-disparate-impact.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part05-dp-ml"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_solutions_disparate_impact_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_disparate_impact.py", tmp_path)
    for expected in (
        "upweight 1: minority share of the update 0.1163",
        "upweight 4: minority share of the update 0.1163",
        "upweight 20: minority share of the update 0.1163",
        "no clipping: minority share 0.5128",
        "effective share with weights 1 and 0.125: 0.0065",
        "share 0.95: a=0.4750  variance per coordinate 1.596e-05  excess loss (51 coords) 4.069e-04",
        "formula 4.069e-04  small-a approximation 3.103e-04",
        "share 0.05: a=0.0250  variance per coordinate 2.341e-04  excess loss (51 coords) 5.970e-03",
        "formula 5.970e-03  small-a approximation 5.896e-03",
        "share 0.2: break-even steps relative to the whole population 25",
        "share 0.05: break-even steps relative to the whole population 400",
        "share 0.01: break-even steps relative to the whole population 10000",
        "70 of 100: 95% CI [0.600, 0.788]",
        "majority 0.95 on 9500, minority 0.9: records needed 140",
        "majority 0.95 on 9500, minority 0.85: records needed 50",
        "oversample x2: eps of minority records at sigma=1.36 8.92  sigma for eps=4 2.44",
        "oversample x4: eps of minority records at sigma=1.36 20.96  sigma for eps=4 4.71",
        "oversample x8: eps of minority records at sigma=1.36 53.04  sigma for eps=4 9.31",
        "largest difference between the two updates 1.7e-18",
    ):
        assert expected in out, expected
