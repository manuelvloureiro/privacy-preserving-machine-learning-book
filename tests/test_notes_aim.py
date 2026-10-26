"""code/notes/aim.py runs and reproduces the numbers in notes/aim/index.md."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_aim_matches_note(tmp_path: Path) -> None:
    out = subprocess.run(
        [sys.executable, str(ROOT / "code" / "notes" / "aim.py"), "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for expected in (
        "n=1000  cells=24  attributes=4",
        "smoker x condition: [[627, 98], [181, 94]]",
        "workload marginals=6  candidates=10",
        "weights: one-way=3  two-way=6  sensitivity=6",
        "sqrt(2/pi)=0.798",
        "rho=0.0244 for eps=1.0 delta=1e-06",
        "T=64  sigma0=38.2  eps1=0.0174  init cost=0.0014  round cost=0.0004",
        "total estimated from the noisy one-way marginals: 1018.7",
        "init workload error=0.1412",
        "age × exercise",
        "penalty=  182.9  score=   396.3",
        "rounds=7  annealings=3",
        "final model workload error=0.0529  synthetic dataset workload error=0.0537  synthetic records=1019",
        "measure everything: sigma=11.1  workload error=0.0350",
        "all 3-way marginals of 15 attributes: 455  sigma if all measured=96.6",
        "final model smoker x condition: [[626.3, 103.0], [185.4, 104.0]]",
    ):
        assert expected in out, expected
    for name in ("_marginal.md", "_init.md", "_scores.md", "_schedule.md", "_rounds.md", "scores.svg", "rounds.svg"):
        assert (tmp_path / name).exists(), name
