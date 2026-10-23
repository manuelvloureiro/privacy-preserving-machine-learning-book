"""code/notes/concentrated_privacy.py runs and reproduces the numbers in notes/concentrated-privacy/index.md."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_concentrated_privacy_matches_note() -> None:
    out = subprocess.run(
        [sys.executable, str(ROOT / "code" / "notes" / "concentrated_privacy.py")],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for expected in (
        "exact=0.368  integrated=0.368  chord=0.462  loose=0.859",
        "P(loss=+1)=0.5  P(loss=-1)=0.184  between=0.052  mean=0.368",
        "ln(1/eta)=11.513",
        "basic=10.00",
        "advanced: sqrt term=4.80  mean term=0.53  total=5.32",
        "zCDP: rho total=0.50  eps=5.30",
        "rho per round=0.02  rho total=2.0",
        "zCDP eps=11.60  sqrt part=9.597",
        "mCDP eps=11.60",
        "exact eps=10.00",
        "basic: eps0=0.932  total=93.2",
        "advanced: eps0=0.938  delta0=8.5e-08  eta=1.5e-06  mean term=73.0  total=121.6",
    ):
        assert expected in out, expected
