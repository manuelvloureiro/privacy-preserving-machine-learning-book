"""code/notes/privacy_budgets.py runs and reproduces the numbers in notes/privacy-budgets/index.md."""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def test_privacy_budgets_matches_note(tmp_path: Path) -> None:
    out = subprocess.run(
        [sys.executable, str(ROOT / "code" / "notes" / "privacy_budgets.py"), "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    for expected in (
        "ln(1/delta)=13.8155",
        "zCDP budget: rho=0.01747",
        "mu=0.01747 tau=0.18692",
        "P(L>1)=0.3085  exact delta(1)=0.1269",
        "pure basic: eps0=0.0100 b=100.00 sd=141.42",
        "Laplace zCDP: eps0=0.01869 rho0=1.747e-04 b=53.50 sd=75.66",
        "Laplace CDP: eps0=0.01869 b=53.50 sd=75.66",
        "Gaussian basic: eps0=0.01 delta0=1e-08 sigma=412.36",
        "Gaussian advanced: share=0.395 delta'=3.95e-07 eps0=0.01782 delta0=6.05e-09 sigma=243.80",
        "Gaussian zCDP: rho0=1.747e-04 sigma=53.50",
        "Gaussian CDP: mu0=1.747e-04 tau0=0.01869 sigma=53.50",
        "Gaussian exact: s=0.23670 sigma=42.25",
        "zCDP over exact: 1.266",
        "Gaussian exact < Laplace basic: k=9",
        "Gaussian zCDP < Laplace basic: k=15",
        "Laplace zCDP < Laplace basic: k=29",
        "Gaussian advanced < Gaussian basic: k=34",
        "Gaussian advanced < Laplace basic: k=313",
        "Laplace, pure DP: eps0=0.01000 delta0=0.00e+00 -> total eps=1.000 delta=0.00e+00",
        "Laplace, zCDP: eps0=0.01869 delta0=0.00e+00 -> total eps=1.869 delta=0.00e+00",
        "Gaussian, basic: eps0=0.01000 delta0=1.00e-08 -> total eps=1.000 delta=1.00e-06",
        "Gaussian, advanced: eps0=0.01782 delta0=6.05e-09 -> total eps=1.782 delta=6.05e-07",
        "Gaussian, CDP or zCDP: eps0=0.11363 delta0=1.00e-08 -> total eps=11.363 delta=1.00e-06",
        "Gaussian, exact: eps0=0.10915 delta0=1.00e-08 -> total eps=10.915 delta=1.00e-06",
        "1           1.41      7.57         4.22      21.81        5.35         4.22",
        "10         14.14     23.93        41.33      73.01       16.92        13.36",
        "100       141.42     75.66       412.36     243.80       53.50        42.25",
        "1000     1414.21    239.26      4122.63     810.72      169.18       133.60",
    ):
        assert expected in out, expected
    assert (tmp_path / "losses.svg").exists()
    assert (tmp_path / "noise_per_query.svg").exists()
