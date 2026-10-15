"""code/part04-privacy-accounting/solutions_renyi_dp.py runs and reproduces the numbers in book-exercises/part04-privacy-accounting/01-renyi-dp.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part04-privacy-accounting"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_solutions_renyi_dp_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_renyi_dp.py", tmp_path)
    for expected in (
        # Exercise 1
        "p=0.75: chi2=1.3333  ln(1 + chi2)=0.8473  closed form=0.8473  eps(2)=0.8473",
        "p=0.6: chi2=0.1667  ln(1 + chi2)=0.1542  closed form=0.1542  eps(2)=0.1542",
        "p=0.55: eps(2)=0.03961  16 gamma^2 with gamma = p - 1/2: 0.04000  KL=0.02007  8 gamma^2=0.02000",
        # Exercise 2
        "basic=10.0000  advanced (textbook)=5.8502  advanced (tanh)=5.2981  RDP=5.0705 at alpha=6.3",
        "single step: eps(alpha=6.3)=0.02898  eps0^2/2=0.00500  k eps(alpha)=2.8983  ln(1/delta)/(alpha-1)=2.1723",
        "k=10: basic=1.0000  advanced (tanh)=1.5674  RDP=1.0090 at alpha=512",
        "k=1000: basic=100.0000  advanced (tanh)=20.1701  RDP=19.6905 at alpha=2.6",
        # Exercise 3
        "Mironov eps at delta=1e-5: 5.2985 (alpha=5.8)  tail bound at that eps: delta=1.000e-05 at lambda=4.8",
        "eps=3: delta=4.394e-02 at lambda=2.5  (continuous: lambda*=eps-1/2=2.5, delta=exp(-(eps-1/2)^2/2)=4.394e-02)",
        "eps=4: delta=2.187e-03 at lambda=3.5",
        # Exercise 4
        "alpha=3, u=2, v=2: D_alpha(P‖Q)=3.3750  orders 6 and 5: rhs=3.4375",
        "alpha=3, u=1.5, v=3: D_alpha(P‖Q)=3.3750  orders 4.5 and 7: rhs=4.1562",
        "alpha=3, u=3, v=1.5: D_alpha(P‖Q)=3.3750  orders 9 and 4: rhs=3.5000",
        "best u = 1 + b(alpha-1)/(alpha a) = 2.3333  v=1.7500  orders 7 and 4.5: rhs=3.3750",
        # Exercise 5
        "alpha* exceeds 512 once sigma > 106.5",
        "sigma=100: grid eps=0.04813 at alpha=512  continuous eps=0.04804 at alpha*=480.9  grid/continuous=1.002",
        "sigma=200: grid eps=0.02893 at alpha=512  continuous eps=0.02401 at alpha*=960.7  grid/continuous=1.205  "
        "exact=0.01251",
    ):
        assert expected in out, expected
    assert list(tmp_path.iterdir()) == []
