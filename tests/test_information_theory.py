"""code/information_theory.py runs, writes its figures, and reproduces the numbers in 05-information-theory.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "book" / "part01-prerequisites" / "code"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_information_theory_matches_section(tmp_path: Path) -> None:
    out = run(CODE / "information_theory.py", tmp_path)
    for expected in (
        "h(0.11)=0.4999",
        "[spam] H(X)=0.5436  H(Y)=0.8113  H(X,Y)=1.0613  H(X|Y=0)=0.0000  H(X|Y=1)=1.0000  H(X|Y)=0.2500",
        "[PIN] uniform: H=13.2877  popular 0.1: H=12.4278  H_min=3.3219",
        "[spam] I=H(X)-H(X|Y)=0.2936  I=KL=0.2936  I via Y,X swapped=0.2936",
        "I(X;Z)=0.1668",
        "[BSC p=0.11] 1-h(p)=0.5001  Blahut-Arimoto=0.5001  optimal Pr[X=1]=0.5000  min-capacity=0.8319",
        "[BSC p=0.25] 1-h(p)=0.1887  Blahut-Arimoto=0.1887  optimal Pr[X=1]=0.5000  min-capacity=0.5850",
        "[BSC p=0.11, prior q=0.2] I=0.3358",
        "[Z-channel p=0.5] Blahut-Arimoto=0.3219  log2(5/4)=0.3219  optimal Pr[X=1]=0.4000",
        "[symmetric M=8 eps=0.3] formula=1.2765  Blahut-Arimoto=1.2765  min-capacity=2.4854",
        "(0.32, 0.08, 0.00)  (0.03, 0.24, 0.03)  (0.00, 0.10, 0.10)  (0.00, 0.02, 0.08)",
        "p(y)=0.35, 0.44, 0.21  MAP guesses=[0, 1, 2]",
        "V(X)=0.4000  H_min(X)=1.3219  V(X|Y)=0.6600  H_min(X|Y)=0.5995  averaged H_min(X|Y=y)=0.6548",
        "simulated MAP success over 1,000,000 draws=0.6594",
        "min-entropy leakage=0.7225  Shannon leakage=0.6865  H(X)=1.8464",
        "uniform prior: min-entropy leakage=1.2630  min-capacity=1.2630  Shannon capacity=0.8269",
        "max leakage over a grid of priors=1.2630",
        "[reveal if X=0, n=4] Shannon=0.3373  min-entropy=1.0000",
        "[reveal if X=0, n=32] Shannon=7.786e-09  min-entropy=1.0000",
        "reveal-if-low-3-zero: Shannon=8.1686  min-entropy=61.0000  one-guess success=0.1250",
        "reveal-low-9: Shannon=9.0000  min-entropy=9.0000  one-guess success=2.78e-17",
        "max gap=7.8e-16",
        "[symmetric M=8 eps=0.2] H(X|Y)=1.2834  MAP error=0.2000  Fano=0.2000  weak Fano=0.0945",
        "[shift M=8 eps=0.2] H(X|Y)=0.7219  MAP error=0.2000  Fano=0.0954  weak Fano=0.0000",
        "[8 classes, I<=1 bit] Fano=0.3730  weak Fano=0.3333",
        "n=100] P_e >= 0.8000",
        "n=400] P_e >= 0.5000",
    ):
        assert expected in out
    assert {f.name for f in tmp_path.iterdir()} == {
        "information_theory_binary_entropy.svg",
        "information_theory_conditioning.svg",
        "information_theory_bsc.svg",
        "information_theory_guessing.svg",
        "information_theory_leakage.svg",
        "information_theory_fano.svg",
    }
