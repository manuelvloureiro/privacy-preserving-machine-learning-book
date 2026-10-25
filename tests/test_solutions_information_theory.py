"""code/part01-prerequisites/solutions_information_theory.py runs and reproduces the numbers in book-exercises/part01-prerequisites/05-information-theory.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part01-prerequisites"


def test_solutions_information_theory_match_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_information_theory.py", tmp_path)
    for expected in (
        "[geometric, k<=200] H=2.0000  H_min=1.0000",
        "[geometric, k<=5] partial sum of k 2^-k=1.7812",
        "[geometric, k<=10] partial sum of k 2^-k=1.9883",
        "[geometric, k<=20] partial sum of k 2^-k=2.0000",
        "[xor] I(X;Y1)=0.0000  I(X;Y2)=0.0000  I(X;Y1,Y2)=1.0000",
        "[two BSC(0.1) copies] I(X;Y1)=0.5310  I(X;Y2)=0.5310  I(X;Y1,Y2)=0.7421  sum=1.0620",
        "[BSC p=0.11] 1-h(p)=0.5001  log2(2max)=0.8319",
        "[BSC p=0.25] 1-h(p)=0.1887  log2(2max)=0.5850",
        "[BSC grid of 1001 p] min gap=0.0000",
        "[Z] closed form vs joint table, max gap=3.1e-16",
        "[Z] grid argmax q=0.4000  I(0.4)=0.3219  log2(5/4)=0.3219  I(0.5)=0.3113",
        "[Z] Blahut-Arimoto=0.3219  optimal Pr[X=1]=0.4000",
        "h(eps)=0.8813  eps*log2(M-1)=0.8422  capacity=1.2765  Blahut-Arimoto=1.2765  max |r - 1/M|=0.0e+00",
        "eps/(M-1)=0.0429  min-capacity=2.4854  from column maxima=2.4854",
        "[walk-through] Smith=0.5995  average=0.6548",
        "Smith=0.9556  average=2.5000  closed forms: 0.9556, 2.5000",
        "[reveal-if-zero(1), n=64] Smith=1.0000  average=31.5000",
        "[M=8 eps=0.1] MAP error=0.1000  H(X|Y)=0.7497  h(eps)+eps*log2(7)=0.7497  Fano bound=0.1000",
        "[M=8 eps=0.2] MAP error=0.2000  H(X|Y)=1.2834  h(eps)+eps*log2(7)=1.2834  Fano bound=0.2000",
        "[M=8 eps=0.5] MAP error=0.5000  H(X|Y)=2.4037  h(eps)+eps*log2(7)=2.4037  Fano bound=0.5000",
        "[M=8 eps=0.875] MAP error=0.8750  H(X|Y)=3.0000  h(eps)+eps*log2(7)=3.0000  Fano bound=0.8750",
        "[M=1024 kappa=0.01 target 0.1] n >= 800.0",
        "[M=1024 kappa=0.01 n=799] P_e >= 0.1010",
        "[M=1024 kappa=0.01 n=800] P_e >= 0.1000",
    ):
        assert expected in out, expected
