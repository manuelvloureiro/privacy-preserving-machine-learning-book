"""code/part04-privacy-accounting/rdp_conversions.py runs, writes its figures, and reproduces the numbers in book-full/part04-privacy-accounting/02-rdp-conversions.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part04-privacy-accounting"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_rdp_conversions_matches_section(tmp_path: Path) -> None:
    out = run(CODE / "rdp_conversions.py", tmp_path)
    for expected in (
        # Toolkit
        "eps=1: delta exact=1.2694e-01  numeric E_P[(1-e^(eps-L))+]=1.2694e-01  tail P[L>eps]=3.0854e-01",
        "eps=3: delta exact=1.5372e-03  numeric E_P[(1-e^(eps-L))+]=1.5372e-03  tail P[L>eps]=6.2097e-03",
        # Section 28.3
        "mu=1: eps exact=4.377  tail step=4.765  Mironov=5.299 at alpha=5.8  (continuous: 5.299 at alpha=5.799)",
        "loss from dropping (1-e^(eps-L)): 0.388   loss from Markov: 0.534  total excess: 0.921",
        "mu=10: eps exact=91.817  tail step=92.649  Mironov=98.026 at alpha=1.5  (continuous: 97.985 at alpha=1.480)",
        "loss from dropping (1-e^(eps-L)): 0.832   loss from Markov: 5.377  total excess: 6.209",
        "mu=1 at eps=5.299: delta exact=1.300e-07  tail=7.992e-07  Mironov bound=1.000e-05  "
        "factors tail/exact=6.1  bound/tail=12.5",
        # Section 28.4
        "alpha=2: delta ratio improved/Mironov=0.2500  1/(e alpha)=0.1839  eps saving=1.3863",
        "alpha=10: delta ratio improved/Mironov=0.0387  1/(e alpha)=0.0368  eps saving=0.3612",
        "alpha=100: delta ratio improved/Mironov=0.0037  1/(e alpha)=0.0037  eps saving=0.0566",
        "best constant alpha=6 eps=4: formula=1.380552e-10  grid=1.380552e-10  at z*=4.1823",
        "mu=1: improved=4.729 at alpha=5.4  Mironov=5.299 at alpha=5.8  exact=4.377  improved removes 0.570 of the excess 0.921",
        "mu=10: improved=96.116 at alpha=1.5  Mironov=98.026 at alpha=1.5  exact=91.817",
        "mu=1: saving=0.570  Delta(5.8)=0.555",
        "(eps, delta) line from the improved conversion at a=1e-3: e^eps a + delta=0.1131",
        "check: delta exact at eps_improved=2.097e-06  delta improved at eps_improved=1.000e-05",
        "single order alpha=2, eps(alpha)=1: Mironov=12.513  improved=11.127  optimal for this order=10.668",
        "single order alpha=8, eps(alpha)=4: Mironov=5.645  improved=5.214  optimal for this order=5.214",
        "single order alpha=32, eps(alpha)=16: Mironov=16.371  improved=16.228  optimal for this order=16.228",
        # Section 28.5
        "a=0.1: true power 1-G_mu=3.8914e-01  largest power allowed by RDP=4.8376e-01  ratio=1.24",
        "a=0.001: true power 1-G_mu=1.8298e-02  largest power allowed by RDP=2.4910e-02  ratio=1.36",
        "a=1e-06: true power 1-G_mu=8.7218e-05  largest power allowed by RDP=1.1636e-04  ratio=1.33",
        "only order 2: largest power at a=0.1: 4.9325e-01",
        "only order 8: largest power at a=0.1: 9.8376e-01",
        "only order 2: largest power at a=0.001: 4.2431e-02",
        "only order 8: largest power at a=0.001: 7.8529e-02",
        "only order 2: largest power at a=1e-06: 1.3118e-03",
        "only order 8: largest power at a=1e-06: 1.8622e-04",
        "orders 2 and 8: bound lines cross at a=e^-8=3.35e-04  exact single-order boundaries cross at a=1.87e-04",
        "binding single order at a=1e-3: 3.7, alone allows 2.4910e-02",
        # Section 28.6
        "sigma=1 k=1: rho=0.5  improved alpha*=5.432 (grid 5.4)  eps continuous=4.728  grid=4.729  Mironov alpha*=5.799",
        "sigma=1 k=100: rho=50  improved alpha*=1.472 (grid 1.5)  eps continuous=96.035  grid=96.116  Mironov alpha*=1.480  "
        "grid cost=0.081",
        "sigma=1 k=10000: rho=5000  improved alpha*=1.048 (grid 1.1)  eps continuous=5475.791  grid=5611.778  "
        "Mironov alpha*=1.048  grid cost=135.987",
        "sigma=10 k=1: rho=0.005  improved alpha*=40.525 (grid 41)  eps continuous=0.375  grid=0.375  Mironov alpha*=48.985",
        "mu=1 delta=1e-10: improved alpha*=7.483",
        # Section 28.7
        "k=1: mu=0.100  exact=0.341  optimal=0.375  improved=0.375  Mironov=0.485  ratios 1.10 1.10 1.42",
        "k=10: mu=0.316  exact=1.199  optimal=1.308  improved=1.308  Mironov=1.568  ratios 1.09 1.09 1.31",
        "k=100: mu=1.000  exact=4.377  optimal=4.729  improved=4.729  Mironov=5.299  ratios 1.08 1.08 1.21",
        "k=1000: mu=3.162  exact=17.857  optimal=19.054  improved=19.054  Mironov=20.175  ratios 1.07 1.07 1.13",
        "k=10000: mu=10.000  exact=91.817  optimal=96.116  improved=96.116  Mironov=98.026  ratios 1.05 1.05 1.07",
        "privacy loss: accept L=4.5954  reject L=-2.257e-04  u - e^eps b=4.4700e-05",
        "max over orders of D_alpha - eps(alpha): forward=2.4e-13  reverse=-5.5e-01",
        "worst two-outcome pair at eps=4.377: P(accept)=2.2797e-04  Q(accept)=2.3020e-06  delta=4.4700e-05 "
        "vs Gaussian 1.0000e-05",
        "binding order=5.1  next smallest slack=0.0011 at order 5",
        "improved conversion's delta at that eps=4.4701e-05",
        "mu=1 to six decimals: optimal from the curve=4.728505  improved=4.728507",
        "pair inside the curve at every real order, eps=4.37718: P(accept)=2.2779e-04  Q(accept)=2.2998e-06  delta=4.4698e-05",
        "privacy loss: accept L=4.5956  reject L=-2.255e-04",
        "max over real alpha>1 of D_alpha - alpha/2, forward: -9.97e-06 at alpha=5.0955",
        "max over real alpha>1 of D_alpha - alpha/2, reverse: -5.00e-01",
        "max over real alpha>1 of D_alpha - alpha/2, forward, printed digits: -1.93e-05",
        "continuous curve at delta=1e-5: worst valid pair forces eps=4.728376",
        "improved at the best real order 5.4318: 4.728387",
        "release-with-prob-delta mechanism: D_2=inf",
        "RR eps=0.5, alpha=2: D_alpha=0.2273  bound min(eps, alpha eps^2/2)=0.2500",
        "RR eps=0.5, alpha=4: D_alpha=0.3519  bound min(eps, alpha eps^2/2)=0.5000",
        "RR eps=0.5, alpha=10: D_alpha=0.4473  bound min(eps, alpha eps^2/2)=0.5000",
    ):
        assert expected in out, expected
    assert {f.name for f in tmp_path.iterdir()} == {
        "rdp_conversions_majorants.svg",
        "rdp_conversions_tradeoff.svg",
        "rdp_conversions_orders.svg",
        "rdp_conversions_slack.svg",
    }
