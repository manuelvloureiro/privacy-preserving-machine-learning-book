"""code/appendix-derivations/apx_amplification.py runs, writes its figures, and reproduces the numbers in book-full/appendix-derivations/12-amplification.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "appendix-derivations"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_amplification_matches_appendix(tmp_path: Path) -> None:
    out = run(CODE / "apx_amplification.py", tmp_path)
    for expected in (
        # The remove direction dominates.
        "P0=(0.7311, 0.2689)  Pq=(0.6848, 0.3152)  b=0.3881  ln A'=0.010803  ln B'=0.009846",
        "centre=0.4440  narrow average=0.416623  wide average=0.413773  (2a-1)d(narrow-wide)=0.000957",
        "points=5294646  violations=0  smallest gap=-2.2e-16",
        "alpha=2] remove=1.7181e-04 add=1.6022e-04  by pairs: remove=1.7181e-04 add=1.6022e-04",
        "alpha=8] remove=8.9364e-04 add=5.7239e-04  by pairs: remove=8.9364e-04 add=5.7239e-04",
        "alpha=32] remove=1.1246e+01 add=1.6923e-03  by pairs: remove=1.1246e+01 add=1.6923e-03",
        # Dominating pairs for Poisson subsampling.
        "[poisson eps=0] remove direct=3.829249e-03  identity=3.829249e-03",
        "[poisson eps=0.5] remove direct=2.214519e-07  identity=2.214519e-07",
        "[poisson eps=1] remove direct=2.732009e-09  identity=2.732009e-09",
        "largest relative gap remove=7.4e-14  add=3.0e-12  add delta vanishes from eps=0.0101",
        "eps'=0.017037  gamma'=2.718282  delta(eps')=1.269367e-03  q*delta_base(1)=1.269367e-03",
        "largest ratio real/pair=0.9687  largest excess real-pair=0.0e+00",
        # The clone reduction.
        "clone=[0.1042 0.1042 0.0417]  leftover=[0.0625 0.0625 0.625 ]  LO=[0.0833 0.0833 0.8333]",
        "most negative=0.0e+00  mixture error=1.1e-16  mean error=1.1e-16",
        "[clones exact reduction n=4] largest gap=8.3e-17",
        "[clones n=1000 eps0=1] numeric=0.1824  this appendix=0.3111  simplified=0.3767  ln(1+4x)=0.5565  FMT=0.5662",
        "[clones n=1000 eps0=2] numeric=0.5455  this appendix=1.0601",
        "c_min=72.6  mu=135.2",
        "[clones n=10000 eps0=1] numeric=0.0530  this appendix=0.0863  simplified=0.1167  ln(1+4x)=0.2096  FMT=0.2140",
        "[clones n=10000 eps0=2] numeric=0.1550  this appendix=0.2437  simplified=0.3204  ln(1+4x)=0.4919  FMT=0.5009",
        "[clones n=10000 eps0=4] numeric=0.6009  this appendix=1.0955",
        "[clones n=100000 eps0=2] numeric=0.0450  this appendix=0.0724  simplified=0.1001  ln(1+4x)=0.1823  FMT=0.1862",
        "[clones n=100000 eps0=4] numeric=0.1698  this appendix=0.2621  simplified=0.3488  ln(1+4x)=0.5251  FMT=0.5346",
        "delta at this appendix's eps=3.84e-11",
        "ratios to numeric: this appendix=1.54  FMT=3.15",
        "ratios to numeric: this appendix=1.94  FMT=nan",
        "ratios to numeric: this appendix=1.71  FMT=3.10",
        "ratios to numeric: this appendix=1.61  FMT=4.14",
        # Shift reduction.
        "[shift s=1 sigma=1 a=0.25] exact D2=0.0160  numeric=0.0160  bound=0.0625",
        "[shift s=1 sigma=1 a=0.5] exact D2=0.0686  numeric=0.0686  bound=0.2500",
        "[shift s=1 sigma=1 a=1] exact D2=0.3574  numeric=0.3574  bound=1.0000",
        "largest |W1|=0.6000  largest |W2|=0.4000",
        # Random stopping.
        "condition x1=1.00  fixed t=1: 0.0010  fixed t=n: 1.0000",
        "exact average t=1: 0.008527  t=n: 0.001717  (e-1) slope: 0.012862  slope 2: 0.014971",
        "stated 4 ln n: 0.013816  H_n=7.4855",
        "for every n >= 37 (checked to 10^6; 2(e-1)(1 + ln n) <= 4 ln n once n >= 445.5)",
    ):
        assert expected in out, expected
    assert {f.name for f in tmp_path.iterdir()} == {
        "apx_amp_direction_pairs.svg",
        "apx_amp_direction_tanh.svg",
        "apx_amp_poisson.svg",
        "apx_amp_clone_bars.svg",
        "apx_amp_clone_loss.svg",
        "apx_amp_shift.svg",
        "apx_amp_stopping.svg",
    }
