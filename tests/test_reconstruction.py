"""code/part02-attacks/reconstruction.py runs, writes its figures, and reproduces the numbers in book-full/part02-attacks/09-reconstruction.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part02-attacks"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_reconstruction_matches_section(tmp_path: Path) -> None:
    out = run(CODE / "reconstruction.py", tmp_path)
    for expected in (
        # Fact 1
        "d=[1, 0, 1, 1]  answers to the 16 subsets: [0, 1, 0, 1, 1, 2, 1, 2, 1, 2, 1, 2, 2, 3, 2, 3]",
        # The exponential attack
        "d=[1, 1, 1, 0, 0, 0, 0, 0, 0, 1]  ones=4",
        "E=0: consistent candidates=1  max Hamming distance=0  bound 4E=0",
        "E=0.5: consistent candidates=35  max Hamming distance=2  bound 4E=2",
        "E=1: consistent candidates=242  max Hamming distance=4  bound 4E=4",
        "E=1.5: consistent candidates=630  max Hamming distance=6  bound 4E=6",
        "E=2: consistent candidates=912  max Hamming distance=8  bound 4E=8",
        "rounding to multiples of 2: consistent candidates=1  max Hamming distance=0",
        "rounding to multiples of 4: consistent candidates=1  max Hamming distance=0",
        # Anti-concentration and the linear-programming decoder
        "E[(s1+s2+s3)^4]: enumeration=21  formula=21",
        "escape bound at theta=1/4: 0.1875",
        "n=1000 beta=0.1: threshold sqrt(beta n)/12=0.8333",
        "aligned z: Pr[|<q,z>| > threshold]=1.0000",
        "balanced z: Pr[|<q,z>| > threshold]=0.6161",
        "error tolerance E <= 0.1667  queries m >= 33296 (33.3 n)",
        # Least squares
        "sigma=0.1 sqrt(n)=2.24: recovered=1.0000  predicted=1.0000",
        "sigma=0.25 sqrt(n)=5.59: recovered=0.9655  predicted=0.9585",
        "sigma=0.5 sqrt(n)=11.18: recovered=0.8130  predicted=0.8069",
        "sigma=1 sqrt(n)=22.36: recovered=0.6550  predicted=0.6676",
        "sigma=2 sqrt(n)=44.72: recovered=0.5875  predicted=0.5858",
        "sigma=2 sqrt(n): m/n for 90% recovery=106.1",
        "sigma=5 sqrt(n): m/n for 90% recovery=657.9",
        "smallest singular value=11.43  (sqrt(m) - sqrt(n))/2=11.18",
        "E=1.00: guaranteed wrong bits <= 61.2  adversarial curator gets 0 wrong",
        "E=2.00: guaranteed wrong bits <= 245.0  adversarial curator gets 20 wrong",
        "E=5.59: guaranteed wrong bits <= 1914.0  adversarial curator gets 143 wrong",
        # The Fundamental Law
        "kept 250 of 500  max error=28.00 (1.25 sqrt(n))  Hoeffding bound (eta=0.05)=106.25  recovered=0.7560",
        "keep=0.9: Hoeffding bound=59.03  ceiling=0.95",
        "n=1000 beta=0.05: h(beta)=0.2864  exact answers m >= 71.6",
        "n=1000 beta=0.01: h(beta)=0.0808  exact answers m >= 92.2",
        "n=1000 beta=0.05 rounded to 10: values=102  m >= 106.9",
        "n=1000 beta=0.05 rounded to 32: values=33  m >= 141.5",
        # The census block
        "total: count=7  median=33  mean=34.29",
        "female: count=4  median=31  mean=30.25",
        "male: count=3  median=33  mean=39.67",
        "black: count=4  median=33.5  mean=36.25",
        "white: count=3  median=33  mean=31.67",
        "married: count=4  median=43.5  mean=47.75",
        "female age tuples=435  male age tuples=34  age vectors=14790  age vectors with the total median=12906",
        "age vectors with a valid black split=387  age vectors with a valid married split=33  "
        "age vectors with valid splits for all three=1",
        "consistent blocks with the full table: 1  equal to the true block: True",
        "withhold female median: 7 consistent blocks, 6 attribute patterns, true pattern in 0.14 of them",
        "withhold black median: 36 consistent blocks, 3 attribute patterns, true pattern in 0.64 of them",
        "withhold married median: 26 consistent blocks, 4 attribute patterns, true pattern in 0.46 of them",
        "withhold married mean: 45 consistent blocks, 7 attribute patterns, true pattern in 0.51 of them",
        # A deployed query system
        "queries=3500",
        "n=73: mean inclusion rate mod 2=0.502  mod 5=0.198  mean rho(1-rho)=0.2016",
        "n=73: sigma for 75% recovery / Diffix sigma=4.9",
        "n=73, sigma=2 sqrt(n)=17.1: predicted fraction with all 3500 queries=0.779",
        (
            "n=73: sigma/sqrt(n)=0.47  perfect in 49 of 50 runs  wrong bits per run=0.02  "
            "predicted fraction=0.99950  predicted wrong bits=0.036  sigma for 75% recovery=19.5"
        ),
        "n=142: sigma/sqrt(n)=0.34  perfect in 45 of 50 runs  wrong bits per run=0.10",
        "predicted wrong bits=0.077",
        "predicted fraction=0.99946",
    ):
        assert expected in out
    for statistic in (
        "total median",
        "total mean",
        "female mean",
        "male median",
        "male mean",
        "black mean",
        "white median",
        "white mean",
    ):
        assert f"withhold {statistic}: 1 consistent blocks, 1 attribute patterns, true pattern in 1.00" in out
    assert {f.name for f in tmp_path.iterdir()} == {
        "recon_exponential.svg",
        "recon_anticoncentration.svg",
        "recon_ls_noise.svg",
        "recon_ls_queries.svg",
        "recon_census.svg",
        "recon_diffix.svg",
    }
