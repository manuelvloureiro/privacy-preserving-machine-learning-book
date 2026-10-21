"""code/appendix-derivations/apx_composition.py runs, writes its figures, and reproduces the numbers in book-full/appendix-derivations/11-composition.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "appendix-derivations"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_appendix_composition_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "apx_composition.py", tmp_path)
    for expected in (
        # canonical pair
        "G_1(0.1)=0.6109  power=0.3891",
        "max gap G_1=2.0e-03  f_(1,0.1)=8.3e-05",
        "corner a*=0.2420",
        # adaptive composition by test replacement
        "vertices a=[0.0, 0.1, 0.25, 0.4375, 1.0] b=[1.0, 0.55, 0.25, 0.0625, 0.0]",
        "product vertices a=[0.0, 0.0625, 0.25, 0.4375, 1.0] b=[1.0, 0.4375, 0.25, 0.0625, 0.0]",
        "min gap=0.0e+00  max gap=0.2812 at a=0.0625",
        "branch y1=0: level=0.4000 power 0.6000 -> 0.8000  joint (0.1000, 0.5500) -> (0.1000, 0.4000)",
        "original=(0.5625, 0.9375)  after step 2=(0.5625, 0.7708)  after step 4=(0.5625, 0.1042)",
        "violations=0  max |change in a|=3.3e-16  mean drop in b=0.4013",
        # dominating pairs
        "min E_gamma margin over 502 gammas in [0, e^4]=0.0e+00",
        "eps=0: adaptive delta=0.5750  product delta=0.6875",
        "eps=1: adaptive delta=0.3926  product delta=0.4190",
        "eps=2: adaptive delta=0.1528  product delta=0.3064",
        "eps=3: adaptive delta=0.0000  product delta=0.1080",
        "gamma=1.0: E_gamma=0.5000",
        "gamma=2.0: E_gamma=0.2500",
        "above the product's boundary: 0  smallest margin=0.0726",
        "gamma=0.5000: E_gamma mu=1 0.5953  mu=0.6 0.5253",
        "gamma=1.0000: E_gamma mu=1 0.3829  mu=0.6 0.2358",
        "gamma=2.7183: E_gamma mu=1 0.1269  mu=0.6 0.0190",
        # the f-DP CLT
        "k=1 eps0=1.0000 mu=1.0422 gamma=0.7663",
        "k=10 eps0=0.3162 mu=1.0042 gamma=0.1837  slack lower=0.2151 upper=0.2420  sup|F-Phi| P=0.1331 Q=0.1331  max|f-G_mu|=0.0137",
        "k=100 eps0=0.1000 mu=1.0004 gamma=0.0562  slack lower=0.0611 upper=0.0647  sup|F-Phi| P=0.0405 Q=0.0405  max|f-G_mu|=0.0014",
        "k=1000 eps0=0.0316 mu=1.0000 gamma=0.0177  slack lower=0.0187 upper=0.0193  sup|F-Phi| P=0.0127 Q=0.0127  max|f-G_mu|=0.0001",
        "k=100: K=0.4996  V=0.9975",
        "eps=0.1: kl=0.004996",
        "over eps in (0, 1]: 0.0417",
        # the subsampled Gaussian
        "v=e^(1/s^2)-1=1.7183  leading mean q^2 v/2=8.5914e-05  leading var q^2 v=1.7183e-04",
        "exact: mean_P=8.1890e-05 var_P=1.6009e-04 mean_Q=-8.3812e-05 var_Q=1.7163e-04",
        "q=0.02: mean P -3.619",
        "q=0.01: mean P -4.024",
        "q=0.005: mean P -4.292",
        "q=0.0025: mean P -4.453 Q 2.257  var P -13.244",
        "E|W|^3 <= 21.09  E W^4 <= 404.43  mean bound (2/3)E|W|^3=14.06",
        "mu=0.1311 CLT eps=0.4575  Berry-Esseen P=0.3045",
        "sup|F-Phi| P in [0.0312, 0.0343]",
        "mu=0.4145 CLT eps=1.6177  Berry-Esseen P=0.0963",
        "mu=1.3108 CLT eps=6.0071  Berry-Esseen P=0.0304",
        "add [0.3377, 0.3387]  remove [0.7175, 0.7185]",
        "add [1.4117, 1.4217]  remove [1.8232, 1.8332]",
        "add [5.5615, 5.6615]  remove [6.1377, 6.2377]",
    ):
        assert expected in out, expected
    assert {f.name for f in tmp_path.iterdir()} == {
        "apx_comp_canonical.svg",
        "apx_comp_replacement.svg",
        "apx_comp_regions.svg",
        "apx_comp_three_step.svg",
        "apx_comp_clt_band.svg",
        "apx_comp_clt_cdfs.svg",
        "apx_comp_sgd_cdfs.svg",
        "apx_comp_sgd_tails.svg",
    }
