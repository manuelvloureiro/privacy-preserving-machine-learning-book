"""code/divergences.py runs, writes its figures, and reproduces the numbers in 03-divergences.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "book" / "part01-prerequisites" / "code"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_divergences_matches_section(tmp_path: Path) -> None:
    out = run(CODE / "divergences.py", tmp_path)
    for expected in (
        # The die
        "P‖Q: KL=0.2939  D_0.5=0.1362  D_1=0.2939  D_2=0.5878  D_5=0.9257  D_10=1.0216  D_inf=1.0986",
        "Q‖P: KL=0.2426  D_0.5=0.1362  D_1=0.2426  D_2=0.3677  D_5=0.4653  D_10=0.4906  D_inf=0.5108",
        "TV=0.3333  chi2=0.8000  Hellinger^2=0.1317  E_2=0.1667  E_3=0.0000  Pinsker=0.3833",
        "power means of r under P: M_-1=1.0000  M_0=1.3416  M_1=1.8000  M_inf=3.0000",
        "evidence per roll, P‖Q: mean=0.2939  sd=0.8047  (sd/mean)^2=7.50",
        "evidence per roll, Q‖P: mean=0.2426  sd=0.5998",
        "KL rebuilt from hockey-stick curves = 0.2939",
        "parity KL=0.0823  is-six KL=0.2939",
        # The Gaussian pair
        "KL=0.5000 (Monte Carlo 0.4990)  D_2=1.0000  D_10=5.0000",
        "TV=0.3829  best single-shot accuracy=0.6915  Pinsker=0.5000",
        "E_2=0.1906 (Monte Carlo 0.1892)  E_4=0.0692",
        "exact=0.0062  Markov e^-c=0.1353  Renyi-Chernoff=0.0439 at alpha=2.5",
        "width 0.5=0.4898  width 1=0.4618  width 2=0.3781  one bit=0.3090",
        # The Laplace pair against a Gaussian pair with the same variance
        "KL=0.3679 (Monte Carlo 0.3684)  D_0.5=0.1891  D_2=0.6191  D_10=0.9287  D_inf=1.0000  TV=0.3935  "
        "E_2=0.1422  Pinsker=0.4289",
        "KL=0.2500  D_0.5=0.1250  D_2=0.5000  D_10=2.5000  D_inf=inf",
        "overtakes Laplace D_alpha at alpha=2.98",
        # n samples
        "one sample: TV=0.0399",
        "n=100: KL=0.50  TV=0.3829  accuracy=0.6915 (simulated 0.6916)  Pinsker=0.5000  BH=0.6273",
        "n=400: KL=2.00  TV=0.6827  accuracy=0.8413 (simulated 0.8424)  Pinsker=1.0000  BH=0.9299",
        "Bretagnolle-Huber beats Pinsker for KL > 1.59",
        # Forward and reverse KL
        "forward KL fit: mu=0.00 sigma=2.06  KL(p‖q)=0.724  KL(q‖p)=2.097",
        "reverse KL fit: mu=2.00 sigma=0.50  KL(p‖q)=15.307  KL(q‖p)=0.693",
        # Exercises
        "KL(N(0,1)‖N(0,4)) = 0.3181",
        "KL(Bern(0.6)‖Bern(0.5)) = 0.02014; flips needed before Pinsker allows TV >= 0.8: 64",
    ):
        assert expected in out
    assert {f.name for f in tmp_path.iterdir()} == {
        "divergences_evidence_walk.svg",
        "divergences_renyi_tilt.svg",
        "divergences_hockey_area.svg",
        "divergences_dpi_staircase.svg",
        "divergences_forward_reverse_kl.svg",
        "divergences_renyi_chords.svg",
        "divergences_gauss_laplace_renyi.svg",
        "divergences_tv_gaussians.svg",
        "divergences_hockey_stick.svg",
        "divergences_f_generators.svg",
        "divergences_dpi_binning.svg",
        "divergences_pinsker.svg",
    }
