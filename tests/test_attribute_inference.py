"""code/attribute_inference.py runs, writes its figures, and reproduces the numbers in 03-attribute-inference.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "book" / "part02-attacks" / "code"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_attribute_inference_matches_section(tmp_path: Path) -> None:
    out = run(CODE / "attribute_inference.py", tmp_path)
    for expected in (
        # The smoking table
        "s=0: x=0,y=0:0.378  x=0,y=1:0.042  x=1,y=0:0.196  x=1,y=1:0.084",
        "s=1: x=0,y=0:0.072  x=0,y=1:0.048  x=1,y=0:0.054  x=1,y=1:0.126",
        "Pr[smoker | x, y]: x=0,y=0:0.1600  x=0,y=1:0.5333  x=1,y=0:0.2160  x=1,y=1:0.6000",
        "V(S)=0.7000  V(S|X)=0.7000  V(S|X,Y)=0.7480  leakage=0.0957 bits",
        # The toy population
        "prior guess Pr[s|x]: 0.7045  Bayes imputation Pr[s|x,y]: 0.7517",
        # The game at n = 100
        "mean log-loss: members=0.2965  non-members=0.6690  gap=0.3725",
        "prior guess:      members=0.7017  non-members=0.7089",
        "imputation:       members=0.7508  non-members=0.7555  advantage=-0.0046",
        "MAP inversion:    members=0.7832  non-members=0.7262  advantage=+0.0569 (standard error 0.0045)",
        "members above the Bayes imputation baseline: +0.0315  non-members below it: -0.0254",
        "as a membership test: TPR=0.7832  FPR=0.7262  balanced accuracy=0.5285",
        # Confidence masking
        "masking=rounded  members=0.7836  non-members=0.7253  advantage=+0.0583",
        "masking=label    members=0.7741  non-members=0.7092  advantage=+0.0650",
        # Training-set size
        "n=25: members=0.8420  non-members=0.6810  advantage=+0.1610",
        "n=400: members=0.7611  non-members=0.7500  advantage=+0.0111",
        "n=1600: members=0.7550  non-members=0.7497  advantage=+0.0052",
    ):
        assert expected in out
    assert {f.name for f in tmp_path.iterdir()} == {
        "attr_smoking_posterior.svg",
        "attr_inversion_evidence.svg",
        "attr_success_vs_n.svg",
        "attr_masking.svg",
    }
