"""code/part05-dp-ml/practitioner_audit.py runs and prints the numbers that the practitioner auditing chapter quotes."""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "code" / "part05-dp-ml" / "practitioner_audit.py"


@pytest.mark.libs
def test_practitioner_audit_seed_0() -> None:
    out = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, check=True).stdout
    for expected in (
        "sigma=0.7  C=1.0  B=64  steps per run=2000  claim mu=1/sigma=1.429",
        "correct: noise std 0.687 (claim 0.7)  FP 206/1000  TP 710/1000",
        "correct: mu_lo=1.201  pass",
        "variance for std: noise std 0.481 (claim 0.7)  FP 206/1000  TP 862/1000",
        "variance for std: mu_lo=1.722  FAIL",
        "no noise: noise std 0.000 (claim 0.7)  FP 0/1000  TP 1000/1000",
        "no noise: mu_lo=5.360  FAIL",
        "DP-SGD eps=4: test accuracy 0.8470",
        "DP-SGD eps=4: mean canary loss in 4.105  out 4.006",
        "DP-SGD eps=4: r=20  correct=10  eps_lo=0.00",
        "DP-SGD eps=4: r=50  correct=18  eps_lo=0.00",
        "DP-SGD eps=4: r=100  correct=42  eps_lo=0.00",
        "DP-SGD eps=4: r=200  correct=94  eps_lo=0.00",
        "DP-SGD eps=4: r=500  correct=248  eps_lo=0.00",
        "DP-SGD eps=4: r=1000  correct=493  eps_lo=0.00",
        "DP-SGD eps=4: certified eps lower bound 0.00 (r=20, 10 correct)",
        "non-private: test accuracy 0.8464",
        "non-private: mean canary loss in 2.240  out 2.222",
        "non-private: r=20  correct=8  eps_lo=0.00",
        "non-private: r=50  correct=25  eps_lo=0.00",
        "non-private: r=100  correct=48  eps_lo=0.00",
        "non-private: r=200  correct=93  eps_lo=0.00",
        "non-private: r=500  correct=251  eps_lo=0.00",
        "non-private: r=1000  correct=483  eps_lo=0.00",
        "non-private: certified eps lower bound 0.00 (r=20, 8 correct)",
        "memorizing control: test accuracy 0.8517",
        "memorizing control: mean canary loss in 1.994  out 2.992",
        "memorizing control: r=20  correct=15  eps_lo=0.00",
        "memorizing control: r=50  correct=39  eps_lo=0.43",
        "memorizing control: r=100  correct=75  eps_lo=0.53",
        "memorizing control: r=200  correct=141  eps_lo=0.49",
        "memorizing control: r=500  correct=324  eps_lo=0.38",
        "memorizing control: r=1000  correct=569  eps_lo=0.12",
        "memorizing control: certified eps lower bound 0.53 (r=100, 75 correct)",
        "train n=30537 (30000 + 537 canaries of 1000)  features=91  test n=15842  delta=1e-05",
        "DP-SGD: sigma=0.9729  lr=2  epochs=10  batch=1000  C=1  q=0.0323  steps=310  eps (PRV)=3.91",
        "ceiling: all 1000 guesses right certifies 5.27",
    ):
        assert expected in out, expected
