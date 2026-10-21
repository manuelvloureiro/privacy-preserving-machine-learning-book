"""code/part02-attacks/practitioner_lira.py runs and prints the numbers that the practitioner edition quotes.

The pages are book-practitioner/part02-attacks/10-risk-assessment.md (steps 2 and 3) and 02-membership-inference.md.
The script fixes BLAS to one thread, so the logistic-regression fits and every printed digit are reproducible.
"""

import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "code" / "part02-attacks" / "practitioner_lira.py"


@pytest.mark.libs
def test_practitioner_lira_matches_pages() -> None:
    out = subprocess.run([sys.executable, str(SCRIPT)], capture_output=True, text=True, check=True).stdout
    lines = {line.strip() for line in out.splitlines()}
    for expected in (
        "member pairs=57,704  non-member pairs=57,304",
        "target 0: train accuracy=1.0000  test accuracy=0.9650",
        "train-test gap, mean over 64 targets=0.0422",
        "loss attack at FPR 0.001: TPR=0.0009",
        "loss attack at FPR 0.01: TPR=0.0105",
        "pairs without a fit, dropped: 0",
        "loss attack: AUC=0.518  TPR@0.001=0.0009  TPR@0.01=0.0105",
        "LiRA offline: AUC=0.577  TPR@0.001=0.0087  TPR@0.01=0.0345",
        "LiRA online: AUC=0.623  TPR@0.001=0.0318  TPR@0.01=0.0790",
        "LiRA online, global variance: AUC=0.614  TPR@0.001=0.0250  TPR@0.01=0.0737",
        "loss attack: TP=53/57704  TPR=0.0009 [0.0007, 0.0012]  FP=57/57304  FPR=0.00099 [0.00075, 0.00129]  ln(TPR/FPR)=-0.08  pass",
        "LiRA offline: TP=502/57704  TPR=0.0087 [0.0080, 0.0095]  FP=56/57304  FPR=0.00098 [0.00074, 0.00127]  ln(TPR/FPR)=2.19  pass",
        "LiRA online: TP=1833/57704  TPR=0.0318 [0.0303, 0.0332]  FP=57/57304  FPR=0.00099 [0.00075, 0.00129]  ln(TPR/FPR)=3.46  block",
        "LiRA online, global variance: TP=1445/57704  TPR=0.0250 [0.0238, 0.0263]  FP=57/57304  FPR=0.00099 [0.00075, 0.00129]  ln(TPR/FPR)=3.23  block",
        "loss attack: TP=608/57704  TPR=0.0105 [0.0097, 0.0114]  FP=573/57304  FPR=0.01000 [0.00920, 0.01085]  ln(TPR/FPR)=0.05",
        "LiRA offline: TP=1992/57704  TPR=0.0345 [0.0330, 0.0360]  FP=573/57304  FPR=0.01000 [0.00920, 0.01085]  ln(TPR/FPR)=1.24",
        "LiRA online: TP=4557/57704  TPR=0.0790 [0.0768, 0.0812]  FP=571/57304  FPR=0.00996 [0.00917, 0.01081]  ln(TPR/FPR)=2.07",
        "LiRA online, global variance: TP=4253/57704  TPR=0.0737 [0.0716, 0.0759]  FP=573/57304  FPR=0.01000 [0.00920, 0.01085]  ln(TPR/FPR)=2.00",
        "image 1658: label 9  separation=9.24  confidence IN=0.812 OUT=0.003  flagged by online LiRA at FPR 0.001 in 33 of 33 models",
        "image 480: label 7  separation=7.33  confidence IN=0.932 OUT=0.010  flagged by online LiRA at FPR 0.001 in 31 of 31 models",
        "image 1361: label 5  separation=6.74  confidence IN=0.946 OUT=0.049  flagged by online LiRA at FPR 0.001 in 33 of 33 models",
        "image 421: label 5  separation=6.33  confidence IN=0.954 OUT=0.022  flagged by online LiRA at FPR 0.001 in 33 of 33 models",
        "image 1553: label 8  separation=5.90  confidence IN=0.842 OUT=0.005  flagged by online LiRA at FPR 0.001 in 27 of 27 models",
        "image 1264: label 1  separation=5.60  confidence IN=0.999 OUT=0.035  flagged by online LiRA at FPR 0.001 in 37 of 37 models",
        "image 492: label 6  separation=5.38  confidence IN=0.944 OUT=0.131  flagged by online LiRA at FPR 0.001 in 33 of 33 models",
        "image 792: label 6  separation=5.10  confidence IN=0.948 OUT=0.018  flagged by online LiRA at FPR 0.001 in 34 of 34 models",
        "image 1729: label 3  separation=5.08  confidence IN=0.953 OUT=0.025  flagged by online LiRA at FPR 0.001 in 38 of 38 models",
        "image 1727: label 3  separation=5.07  confidence IN=0.937 OUT=0.020  flagged by online LiRA at FPR 0.001 in 30 of 30 models",
    ):
        assert expected in lines, expected
