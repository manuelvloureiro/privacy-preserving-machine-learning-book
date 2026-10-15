"""code/part05-dp-ml/disparate_impact.py runs, writes its figures, and reproduces the numbers in book-full/part05-dp-ml/08-disparate-impact.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part05-dp-ml"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_disparate_impact_matches_chapter(tmp_path: Path) -> None:
    out = run(CODE / "disparate_impact.py", tmp_path)
    for expected in (
        # The task
        "minority records=500  share=0.05  mean input norm: majority 7.04  minority 8.89",
        # Per-group accuracy against epsilon
        "non-private: majority 0.9529 +- 0.0018  minority 0.9411 +- 0.0052  gap 0.0119",
        "clipped, no noise: majority 0.9581 +- 0.0015  minority 0.9007 +- 0.0048  gap 0.0574",
        "eps=1 (sigma=4.13): majority 0.8822 +- 0.0032  minority 0.6133 +- 0.0173  gap 0.2689",
        "eps=2 (sigma=2.28): majority 0.9366 +- 0.0082  minority 0.7574 +- 0.0379  gap 0.1791",
        "eps=4 (sigma=1.36): majority 0.9470 +- 0.0037  minority 0.8477 +- 0.0211  gap 0.0993",
        "eps=8 (sigma=0.917): majority 0.9550 +- 0.0006  minority 0.8553 +- 0.0110  gap 0.0997",
        "privacy cost clipped, no noise: majority -0.0051  minority 0.0404",
        "privacy cost eps=1: majority 0.0707  minority 0.3278  ratio 4.6",
        "privacy cost eps=2: majority 0.0163  minority 0.1836  ratio 11.2",
        "privacy cost eps=4: majority 0.0059  minority 0.0934  ratio 15.7",
        "privacy cost eps=8: majority -0.0020  minority 0.0858",
        # Clipping in two dimensions
        "unclipped mean (0.1900, 0.2000) length 0.2759",
        "clipped mean (0.1900, 0.0250) length 0.1916",
        "angle between them 39.0 degrees",
        "minority share of the update: unclipped 0.513  clipped 0.116",
        "scaled by C/Z with Z=4: (0.0238, 0.0250)  angle 0.0 degrees",
        # Clipping during training
        (
            "clipped, no noise step 0: median norm 3.4e+00 / 3.7e+00  clipped 1.00 / 1.00  mean weight 0.165 / 0.163  "
            "effective share 0.0495  test acc 0.5064 / 0.5019"
        ),
        (
            "clipped, no noise step 2500: median norm 3.8e-04 / 2.7e+00  clipped 0.18 / 0.70  mean weight 0.869 / 0.424  "
            "effective share 0.0251  test acc 0.9569 / 0.6724"
        ),
        (
            "clipped, no noise step 5000: median norm 1.4e-04 / 7.1e-01  clipped 0.14 / 0.54  mean weight 0.891 / 0.579  "
            "effective share 0.0331  test acc 0.9607 / 0.8137"
        ),
        (
            "clipped, no noise step 7500: median norm 7.7e-05 / 3.0e-01  clipped 0.13 / 0.45  mean weight 0.906 / 0.651  "
            "effective share 0.0364  test acc 0.9621 / 0.8661"
        ),
        (
            "clipped, no noise step 10000: median norm 7.0e-05 / 1.6e-01  clipped 0.11 / 0.39  mean weight 0.917 / 0.702  "
            "effective share 0.0387  test acc 0.9613 / 0.8900"
        ),
        (
            "eps=4 step 2500: median norm 6.2e-04 / 2.3e+00  clipped 0.19 / 0.69  mean weight 0.858 / 0.437  "
            "effective share 0.0261"
        ),
        (
            "eps=4 step 10000: median norm 1.5e-04 / 4.7e-01  clipped 0.13 / 0.49  mean weight 0.902 / 0.637  "
            "effective share 0.0358"
        ),
        # Clipping norm
        "C=0.05 eps=4: majority 0.9541 +- 0.0008  minority 0.8313 +- 0.0157  gap 0.1229",
        "C=5 eps=4: majority 0.9463 +- 0.0027  minority 0.8133 +- 0.0166  gap 0.1330",
        # Noise floor
        "tau = sigma*C/B at eps=4: 0.0068",
        "share 0.95: predicted excess loss 4.07e-04  simulated 4.07e-04  relaxation steps 2",
        "share 0.2: predicted excess loss 1.55e-03  simulated 1.55e-03  relaxation steps 10",
        "share 0.05: predicted excess loss 5.97e-03  simulated 5.99e-03  relaxation steps 40",
        "share 0.01: predicted excess loss 2.96e-02  simulated 2.97e-02  relaxation steps 200",
        "break-even, signal share 1 of 0.026: 228 steps",
        "break-even, signal share 0.2 of 0.026: 5693 steps",
        "break-even, signal share 0.05 of 0.026: 91085 steps",
        "break-even, signal share 0.01 of 0.026: 2277115 steps",
        # Group size
        "share 0.01 (100 records) non-private: majority 0.9556 +- 0.0015  minority 0.8498 +- 0.0021",
        (
            "share 0.01 eps=4: majority 0.9558 +- 0.0036  minority 0.5009 +- 0.0102  gap 0.4549  "
            "privacy cost majority -0.0002  minority 0.3488"
        ),
        "share 0.2 (2000 records) non-private: majority 0.9395 +- 0.0016  minority 0.9787 +- 0.0004",
        (
            "share 0.2 eps=4: majority 0.9355 +- 0.0019  minority 0.9507 +- 0.0038  gap -0.0152  "
            "privacy cost majority 0.0041  minority 0.0280"
        ),
        # Measuring
        "model at eps=4: minority accuracy on 20000 test records 0.8641",
        "100 minority test records: 88 correct  estimate 0.880  95% CI [0.800, 0.936]",
        "500 minority test records: 424 correct  estimate 0.848  95% CI [0.813, 0.878]",
        "2000 minority test records: 1742 correct  estimate 0.871  95% CI [0.856, 0.885]",
        "1900 majority test records: 1812 correct  estimate 0.954  95% CI [0.943, 0.963]",
        # Mitigation
        "oversample x4: sigma=4.71  eps of majority records=0.86  majority 0.8669 +- 0.0079  minority 0.8270 +- 0.0111",
        "oversample x4 at sigma=1.36: eps of minority records=20.96",
    ):
        assert expected in out, expected
    for name in (
        "fair_privacy_accuracy.svg",
        "fair_clipping_toy.svg",
        "fair_clipping_training.svg",
        "fair_group_size.svg",
        "fair_intervals.svg",
    ):
        assert (tmp_path / name).exists()
