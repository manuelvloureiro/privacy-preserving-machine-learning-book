"""code/part05-dp-ml/dp_sgd.py runs, writes its figures, and reproduces the numbers in book-full/part05-dp-ml/02-dp-sgd.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part05-dp-ml"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_dp_sgd_matches_chapter(tmp_path: Path) -> None:
    out = run(CODE / "dp_sgd.py", tmp_path)
    for expected in (
        # The algorithm
        "q=0.01, sigma=1, T=10000, delta=1e-5: eps=6.71 at alpha=4.1",
        "weights p=3329  expected batch B=100  epochs=100",
        "sd per coordinate=0.0050  expected length about sd*sqrt(p)=0.288",
        "eps=0.5: sigma=7.72",
        "eps=1: sigma=4.13",
        "eps=2: sigma=2.28",
        "eps=4: sigma=1.36",
        "eps=8: sigma=0.917",
        # Ghost norms
        "largest relative difference ghost vs explicit = 2.2e-16",
        "sequence of 16 tokens: Gram-matrix norm=33.144228  explicit=33.144228",
        "B=100, T=1, p=51, r=64: explicit 326,400 floats  ghost 11,500 floats  ratio 28.4",
        "B=32, T=1, p=768, r=768: explicit 18,874,368 floats  ghost 49,152 floats  ratio 384.0",
        "B=32, T=512, p=768, r=768: explicit 18,874,368 floats  ghost 16,777,216 floats  ratio 1.1",
        "B=32, T=512, p=768, r=3072: explicit 75,497,472 floats  ghost 16,777,216 floats  ratio 4.5",
        "B=8, T=4096, p=4096, r=4096: explicit 134,217,728 floats  ghost 268,435,456 floats  ratio 0.5",
        # Training
        "non-private SGD, 1000 steps: test accuracy 0.9627",
        "non-private SGD, 10000 steps: test accuracy 0.9617",
        "DP-SGD sigma=1 (eps=6.71): test accuracy 0.9589",
        "step 0: test acc 0.4872  clipped 1.00  median norm 3.592  length of clipped mean 0.070",
        "step 1000: test acc 0.9517  clipped 0.25  median norm 0.006  length of clipped mean 0.026",
        "step 3000: test acc 0.9519  clipped 0.17  median norm 0.000  length of clipped mean 0.021",
        "step 7000: test acc 0.9544  clipped 0.09  median norm 0.000  length of clipped mean 0.015",
        # Privacy against accuracy
        "eps=0.5 (sigma=7.72): test accuracy 0.8209 +- 0.0180",
        "eps=1 (sigma=4.13): test accuracy 0.9175 +- 0.0161",
        "eps=2 (sigma=2.28): test accuracy 0.9441 +- 0.0051",
        "eps=4 (sigma=1.36): test accuracy 0.9548 +- 0.0019",
        "eps=8 (sigma=0.917): test accuracy 0.9604 +- 0.0019",
        "non-private, 1000 steps: test accuracy 0.9610 +- 0.0005",
        "clipped, no noise, 10000 steps: test accuracy 0.9649 +- 0.0009",
        # Clipping
        "C=0.01: fixed lr 0.9244  lr*C fixed 0.9627",
        "C=0.03: fixed lr 0.9510  lr*C fixed 0.9623",
        "C=0.1: fixed lr 0.9567  lr*C fixed 0.9605",
        "C=0.3: fixed lr 0.9607  lr*C fixed 0.9599",
        "C=1: fixed lr 0.9583  lr*C fixed 0.9590",
        "C=3: fixed lr 0.9523  lr*C fixed 0.9558",
        "C=10: fixed lr 0.8347  lr*C fixed 0.9494",
        # Batch size
        "q=0.001: B=10  steps=100000  sigma=0.704  sigma/B=0.0704  sigma/sqrt(q)=22.3  (e^(1/sigma^2)-1)*sigma^2=3.23",
        "q=0.003: B=30  steps=33333  sigma=0.909  sigma/B=0.0303  sigma/sqrt(q)=16.6  (e^(1/sigma^2)-1)*sigma^2=1.95",
        "q=0.01: B=100  steps=10000  sigma=1.36  sigma/B=0.0136  sigma/sqrt(q)=13.6  (e^(1/sigma^2)-1)*sigma^2=1.33",
        "q=0.03: B=300  steps=3333  sigma=2.16  sigma/B=0.0072  sigma/sqrt(q)=12.5  (e^(1/sigma^2)-1)*sigma^2=1.12",
        "q=0.1: B=1000  steps=1000  sigma=3.78  sigma/B=0.00378  sigma/sqrt(q)=12  (e^(1/sigma^2)-1)*sigma^2=1.04",
        "q=0.3: B=3000  steps=333  sigma=6.44  sigma/B=0.00215  sigma/sqrt(q)=11.8  (e^(1/sigma^2)-1)*sigma^2=1.01",
        "q=1: B=10000  steps=100  sigma=10.8  sigma/B=0.00108  sigma/sqrt(q)=10.8  (e^(1/sigma^2)-1)*sigma^2=1",
        "q=0.003: lr=0.25: 0.9544  lr=0.5: 0.9487  lr=1: 0.9453",
        "q=0.01: lr=0.5: 0.9522  lr=1: 0.9569  lr=2: 0.9465",
        "q=0.03: lr=1: 0.9591  lr=2: 0.9565  lr=4: 0.9577",
        "q=0.1: lr=2: 0.9589  lr=4: 0.9565  lr=8: 0.9576",
    ):
        assert expected in out, expected
    for name in ("dpsgd_privacy_accuracy.svg", "dpsgd_training.svg", "dpsgd_clipping.svg", "dpsgd_batch_size.svg"):
        assert (tmp_path / name).exists()
