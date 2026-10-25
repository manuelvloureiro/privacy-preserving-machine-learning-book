"""code/part02-attacks/property_inference.py runs, writes its figures, and reproduces the numbers in book-full/part02-attacks/04-property-inference.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part02-attacks"


def test_property_inference_matches_section(tmp_path: Path) -> None:
    out = run(CODE / "property_inference.py", tmp_path)
    for expected in (
        # A released count
        "n=100: property TV=0.6827  normal approx=0.6827  membership=0.0398  (1-t) max pmf=0.0398  "
        "normal approx=0.0401",
        "n=1000: property TV=0.9985  normal approx=0.9984  membership=0.0126  (1-t) max pmf=0.0126  "
        "normal approx=0.0126",
        "n=10000: property TV=1.0000  normal approx=1.0000  membership=0.0040  (1-t) max pmf=0.0040  "
        "normal approx=0.0040",
        "simulated property game, n=100, 200000 games: advantage=0.6811",
        # Logistic regression: the population view
        "t=0.00: w1=2.000  w2=1.000  bias=0.000",
        "t=0.25: w1=1.767  w2=0.442  bias=0.000",
        "t=0.50: w1=1.704  w2=0.000  bias=0.000",
        "t=0.75: w1=1.767  w2=-0.442  bias=0.000",
        "t=1.00: w1=2.000  w2=-1.000  bias=0.000",
        "Sigma_1(0.5) diagonal: 12.59  6.94  6.04",
        "d theta/dt at 0.5: (0.000, -1.704, 0.000)  kappa=0.647  released-count kappa=2  "
        "effective fraction (kappa/2)^2=0.105",
        "2000 models at n=1000, t=0.5: mean w1=1.693  n * var: 11.92  6.80  6.08",
        # Logistic regression: the meta-classifier sweep
        "n=50: d=0.05: 0.523 (0.546)  d=0.1: 0.520 (0.590)  d=0.2: 0.653 (0.676)  d=0.4: 0.830 (0.818)",
        "n=100: d=0.05: 0.523 (0.564)  d=0.1: 0.670 (0.627)  d=0.2: 0.733 (0.741)  d=0.4: 0.863 (0.901)",
        "n=200: d=0.05: 0.503 (0.590)  d=0.1: 0.670 (0.676)  d=0.2: 0.803 (0.819)  d=0.4: 0.957 (0.965)",
        "n=500: d=0.05: 0.633 (0.641)  d=0.1: 0.787 (0.765)  d=0.2: 0.910 (0.926)  d=0.4: 0.997 (0.998)",
        "n=1000: d=0.05: 0.713 (0.695)  d=0.1: 0.867 (0.847)  d=0.2: 0.980 (0.979)  d=0.4: 1.000 (1.000)",
        # Tanh network
        "raw parameters: accuracy=0.497  largest standardized mean gap=0.34",
        "canonical order: accuracy=0.807  largest standardized mean gap=0.66",
        "unit second moments: accuracy=0.877  largest standardized mean gap=2.24",
        "probe outputs: accuracy=1.000  largest standardized mean gap=2.81",
        "permuted and sign-flipped network: same logits on 1000 inputs (to 1e-12): True",
        "equivalent parameter vectors per network: h! 2^h = 384",
    ):
        assert expected in out
    assert {f.name for f in tmp_path.iterdir()} == {
        "prop_count_leak.svg",
        "prop_parameter_cloud.svg",
        "prop_meta_accuracy.svg",
        "prop_symmetries.svg",
    }
