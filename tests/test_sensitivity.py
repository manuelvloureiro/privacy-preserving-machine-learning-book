"""code/part03-dp-foundations/sensitivity.py runs, writes its figures, and reproduces the numbers in book-full/part03-dp-foundations/07-sensitivity.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part03-dp-foundations"


def test_sensitivity_matches_section(tmp_path: Path) -> None:
    out = run(CODE / "sensitivity.py", tmp_path)
    squeezed = " ".join(out.split())  # the walkthrough rows are column-aligned
    for expected in (
        # Local sensitivity leak
        "x: median=30  LS=10;  x': median=30  LS=60",
        "L_x,x'=1.792",
        "|y - median| > 33.50; Pr under x' = 0.572, under x = 0.0351",
        "delta(ε=1): x' vs x = 0.477  x vs x' = 0.215",
        # Walkthrough
        "formula: 2.00 5.00 6.00 8.00 9.00 10.00",
        "brute: 2.00 5.00 6.00 8.00 9.00 10.00",
        "weighted: 2.00 3.03 2.21 1.79 1.22 0.82",
        "S*=3.033 at k=1  max |formula - brute|=0",
        # Noise parameters
        "Cauchy: α=0.1667 β=0.1667  loss bound=0.8333",
        "Laplace: α=0.5 β=0.0322 (NRS β=0.0345",
        # Median of the ages
        "median=36.96  LS=0.1386  global scale Λ/ε=100",
        "Cauchy: S*=0.1632 at k=7  noise scale=0.979",
        "Laplace: S*=0.5144 at k=21  noise scale=1.029  median |error|=0.713",
        "global Laplace median |error|=69.31",
        "simulated median |error|: Cauchy=0.980  Laplace=0.713",
        "(exact 0.0635)",
        # Propose-test-release
        "threshold ln(1/δ)/ε=13.82",
        "n=1001: distance=21  pass probability=0.999621",
        "n=101: distance=1  pass probability=1.36e-06",
        # Subsample-and-aggregate
        "B=10: bias=0.0010  sampling sd=0.0100  noise sd=0.7071  rmse=0.7072  simulated=0.7205",
        "B=100: bias=0.0101  sampling sd=0.0102  noise sd=0.0707  rmse=0.0722  simulated=0.0733",
        "B=250: bias=0.0256  sampling sd=0.0105  noise sd=0.0283  rmse=0.0396  simulated=0.0401",
        "B=1000: bias=0.1111  sampling sd=0.0124  noise sd=0.0071  rmse=0.1120  simulated=0.1114",
        "250  rmse=0.0396  non-private sd θ/sqrt(n)=0.0100",
        # Clipping
        "true mean=55.085  non-private sd of the unclipped mean=1.649",
        "C*: C=331.34  tail=0.0041  bias=0.438  sampling sd=1.538  noise sd=0.469  rmse=1.666  simulated=1.649",
        "q50: C=40.00  tail=0.5000  bias=23.415  sampling sd=0.335  noise sd=0.057  rmse=23.417  simulated=23.411",
        "q99: C=257.23  tail=0.0100  bias=0.924  sampling sd=1.462  noise sd=0.364  rmse=1.767  simulated=1.747",
        "C=1000: C=1000.00  tail=0.0000  bias=0.006  sampling sd=1.645  noise sd=1.414  rmse=2.169  simulated=2.134",
        "n=100: C*=177.5",
        "n=10000: C*=557.3",
        "n=100000: C*=876.3",
        # Inverse sensitivity
        "lengths: 1 2 1 2 3 1",
        "len: 3 2 1 1 2 3",
        "probs: 0.0544 0.1792 0.1477 0.2955 0.2688 0.0544",
        "Pr[3 <= t <= 6]=0.4432",
        "median |error|=0.0783  simulated=0.0784  Cauchy smooth=0.979  LS=0.1386",
        "n=101: median |error|=0.696",
        # Stability-based release
        "lead=10: distance=4  pass probability=2.73e-05",
        "lead=30: distance=14  pass probability=0.5842",
        "lead=40: distance=19  pass probability=0.9972",
        "lead needed for pass probability 0.95: 35",
    ):
        assert expected in out or expected in squeezed, expected
    assert {f.name for f in tmp_path.iterdir()} == {
        "sens_local_leak.svg",
        "sens_smooth_profile.svg",
        "sens_methods_vs_n.svg",
        "sens_saa_blocks.svg",
        "sens_clip_tradeoff.svg",
    }
