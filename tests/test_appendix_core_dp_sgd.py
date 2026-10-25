"""code/appendix-derivations/apx_core_dp_sgd.py runs, writes its figures, and reproduces the numbers in book-full/appendix-derivations/03-dp-sgd.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "appendix-derivations"


def test_appendix_core_dp_sgd_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "apx_core_dp_sgd.py", tmp_path)
    for expected in (
        # Clipping gives sensitivity
        "g=(3, 4) |g|=5.00 -> clipped (0.60, 0.80)",
        "g=(0.3, -0.4) |g|=0.50 -> clipped (0.30, -0.40)",
        "g=(-2.4, 0.7) |g|=2.50 -> clipped (-0.96, 0.28)",
        "S(D)=(0.90, 0.40)  S(D+x)=(-0.06, 0.68)  change=1.00",
        "replace (3,4) by (-3,-4): S=(-0.30, -1.20)  change=2.00",
        "largest change / C  add/remove=1.0000  replace=2.0000",
        # One noisy step
        "T=1: mu=1  eps=4.38",
        "T=100: mu=10  eps=91.82",
        "T=10000: mu=100  eps=5425.51",
        "full batch, sigma=10.8, T=100: eps=4.005",
        # Amplification by subsampling
        "q=0.01 eps=0.1: eps'=0.0011  q*eps=0.0010",
        "q=0.01 eps=1: eps'=0.0170  q*eps=0.0100",
        "q=0.01 eps=5: eps'=0.9059  q*eps=0.0500",
        "q=0.01 eps=10: eps'=5.3993  q*eps=0.1000",
        "1 + q(e - 1) = 1.017183",
        "largest log ratio remove=0.017037 add=0.006341  formula=0.017037",
        # RDP of the subsampled Gaussian
        "alpha=2 terms: 0.980100  0.019800  0.000272  A_2=1.00017183",
        "eps(2)=1.7181e-04",
        "E[r^2]: formula=2.7183  Monte Carlo=2.7084",
        "alpha=2: exact=1.7181e-04 small-q=1.7183e-04 ratio=1.00",
        "alpha=4: exact=3.6315e-04 small-q=3.4366e-04 ratio=1.06",
        "alpha=8: exact=8.9364e-04 small-q=6.8731e-04 ratio=1.30",
        "alpha=16: exact=3.0879e+00 small-q=1.3746e-03 ratio=2246.32",
        "alpha=32: exact=1.1246e+01 small-q=2.7493e-03 ratio=4090.67",
        "alpha=2: binomial=1.718134e-04 integral=1.718134e-04 add direction=1.6022e-04",
        "alpha=8: binomial=8.936439e-04 integral=8.936439e-04 add direction=5.7239e-04",
        "alpha=32: binomial=1.124628e+01 integral=1.124628e+01 add direction=1.6923e-03",
        "sigma=2, alpha=8: exact/small-q=1.019",
        "sigma=4, alpha=8: exact/small-q=1.004",
        # Accounting
        "epochs=1 T=100: no sampling=91.8  Mironov=1.61 (alpha=8.9)  improved=1.21 (alpha=8.8)",
        "epochs=5 T=500: no sampling=344.5  Mironov=2.07 (alpha=8.3)  improved=1.65 (alpha=8.2)",
        "epochs=10 T=1000: no sampling=633.9  Mironov=2.54 (alpha=7.9)  improved=2.10 (alpha=7.8)",
        "epochs=20 T=2000: no sampling=1189.8  Mironov=3.34 (alpha=7.2)  improved=2.87 (alpha=6.9)",
        "epochs=50 T=5000: no sampling=2800.6  Mironov=5.18 (alpha=5.5)  improved=4.59 (alpha=5.2)",
        "epochs=100 T=10000: no sampling=5425.5  Mironov=7.43 (alpha=4.3)  improved=6.71 (alpha=4.1)",
        "integer orders only: Mironov=7.4692 (alpha=4)",
        "at alpha=4.3: eps(alpha)=3.9400e-04  T*eps(alpha)=3.9400  ln(1/delta)/(alpha-1)=3.4888  sum=7.4287",
        "fixed-size batches (noise multiplier sigma/2): improved=47.42 against Poisson 6.71",
        "delta=0.001: Mironov eps=5.89",
        "delta=1e-07: Mironov eps=8.71",
        # Recipe
        "target=1: sigma Mironov=4.974 (eps=1.0000)  improved=4.126",
        "target=2: sigma Mironov=2.617 (eps=2.0000)  improved=2.278",
        "target=4: sigma Mironov=1.484 (eps=4.0000)  improved=1.358",
        "target=8: sigma Mironov=0.962 (eps=8.0000)  improved=0.917",
    ):
        assert expected in out, expected
    assert {f.name for f in tmp_path.iterdir()} == {
        "sgd_clipping.svg",
        "sgd_subsampling_ratio.svg",
        "sgd_rdp_curve.svg",
        "sgd_eps_epochs.svg",
        "sgd_conversion.svg",
    }
