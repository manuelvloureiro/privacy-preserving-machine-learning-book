"""code/part03-dp-foundations/mean_estimation.py runs, writes its figures, and reproduces the numbers in book-full/part03-dp-foundations/11-mean-estimation.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part03-dp-foundations"


def test_mean_estimation_matches_section(tmp_path: Path) -> None:
    out = run(CODE / "mean_estimation.py", tmp_path)
    for expected in (
        # Toolkit
        "tau=sqrt(2 ln(2n/beta))=4.604  simulated Pr[max|x_i - mu| > tau]=0.0039",
        # Bounded data
        "n=100: rmse=0.0520  non-private=0.0500",
        "n=1000: rmse=0.0159  non-private=0.0158",
        "n=10000: rmse=0.0050  non-private=0.0050",
        "alpha=0.05: n needed=108  non-private n=100",
        "alpha=0.01: n needed=2508  non-private n=2500  extra=8",
        # Naive and coarse then fine
        "bin width b=2tau=9.207  fine window 3b=27.62  sampling sd=0.0316",
        (
            "R=1e+01: naive noise sd=0.0413 rmse=0.0520 | bins m=5  coarse failure <= 3.6e-27  n needed for beta: 74 "
            "| fine noise sd=0.0521 rmse=0.0609"
        ),
        "R=1e+03: naive noise sd=2.8414 rmse=2.8416 | bins m=220  coarse failure <= 1.6e-25  n needed for beta: 135",
        (
            "R=1e+06: naive noise sd=2828.4401 rmse=2828.4401 | bins m=217223  coarse failure <= 1.6e-22  "
            "n needed for beta: 245"
        ),
        "coarse-then-fine beats naive from R = 13.81",
        "simulated R=1e+01: naive=0.0517  coarse-fine=0.0588",
        "simulated R=1e+03: naive=2.7615  coarse-fine=0.0606",
        "simulated R=1e+06: naive=2803.0839  coarse-fine=0.0596",
        # Heavy tails
        "x_m=1.1547  mean=1.7321",
        "n=1000 no clipping: tau=141.42  bias=-0.0000  sampling sd=0.0313  noise sd=0.4000  rmse=0.4012  simulated=0.3997",
        "n=1000 bound: tau=18.80  bias=-0.0018  sampling sd=0.0293  noise sd=0.0532  rmse=0.0607  simulated=0.0591",
        "n=1000 optimal: tau=5.36  bias=-0.0153  sampling sd=0.0249  noise sd=0.0152  rmse=0.0329  simulated=0.0325",
        "n=1000: worst-case bound on rmse=0.1380  non-private sd=0.0316",
        "n=100000 no clipping: tau=1414.21",
        "rmse=0.0401",
        "n=100000 bound: tau=188.03",
        "rmse=0.0062",
        "n=100000 optimal: tau=29.72",
        "rmse=0.0032",
        # Many dimensions
        "mu(eps=1, delta=1e-6)=0.2367",
        (
            "d=10: non-private=0.0100  naive noise=0.2695 rmse=0.2697  shrinking rounds=2 radii=1e+03,4.12,0.0725  "
            "noise=0.0032 rmse=0.0105  noise*n*eps/d=32.20"
        ),
        (
            "d=100: non-private=0.0316  naive noise=0.8580 rmse=0.8586  shrinking rounds=2 radii=1e+03,8.54,0.243  "
            "noise=0.0183 rmse=0.0366  noise*n*eps/d=18.34"
        ),
        (
            "d=1000: non-private=0.1000  naive noise=2.7711 rmse=2.7730  shrinking rounds=2 radii=1e+03,22.9,1.43  "
            "noise=0.1419 rmse=0.1736  noise*n*eps/d=14.19"
        ),
        "simulated d=100: rmse=0.0363 (formula 0.0366)  every ball held mu in 40/40",
        # Coarse step on one dataset
        "bins=24  non-empty counts=876,124  Pr[top bin]=1.000000000000  Pr[one empty bin]=2.8e-48",
    ):
        assert expected in out, expected
    assert {f.name for f in tmp_path.iterdir()} == {
        "mean_range.svg",
        "mean_coarse.svg",
        "mean_heavy.svg",
        "mean_dimension.svg",
    }
