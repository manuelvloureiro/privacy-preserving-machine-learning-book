"""code/part03-dp-foundations/discrete_mechanisms.py runs, writes its figures, and reproduces the numbers in book-full/part03-dp-foundations/08-discrete-mechanisms.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part03-dp-foundations"


def test_discrete_mechanisms_matches_section(tmp_path: Path) -> None:
    out = run(CODE / "discrete_mechanisms.py", tmp_path)
    for expected in (
        # Floating-point Laplace and snapping
        "p=8: outputs of M(D)=1604  of M(D')=1409  mass of M(D) impossible under M(D')=0.4795  reverse=0.2070",
        "p=12: outputs of M(D)=25646  of M(D')=22477  mass of M(D) impossible under M(D')=0.4791  reverse=0.2033",
        "p=14: outputs of M(D)=102586  of M(D')=89859  mass of M(D) impossible under M(D')=0.4826  reverse=0.2053",
        "p=12, outputs with |y| < 0.5: mass=0.3933  impossible under M(D')=0.3173",
        "p=12, outputs in [0.25, 0.2525]: M(D) 9  M(D') 10  shared 6",
        "snapped, p=12, grid 1, clamp 10: outputs 21  mass outside=0.0000  max |log ratio|=1.0010",
        "snapped, p=14, grid 1, clamp 10: outputs 21  mass outside=0.0000  max |log ratio|=1.0002",
        # Algorithm 1
        "gamma=1/2: empirical=0.6024  exact=0.6065  draws per sample=1.662 (expected 1.649)",
        "gamma=1: empirical=0.3709  exact=0.3679  draws per sample=3.096 (expected 3.086)",
        "gamma=5/2: empirical=0.0869  exact=0.0821  draws per sample=3.976 (expected 3.941)",
        "gamma=4: empirical=0.0168  exact=0.0183  draws per sample=4.218 (expected 4.240)",
        "k=1: 0.2987/0.3000  k=2: 0.4618/0.4550  k=3: 0.1824/0.1878  k=4: 0.0473/0.0472  k=5: 0.0083/0.0086",
        "e^-0.7=0.4966",
        # Algorithm 2
        "var=1.8305 (exact 1.8413, continuous 2.0000)  TV to exact=0.0114 (noise floor 0.0074)  "
        "draws per sample=11.49  loop passes=1.462",
        "var=4.3834 (exact 4.3370, continuous 4.5000)  TV to exact=0.0119 (noise floor 0.0243)  draws per sample=12.08",
        "var=195.2985 (exact 199.8334, continuous 200.0000)  TV to exact=0.0339 (noise floor 0.0331)  "
        "draws per sample=10.25",
        "Pr[Z=0] at scale 1: 0.4621",
        # Discrete Gaussian: privacy
        "sigma=0.5, shift 1: D_2=4.0000/4.0000  D_2.5=4.9808/5.0000  D_5=10.0000/10.0000",
        "sigma=1, shift 1: D_2=1.0000/1.0000  D_2.5=1.2500/1.2500  D_5=2.5000/2.5000",
        "sigma=2, shift 1: D_2=0.2500/0.2500  D_2.5=0.3125/0.3125  D_5=0.6250/0.6250",
        "sigma=0.5: normalizer at c=0 1.271342  at c=0.5 1.235287  max over c at c=0.00",
        "sigma=1: normalizer at c=0 2.506628  at c=0.5 2.506628",
        "sigma=0.5, eps=1: delta discrete=6.0320e-01  continuous=5.0986e-01",
        "sigma=1, eps=1: delta discrete=1.4135e-01  continuous=1.2694e-01",
        "sigma=2, eps=1: delta discrete=7.2488e-03  continuous=6.8296e-03",
        # Discrete Gaussian: variance
        "sigma=0.25: Var/sigma^2 summation=0.010728  Poisson=0.010728",
        "sigma=0.5: Var/sigma^2 summation=0.860051  Poisson=0.860051",
        "sigma=1: Var/sigma^2 summation=1.000000  Poisson=1.000000  1 - Var/sigma^2=2.11e-07",
        # Algorithm 3
        "sigma^2=1/4: t=1  mean=0.0019  var=0.2189 (exact 0.2150)  TV to exact=0.0028 (noise floor 0.0045)  "
        "acceptance=0.5185  draws per sample=26.6",
        "sigma^2=1: t=2  mean=-0.0001  var=0.9977 (exact 1.0000)  TV to exact=0.0076 (noise floor 0.0042)  "
        "acceptance=0.5418  draws per sample=24.8",
        "sigma^2=4: t=3  mean=0.0018  var=4.0826 (exact 4.0000)  TV to exact=0.0154 (noise floor 0.0142)  "
        "acceptance=0.6629  draws per sample=18.9",
        "sigma^2=100: t=11  mean=-0.0711  var=99.4386 (exact 100.0000)  TV to exact=0.0243 (noise floor 0.0325)  "
        "acceptance=0.7532  draws per sample=15.6",
    ):
        assert expected in out
    assert {f.name for f in tmp_path.iterdir()} == {
        "discrete_float_support.svg",
        "discrete_bernoulli_exp.svg",
        "discrete_laplace.svg",
        "discrete_gaussian_sampler.svg",
        "discrete_gaussian_variance.svg",
        "discrete_gaussian_delta.svg",
    }
