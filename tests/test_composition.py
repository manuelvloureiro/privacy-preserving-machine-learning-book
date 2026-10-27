"""code/part03-dp-foundations/composition.py runs, writes its figures, and reproduces the numbers in book-full/part03-dp-foundations/04-composition.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part03-dp-foundations"


def test_composition_matches_section(tmp_path: Path) -> None:
    out = run(CODE / "composition.py", tmp_path)
    for expected in (
        # Toolkit
        "RR keep prob=0.5250  mean loss eps*tanh(eps/2)=0.004996  DRV bound eps(e^eps-1)=0.010517",
        "eps=1.0000: tanh form=0.4621  DRV=1.7183  eps^2/2=0.5000  Laplace mean loss=0.3679",
        "3-output pair: max loss=0.9163",
        "K(.|0)=[0.65, 0.3, 0.05]  K(.|1)=[0.05, 0.3, 0.65]  rebuilt P=[0.5, 0.3, 0.2]  Q=[0.2, 0.3, 0.5]",
        "3-output pair: KL=0.2749  RR_ln3 KL=0.5493  delta(0)=0.3000  RR delta(0)=0.5000",
        # The loss of 100 rounds of randomized response
        "mean=0.4996  sd=0.9988  basic bound k*eps=10.0  (k*eps - mean)/sd=9.5",
        "P[L = k*eps] = a^k = 1.033e-28",
        "mean=47.5  sd=4.99  L > 4 iff J < 30  optimal eps' is 3.8 sd above the mean  e^(k*eps)=22026",
        # Basic, advanced, optimal
        "basic=10.0000  advanced DRV=5.8502  advanced tanh=5.2981  optimal=4.3068",
        "Azuma term eps*sqrt(2k ln(1/delta'))=4.7985  DRV mean term=1.0517  tanh mean term=0.4996",
        "exact delta at the advanced eps: 1.956e-09  at the tanh eps: 6.249e-08",
        "crossover: advanced (DRV) beats basic from k=29",
        "optimal below: tanh advanced 19%  DRV advanced 26%  basic 57%  tanh advanced above optimal by 23%",
        "k=10000 terms: Azuma=47.99  DRV mean=105.17  tanh mean=49.96  sqrt(2 ln(1/delta'))=4.7985  2 ln(1/delta')=23.03",
        "k=10: basic=1.000  advanced DRV=1.623  advanced tanh=1.567  optimal=0.994",
        "k=1000: basic=100.000  advanced DRV=25.691  advanced tanh=20.170  optimal=17.787",
        "k=10000: basic=1000.000  advanced DRV=153.156  advanced tanh=97.944  optimal=91.691",
        # Checks of the optimal formula
        "eps=0.5 delta=0 k=12: |formula - enumeration| <= 3.9e-16",
        "eps=0.3 delta=0.001 k=6: |formula - enumeration| <= 1.1e-16",
        "eps=1 delta=0.01 k=5: |formula - enumeration| <= 1.1e-16",
        "eps=ln 3, k=8, exact rationals: max error=2.2e-16  delta(0)=1759/2048",
        "naive=OverflowError  log-space=6.094e-04",
        "optimal eps=19.4228  advanced tanh=21.6225",
        "advanced tanh above optimal at eps=0.01 k=100000: 11%  log10 (1+e)^1000=570.3",
        "1-(1-delta)^k=5.000e-06  optimal=4.4469  advanced (delta'=5e-06)=5.9926",
        # Monte Carlo
        "RR exact=0.0201  RR Monte Carlo=0.0198  Laplace Monte Carlo=0.0189  adaptive Monte Carlo=0.0196",
        "mean total loss: RR=0.4996  Laplace=0.4837 (Monte Carlo 0.4838)  adaptive Monte Carlo=0.4971",
        # Gaussian
        "sigma=5 k=100: basic=93.178  advanced=89.843  optimal=76.223  exact=9.997",
        "optimal / exact = 7.6",
        "sigma=20 k=1000: basic=238.578  advanced=68.008  optimal=62.720  exact=7.511",
        # Laplace noise on dense counts
        "d=100: pure Laplace=141.4  Laplace + advanced=69.3  Laplace + optimal=52.3  Gaussian=37.3",
        "d=1000: pure Laplace=1414.2  Laplace + advanced=219.2  Laplace + optimal=166.6  Gaussian=118.0",
        "d=10000: pure Laplace=14142.1  Laplace + advanced=693.0  Laplace + optimal=527.5  Gaussian=373.1",
        "d=1000 per-count eps0: advanced=0.00645  optimal=0.00849",
        "optimal / exact = 8.4",
    ):
        assert expected in out, expected
    assert {f.name for f in tmp_path.iterdir()} == {
        "comp_tradeoff_region.svg",
        "comp_loss_distribution.svg",
        "comp_epsilon_vs_k.svg",
        "comp_delta_curves.svg",
        "comp_gaussian.svg",
    }
