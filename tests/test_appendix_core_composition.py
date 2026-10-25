"""code/appendix-derivations/apx_core_composition.py runs, writes its figures, and reproduces the numbers in
book-full/appendix-derivations/02-composition.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "appendix-derivations"


def test_core_composition_matches_appendix(tmp_path: Path) -> None:
    out = run(CODE / "apx_core_composition.py", tmp_path)
    for expected in (
        # Basic composition
        (
            "total eps=1.000  simulated max loss=1.0000  Pr[loss = k eps] simulated=0.000997 exact=0.000977  "
            "mean=0.0484 (exact 0.0484)"
        ),
        "(1,1): bit 1=0.5625  bit 0=0.0625  loss=+2.1972",
        "(1,0): bit 1=0.1875  bit 0=0.1875  loss=+0.0000",
        "(0,1): bit 1=0.1667  bit 0=0.2500  loss=-0.4055",
        "(0,0): bit 1=0.0833  bit 0=0.5000  loss=-1.7918",
        "delta forward=0.0476  backward=0.0000  Pr[L > 1]=0.50",
        # Advanced composition
        "eps=0.1: bound eps(e^eps-1)=0.0105  randomized response=0.0050",
        "eps=1: bound eps(e^eps-1)=1.7183  randomized response=0.4621",
        "basic=10.0000  azuma=4.7985  mean term=1.0517  advanced=5.8502  tanh form=5.2981  optimal=4.3068",
        "mean loss=0.4996  sd=0.9988  Pr[L > 5.8502]=1.022e-08  exact delta=1.956e-09",
        "= 28.76",
        "max=18.4 (basic 100, advanced 25.6914)",
        # Gaussian releases
        (
            "k=16 sigma=4: loss mean=0.4971 (k mu^2/2=0.5000)  var=1.0001 (k mu^2=1.0000)  "
            "delta(1) simulated=0.1264 formula=0.1269"
        ),
        "one release at (1, 1e-5): sigma_1=3.7306",
        (
            "k=10: sigma_k=sqrt(k) sigma_1=11.7973  exact eps check=1.0000  basic split (0.1, 1e-06) each: "
            "sigma=36.3047  ratio=3.08"
        ),
        (
            "k=100: sigma_k=sqrt(k) sigma_1=37.3063  exact eps check=1.0000  basic split (0.01, 1e-07) each: "
            "sigma=362.0183  ratio=9.70"
        ),
        # Rényi DP
        "alpha=2 sigma=1: formula=1.0000  integral=1.0000  moment e^((a-1)D)=2.718",
        "alpha=10 sigma=1: formula=5.0000  integral=5.0000",
        "alpha=2 sigma=4: formula=0.0625  integral=0.0625",
        "k=1000 sigma=20: RDP eps=8.8376 at alpha=4  closed form=8.8371  exact=7.5113",
        # Comparison
        "k=1: basic=0.160  advanced=0.864 (s=0.56)  RDP=0.251 (alpha=128)  exact=0.160  RDP/exact=1.57",
        "k=10: basic=1.892  advanced=3.502 (s=0.56)  RDP=0.771 (alpha=31)  exact=0.561  RDP/exact=1.37",
        "k=100: basic=21.510  advanced=16.462 (s=0.56)  RDP=2.524 (alpha=10.6)  exact=1.993  RDP/exact=1.27",
        "k=1000: basic=238.578  advanced=105.863 (s=0.56)  RDP=8.838 (alpha=4)  exact=7.511  RDP/exact=1.18",
        "k=10000: basic=2601.968  advanced=938.234 (s=0.56)  RDP=36.513 (alpha=2)  exact=33.104  RDP/exact=1.10",
        "k=1000: advanced, tanh form=68.008",
    ):
        assert expected in out, expected
    assert {f.name for f in tmp_path.iterdir()} == {
        "ccomp_product.svg",
        "ccomp_walk.svg",
        "ccomp_gauss_sum.svg",
        "ccomp_rdp_square.svg",
        "ccomp_compare.svg",
    }
