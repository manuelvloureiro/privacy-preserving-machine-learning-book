"""code/appendix-derivations/apx_tails.py runs, writes its figures, and reproduces the numbers in book-full/appendix-derivations/06-tails-tests-divergences.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "appendix-derivations"


def test_appendix_tails_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "apx_tails.py", tmp_path)
    for expected in (
        # Bernstein
        "series bound holds on [0,3): True",
        "k=2: E[Y^k]=0.0099000 bound=0.0099000  k=3: E[Y^k]=0.0097020 bound=0.0098010",
        "k=4: E[Y^k]=0.0096060 bound=0.0097030",
        "exact=3.29e-03  Hoeffding=8.19e-01  Bernstein=2.26e-02  Chernoff=2.00e-02",
        "root=0.00875  simple=0.00968  Hoeffding=0.03870",
        # Cramér and Chernoff–Stein
        "kl=0.04440  lambda=0.8109  psi=0.11778",
        "[Cramer n=100] exact rate=0.06225  tilting bound=0.23163",
        "[Cramer n=1000] exact rate=0.04728  tilting bound=0.09152",
        "[Cramer n=10000] exact rate=0.04481  tilting bound=0.05800",
        "KL=0.4564  V=0.8348",
        "[Stein n=10] beta=7.702e-02  rate=0.2564  proof bounds=[-0.8357, 0.9501]",
        "[Stein n=50] beta=1.160e-07  rate=0.3194  proof bounds=[-0.1214, 0.6588]",
        "[Stein n=200] beta=3.959e-33  rate=0.3730  proof bounds=[0.1675, 0.5539]",
        # Clopper–Pearson
        "u(0)=0.003682  u(1)=0.005559  u(10)=0.018313",
        "k=10 n=10000: [0.00048, 0.00184]  k=1 n=1000 one-sided 0.05: 0.0047  k=0 n=3000: 9.9808e-04",
        "Clopper-Pearson min=0.9504  mean=0.9645  Wald min=0.0488  Wald at p=0.01: 0.6334  CP at p=0.01: 0.9816",
        # Rényi to KL
        "[Renyi die] KL=0.2939",
        "[Renyi die lambda=-0.01] D=0.2907  sandwich=0.2902",
        "[Renyi die lambda=+0.01] D=0.2971  sandwich=0.2976",
        "h_lambda(x) non-decreasing in lambda: True",
        "[Renyi slopes x=3] -0.5: 0.8453  -0.1: 1.0404  +0: 1.0986  +0.1: 1.1612  +0.5: 1.4641",
        "[Renyi slopes x=0.6] -0.5: -0.5820  -0.1: -0.5241  +0: -0.5108  +0.1: -0.4980  +0.5: -0.4508",
        "H=12.4278  H_0.999=12.4310  H_1.001=12.4246  H_2=6.6322  H_100=3.3555  H_inf=3.3219",
        # Perspective
        "joint convexity violations=0  subadditivity violations=0  quasi-convexity violations=0",
        "D_2(P||Q)=0.5878  test coin=0.3920",
        "moment=1.0115  bound=1.1718",
        "f_RDP(0.1)=0.5068  power=0.4932",
        # Noisy count
        "[Noisy count Bernoulli+N(0,0.2^2)] modes=2",
        "crossing=50.5000  advantage=0.0779",
        "[Noisy count m=100 sigma=1.0] max second difference of ln f=-3.8e-06",
        "[Noisy count m=100 sigma=0.5] max second difference of ln f=+5.5e-05",
    ):
        assert expected in out, expected
    assert {f.name for f in tmp_path.iterdir()} == {
        "tail_bernstein_mgf.svg",
        "tail_bernstein_tail.svg",
        "tail_cramer.svg",
        "tail_stein.svg",
        "tail_clopper_cdf.svg",
        "tail_clopper_coverage.svg",
        "tail_renyi_sandwich.svg",
        "tail_perspective.svg",
        "tail_noisy_count.svg",
    }
