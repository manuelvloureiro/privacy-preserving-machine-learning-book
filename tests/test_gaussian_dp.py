"""code/part04-privacy-accounting/gaussian_dp.py runs, writes its figures, and reproduces the numbers in book-full/part04-privacy-accounting/04-gaussian-dp.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part04-privacy-accounting"


def test_gaussian_dp_matches_section(tmp_path: Path) -> None:
    out = run(CODE / "gaussian_dp.py", tmp_path)
    for expected in (
        # Section 30.3
        "[f_eps,delta eps=1 delta=0] f(0)=1.0000  f(0.1)=0.7282  f(0.2)=0.4563  f(0.5)=0.1839  kinks at a=0.2689",
        "[f_eps,delta eps=1 delta=0.1] f(0)=0.9000  f(0.1)=0.6282  f(0.2)=0.3563  f(0.5)=0.1472  kinks at a=0.2420",
        "p=(0.7311, 0.2689)  q=(0.2689, 0.7311)  corners=[(0.0, 1.0), (0.2689, 0.2689), (1.0, 0.0)]",
        "[eps=1 at a=0.01] f_1,0=0.9728  Laplace=0.9728  G_1=0.9076",
        "[eps=1 at a=0.1] f_1,0=0.7282  Laplace=0.7282  G_1=0.6109",
        "[eps=1 at a=0.25] f_1,0=0.3204  Laplace=0.3679  G_1=0.3724",
        # Section 30.4
        "[G_1(0.05)] 0.7405  power=0.2595",
        "sigma=1 -> mu=1.0000  sigma=4 -> mu=0.2500  pair 0.5 sensitivities apart, sigma=1 -> mu=0.5000",
        "[simulated G_1 at a=0.05] a=0.0502  b=0.7408  exact b=0.7405",
        # Section 30.5
        "[mu=1 eps=0] delta formula=0.382925  hockey-stick integral=0.382925",
        "[mu=1 eps=1] delta formula=0.126937  hockey-stick integral=0.126937",
        "[mu=1 eps=2] delta formula=0.020924  hockey-stick integral=0.020924",
        "[mu=0.5 eps=1] delta formula=0.006830  hockey-stick integral=0.006830",
        "[mu=2 eps=1] delta formula=0.509862  hockey-stick integral=0.509862",
        "on a grid of 20,001 points=0.126937 at a=0.0668",
        "threshold t=1.5000  a*=Phi(-1.5)=0.066807  1-b*=Phi(-0.5)=0.308538  e*a*=0.181601  delta=1-b*-e*a*=0.126937",
        "mu=0.5: eps=1.9931",
        "mu=1: eps=4.3772",
        "mu=2: eps=9.9973",
        "[one line vs the curve] f_(4.3772,1e-5)(0.1)=0.0113  G_1(0.1)=0.6109",
        "[delta at eps=0] mu=1: 0.3829 = TV = 2Phi(1/2)-1 = 0.3829",
        "mu=1: delta=1e-4 -> 3.8044, 1e-5 -> 4.3772, 1e-6 -> 4.8866  steps 0.57, 0.51",
        "mu=2: delta=1e-4 -> 8.8769, 1e-5 -> 9.9973, 1e-6 -> 10.9972  steps 1.12, 1.00",
        "max |sup f_eps,delta(eps) - G_1| on a grid = 4.9e-06",
        # Section 30.6
        "[compose] mu=(1,1)->1.4142  mu=(0.5,0.5,1)->1.2247  4 x mu=1 -> 2.0000  100 x sigma=10 -> 1.0000",
        "a=0.0497  b=0.5913  G_sqrt2(0.05)=0.5912  G_1(0.05)=0.7405  G_2(0.05)=0.3612",
        "p=0.5344,0.1966,0.1966,0.0723  q=0.0723,0.1966,0.1966,0.5344  q/p=0.1353,1.0000,1.0000,7.3891",
        "[(0.0, 1.0), (0.0723, 0.4656), (0.2689, 0.2689), (0.4656, 0.0723), (1.0, 0.0)]  tanh(1/2)=0.4621",
        "exact via G_2: eps=9.9973  per-step eps(delta/4)=4.6902  basic composition of points: eps=18.7607",
        # Section 30.7
        "[k=1 eps0=1.0000] RR: exact delta(1)=0.0000  Thm3.4 mu=1.0422 gamma=0.7663 -> delta(1)=0.1420",
        "Laplace: delta(1) in [0.0000, 0.0000]  Thm3.4 mu=0.9075 gamma=0.7348 -> delta(1)=0.0956 | limit G_1: 0.1269",
        "[k=10 eps0=0.3162] RR: exact delta(1)=0.1323  Thm3.4 mu=1.0042 gamma=0.1837 -> delta(1)=0.1284",
        "Laplace: delta(1) in [0.1112, 0.1114]  Thm3.4 mu=0.9569 gamma=0.1889 -> delta(1)=0.1120",
        "[k=100 eps0=0.1000] RR: exact delta(1)=0.1257  Thm3.4 mu=1.0004 gamma=0.0562 -> delta(1)=0.1271",
        "Laplace: delta(1) in [0.1211, 0.1214]  Thm3.4 mu=0.9844 gamma=0.0569 -> delta(1)=0.1215",
        "[k=1000 eps0=0.0316] RR: exact delta(1)=0.1269  Thm3.4 mu=1.0000 gamma=0.0177 -> delta(1)=0.1270",
        "Laplace: delta(1) in [0.1250, 0.1252]  Thm3.4 mu=0.9948 gamma=0.0178 -> delta(1)=0.1251",
        "[k=10 eps at delta=1e-5] RR exact=3.1599  GDP(Thm3.4 mu)=4.3983  GDP(mu=1)=4.3772  basic composition k*eps0=3.1623",
        "[k=100 eps at delta=1e-5] RR exact=4.3068  GDP(Thm3.4 mu)=4.3793  GDP(mu=1)=4.3772  basic composition k*eps0=10.0000",
        "[k=1000 eps at delta=1e-5] RR exact=4.3683  GDP(Thm3.4 mu)=4.3774  GDP(mu=1)=4.3772  basic composition k*eps0=31.6228",
        # Section 30.8
        "[e^(1/sigma^2)-1] sigma=1: 1.7183  sqrt: 1.3108  sigma=0.8: 3.7707  sigma=1.5: 0.5596",
        "[DP-SGD q=0.01 sigma=1 k=100] mu=0.1311  eps(1e-5)=0.4575  q sqrt(k)=0.10  DRS Cor 5.4 mu=0.1710",
        "[DP-SGD q=0.01 sigma=1 k=1000] mu=0.4145  eps(1e-5)=1.6177  q sqrt(k)=0.32  DRS Cor 5.4 mu=0.5408",
        "[DP-SGD q=0.01 sigma=1 k=10000] mu=1.3108  eps(1e-5)=6.0071  q sqrt(k)=1.00  DRS Cor 5.4 mu=1.7101",
        "full-batch mu=sqrt(k)/sigma=100.0",
        "[numerical accountant k=100] eps(1e-5) in [0.7175, 0.7185]  CLT eps=0.4575  CLT below the lower value: True",
        "[numerical accountant k=1000] eps(1e-5) in [1.8232, 1.8332]  CLT eps=1.6177  CLT below the lower value: True",
        "[numerical accountant k=10000] eps(1e-5) in [6.1377, 6.2377]  CLT eps=6.0071  CLT below the lower value: True",
        "[one step, P=mixture] mean=8.3812e-05  var=1.7163e-04  heuristic q^2(e-1)/2=8.5914e-05, q^2(e-1)=1.7183e-04  "
        "Thm3.4 mu at k=10000=1.2795 -> eps(1e-5)=5.8379",
        "[one step, P=N(0,1)] mean=8.1890e-05  var=1.6009e-04",
        "Thm3.4 mu at k=10000=1.2944 -> eps(1e-5)=5.9184",
        "CLT/lower=0.638  width=0.0010 = k h=0.0010",
        "CLT/lower=0.887  width=0.0100 = k h=0.0100",
        "CLT/lower=0.979  width=0.1000 = k h=0.1000",
        "CLT/lower=0.638",
        "CLT/lower=0.887",
        "CLT/lower=0.979",
        # Section 30.9
        "[mu=1 a=Phi(-1/2)=0.3085] G_1=0.3085  f_RDP on the grid of orders=0.2201  "
        "f_RDP with 5010 orders in [1.01, 3000]=0.2219",
        "[mu=1 diagonal point] G_1: a=0.3085 advantage=0.3829  f_RDP: a=0.2585 advantage=0.4830  delta_RDP(0)=0.4830",
        "[mu=1 eps=1] GDP delta=0.1269  delta certified by the RDP curve=0.2149 at a=0.0838  ratio=1.69",
    ):
        assert expected in out, expected
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == sorted(
        f"gaussian_dp_{n}.svg" for n in ("eps_delta", "duality", "composition", "clt", "subsampled", "rdp_lossy")
    )
