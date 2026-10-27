"""code/part04-privacy-accounting/adaptive_composition.py runs, writes its figures, and reproduces the numbers in book-full/part04-privacy-accounting/08-adaptive-composition.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part04-privacy-accounting"


def test_adaptive_composition_matches_section(tmp_path: Path) -> None:
    out = run(CODE / "adaptive_composition.py", tmp_path)
    for expected in (
        # Section 34.3: the plug-in odometer fails under stop-when-ahead
        "per-step mean loss eps*tanh(eps/2)=0.004996",
        (
            "delta=0.05: P(cross by n): n=10: 0.0110  n=100: 0.0575  n=1000: 0.1004  n=10000: 0.1424  "
            "| max fixed-t P over t<=1000: 0.0110 at t=17  | cross by 10^4 / delta=2.85"
        ),
        (
            "delta=0.01: P(cross by n): n=10: 0.0000  n=100: 0.0090  n=1000: 0.0202  n=10000: 0.0318  "
            "| max fixed-t P over t<=1000: 0.0016 at t=21  | cross by 10^4 / delta=3.18"
        ),
        (
            "delta=1e-05: P(cross by n): n=10: 0.00e+00  n=100: 3.12e-06  n=1000: 1.80e-05  n=10000: 3.69e-05  "
            "| max fixed-t P over t<=1000: 8.18e-07 at t=758  | cross by 10^4 / delta=3.69"
        ),
        "bound at t=100, delta=0.01: 3.534",
        # Section 34.4: filters for pure DP
        "eps0=0.01 k=100: basic=1.000  advanced=0.536  Rogers filter=0.803  Whitehouse filter=0.531  Rogers/Whitehouse=1.51",
        (
            "eps0=0.01 k=1000: basic=10.000  advanced=1.763  Rogers filter=2.570  Whitehouse filter=1.712  "
            "Rogers/Whitehouse=1.50"
        ),
        (
            "eps0=0.01 k=10000: basic=100.000  advanced=6.262  Rogers filter=8.450  Whitehouse filter=5.757  "
            "Rogers/Whitehouse=1.47"
        ),
        "eps0=0.1 k=100: basic=10.000  advanced=6.308  Rogers filter=8.472  Whitehouse filter=5.757  Rogers/Whitehouse=1.47",
        (
            "eps0=0.1 k=1000: basic=100.000  advanced=27.140  Rogers filter=30.244  Whitehouse filter=21.623  "
            "Rogers/Whitehouse=1.40"
        ),
        "rho*=0.017469  zcdp_to_dp(rho*)=1.0000  Whitehouse at sum eps^2 = 2 rho*: 1.0000",
        "steps allowed at eps_g=1, eps0=0.01: basic=100  advanced (fixed k)=337  Rogers=154  Whitehouse=349",
        "steps allowed at eps_g=1, eps0=0.001: basic=1000  advanced (fixed k)=33785  Rogers=15481  Whitehouse=34937",
        # Section 34.5: the Rényi filter
        "rho(ln 3)=0.8473  rho(ln 1.5)=0.1542  budget B=2 rho(ln 3)=1.6946",
        "y=(1,1): P=0.5625  Q=0.0625  L=+2.1972  spent=1.6946  N2=1.6531  P*N2=0.9298",
        "y=(1,0): P=0.1875  Q=0.1875  L=+0.0000  spent=1.6946  N2=0.1837  P*N2=0.0344",
        "y=(0,1): P=0.1500  Q=0.3000  L=-0.6931  spent=1.0014  N2=0.1837  P*N2=0.0276",
        "y=(0,0): P=0.1000  Q=0.4500  L=-1.5041  spent=1.0014  N2=0.0816  P*N2=0.0082",
        "E_P[N2]=1.0000  joint D_2=1.6766  max path spend=1.6946",
        "fixed-curve bound for the two-step run: 1.6946",
        "y=(1,1,1): P=0.3375  Q=0.0250  L=+2.6027  spent=1.8487  N3=2.1254  P*N3=0.7173",
        "y=(1,1,0): P=0.2250  Q=0.0375  L=+1.7918  spent=1.8487  N3=0.9446  P*N3=0.2125",
        "y=(1,0,1): P=0.1125  Q=0.0750  L=+0.4055  spent=1.8487  N3=0.2362  P*N3=0.0266",
        "y=(1,0,0): P=0.0750  Q=0.1125  L=-0.4055  spent=1.8487  N3=0.1050  P*N3=0.0079",
        "y=(0,1,1): P=0.1125  Q=0.0750  L=+0.4055  spent=1.8487  N3=0.2362  P*N3=0.0266",
        "y=(0,1,0): P=0.0375  Q=0.2250  L=-1.7918  spent=1.8487  N3=0.0262  P*N3=0.0010",
        "y=(0,0,1): P=0.0750  Q=0.1125  L=-0.4055  spent=1.8487  N3=0.1050  P*N3=0.0079",
        "y=(0,0,0): P=0.0250  Q=0.3375  L=-2.6027  spent=1.8487  N3=0.0117  P*N3=0.0003",
        "E_P[N3]=1.0000  joint D_2=1.8487  filter budget B=1.8487  fixed-curve bound=2.5419",
        "y1=1: L1=+1.0986  N1=1.2857  E_P[N3 | y1]=1.2857",
        "y1=0: L1=-1.0986  N1=0.1429  E_P[N3 | y1]=0.1429",
        "top_up=True: MC D_2=1.0004  max spend=1.0000  mean spend=1.0000  steps min=3 max=9",
        "top_up=False: MC D_2=0.9305  max spend=0.9900  mean spend=0.8998  steps min=2 max=8",
        (
            "per step sigma=10 0.0290, sigma=5 0.1160; sigma=10 throughout: 100 steps; after 60 steps left 1.160, "
            "then 10 steps at sigma=5"
        ),
        # Section 34.6: odometers
        "|orders|=154  ln(1/delta)=11.513  ln(|orders|/delta)=16.550",
        (
            "t=10: rho=0.050  fixed-time=1.568  Ville over orders=1.869  Lecuyer=1.916  Ville at alpha=5.8 only=2.689  "
            "Ville/fixed=1.19"
        ),
        (
            "t=100: rho=0.500  fixed-time=5.299  Ville over orders=6.253  Lecuyer=6.507  Ville at alpha=5.8 only=5.299  "
            "Ville/fixed=1.18"
        ),
        (
            "t=300: rho=1.500  fixed-time=9.812  Ville over orders=11.465  Lecuyer=11.892  Ville at alpha=5.8 only=11.099  "
            "Ville/fixed=1.17"
        ),
        (
            "t=1000: rho=5.000  fixed-time=20.175  Ville over orders=23.194  Lecuyer=24.633  Ville at alpha=5.8 only=31.399  "
            "Ville/fixed=1.15"
        ),
        "running example q=0.01 sigma=1 k=10000: fixed-time=7.43  Ville over orders=8.82  Lecuyer=9.08",
        "MC P(cross)=0.0461  Brownian limit=0.0500",
        "crossed with probability 0.0072",
        # Section 34.7: individual accounting
        (
            "worst case: rho=0.500  eps=5.299 at alpha*=5.8  budget B=eps(alpha*)=2.900  "
            "floor ln(1/delta)/(alpha*-1)=2.399"
        ),
        "hard records: 962 (9.6%)",
        "all: eps at alpha*: p10=2.56 median=3.25 p90=3.81  median optimized over orders=2.75",
        "typical: eps at alpha*: p10=2.55 median=3.17 p90=3.67  median optimized over orders=2.61",
        "hard: eps at alpha*: p10=4.23 median=5.02 p90=5.26  median optimized over orders=5.01",
        "records below half the worst-case eps: 16.3%",
        "step 100: participating all=100.0%  typical=100.0%  hard=100.0%",
        "step 200: participating all=90.4%  typical=100.0%  hard=0.0%",
        "step 300: participating all=90.4%  typical=100.0%  hard=0.1%",
        "hard records first excluded: median step 111  earliest 101",
        "typical records ever excluded: 0.0%  median final spend typical=0.778  of B=2.900",
        "per-step worst case alpha*/(2 sigma^2)=0.0290",
    ):
        assert expected in out, expected
    assert {f.name for f in tmp_path.iterdir()} == {
        "adaptive_composition_plugin_failure.svg",
        "adaptive_composition_filter_budgets.svg",
        "adaptive_composition_supermartingale.svg",
        "adaptive_composition_odometers.svg",
        "adaptive_composition_individual_hist.svg",
        "adaptive_composition_individual_spend.svg",
    }
