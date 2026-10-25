"""code/appendix-derivations/apx_mechanisms.py runs, writes its figures, and reproduces the numbers in book-full/appendix-derivations/10-mechanisms.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "appendix-derivations"


def test_appendix_mechanisms_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "apx_mechanisms.py", tmp_path)
    for expected in (
        # Four-point decomposition, the small example and the binned Gaussian pair.
        "E(P||Q)=0.2000  E(Q||P)=0.3000  delta=0.3000",
        "m_P=[0.4, 0.3, 0.1]  A=0.8000  m_Q=[0.2, 0.3, 0.2]  B=0.7000",
        "theta=0.2222  scale=1.0000",
        "pure_P=[0.3333, 0.2667, 0.1]  pure_Q=[0.2, 0.3, 0.2]",
        "Pbar=[0.4762, 0.381, 0.1429]  Qbar=[0.2857, 0.4286, 0.2857]",
        "R_D=[0.8889, 0.1111, 0.0]  R_D'=[0.0, 0.0, 1.0]",
        "K(.|0)=[0.6667, 0.3333, 0.0]  K(.|1)=[0.0952, 0.4762, 0.4286]",
        "P*K=[0.6, 0.3, 0.1]  Q*K=[0.2, 0.3, 0.5]",
        "E(P||Q)=0.2259  E(Q||P)=0.0844  delta=0.2259  A=0.7741  B=0.9156  theta=0.2243",
        "2000 random pairs, 3 outputs, eps=0.5] reproduction=2.2e-16  smallest entry=-1.4e-13",
        "2000 random pairs, 20 outputs, eps=2] reproduction=1.1e-16  smallest entry=-9.5e-15",
        # Approximate zCDP.
        "rho=eps^2/2=0.2402  largest D_a/(a rho)=0.4275; binned Gaussian: rho=0.5000  largest D_a/(a rho)=0.4306",
        # Stability histogram.
        "T=14.1224  release at count 1=1.000e-06  release at count T=0.50",
        "bound T+ln(k/beta)/eps=19.4207  exact worst quantile=18.0000 at count 18  simulated=18.0000",
        "N=1e+06: Laplace exact=16.7857  packing lower bound=6.3821",
        "N=1e+12: Laplace exact=30.6012  packing lower bound=13.2899",
        "beats Laplace from N=1.394e+07  and the pure-DP lower bound from N=2.114e+17",
        # Heavy tails.
        "a=10.0000  p=0.0100  variance=0.9900  n >= 172.8 (simple form 171.4)  clipped mean needs 7616",
        "k=3: a=3.1623  p=0.0316  moment=0.9091  n >= 55.6",
        "k=4: a=2.1544  p=0.0464  moment=0.8270  n >= 38.4",
        "bounded data in [0, 1]: n >= 18.6",
        # Sparse vector.
        "[Sparse vector c=1, eps1=eps2=0.5] m=1: 0.3958  m=10: 0.9940  m=40: 1.0000",
        "[Sparse vector c=4, eps1=eps2=0.5] m=1: 0.2670  m=10: 0.7602  m=40: 1.0000",
        "c=4 with noise Lap(2/eps2), m=40: 2.5000",
        # Gumbel top-2.
        "Pr[(0,1)] exact=0.1313  simulated=0.1313  largest difference=0.0003  TV=0.0017",
        # Alarm mechanism.
        "cap (e^eps+delta)/(e^eps+1)=0.6225  a_0 at K=10=2.272e-03  largest K=21",
        "K=22: a_0=0.9229  a_1=0.5598  largest slack=3.132e-01",
    ):
        assert expected in out, expected
    assert {f.name for f in tmp_path.iterdir()} == {
        "apx_mech_fourpoint_split.svg",
        "apx_mech_fourpoint_tradeoff.svg",
        "apx_mech_histogram_error.svg",
        "apx_mech_histogram_release.svg",
        "apx_mech_heavy_two_point.svg",
        "apx_mech_heavy_rates.svg",
        "apx_mech_svt_losses.svg",
        "apx_mech_gumbel_top2.svg",
        "apx_mech_gumbel_race.svg",
    }
