"""code/part05-dp-ml/solutions_public_data.py runs and reproduces the numbers in book-exercises/part05-dp-ml/06-public-data.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part05-dp-ml"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_solutions_public_data_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_public_data.py", tmp_path)
    for expected in (
        "768x768, rank 1: adapter 1,536  full 589,824  share 0.0026  noise length ratio sqrt(full/adapter) 19.60",
        "768x768, rank 8: adapter 12,288  full 589,824  share 0.0208  noise length ratio sqrt(full/adapter) 6.93",
        "768x768, rank 64: adapter 98,304  full 589,824  share 0.1667  noise length ratio sqrt(full/adapter) 2.45",
        "51x64, rank 4: adapter 460  full 3,264  share 0.1409  noise length ratio sqrt(full/adapter) 2.66",
        "h=1e-05, eta=0.1, T=10000: share of stationary variance reached 0.0198",
        "h=0.0001, eta=0.1, T=10000: share of stationary variance reached 0.1813",
        "h=0.001, eta=0.1, T=10000: share of stationary variance reached 0.8647",
        "h=0.1, eta=0.1, T=10000: share of stationary variance reached 1.0000",
        "h_j=0.7 j^-1, p=3329: trace 6.081  d_eff 2079  bound 2 eta T trace 12163",
        "h_j=0.7 j^-1.5, p=3329: trace 1.804  d_eff 286  bound 2 eta T trace 3609",
        "h_j=0.7 j^-2, p=3329: trace 1.151  d_eff 65  bound 2 eta T trace 2302",
        "Lipschitz bound sqrt(65)=8.062  largest feature norm 7.035",
        "clip=inf, T=100: per-step noise sd 0.0602  noise length 0.4850  report accuracy 0.9383",
        "clip=inf, T=400: per-step noise sd 0.1203  noise length 0.9700  report accuracy 0.9361",
        "clip=1, T=100: per-step noise sd 0.0075  noise length 0.0602  report accuracy 0.9410",
        "clip=1, T=400: per-step noise sd 0.0149  noise length 0.1203  report accuracy 0.9454",
        "eta/B=0.001: rank of the update 4  relative size of the part outside span(V) 1.6e-02",
        "eta/B=0.003: rank of the update 4  relative size of the part outside span(V) 4.1e-02",
        "eta/B=1: rank of the update 4  relative size of the part outside span(V) 9.7e-01",
        "shift=0, rotation=0.00: teacher agreement 1.0000  pretrained 0.9784  private last layer eps=1 0.9749",
        "shift=0.15, rotation=0.30: teacher agreement 0.9004  pretrained 0.9035  private last layer eps=1 0.9507",
        "shift=0.5, rotation=1.00: teacher agreement 0.6378  pretrained 0.6334  private last layer eps=1 0.9054",
        "shift=1, rotation=1.57: teacher agreement 0.4866  pretrained 0.4805  private last layer eps=1 0.8586",
        "eps=1, delta=1e-5, FPR=0.001: TPR <= 0.00273",
        "eps=1, delta=1e-5, FPR=0.01: TPR <= 0.02719",
        "two copies in the private data: eps=2  delta=5.44e-05",
    ):
        assert expected in out, expected
