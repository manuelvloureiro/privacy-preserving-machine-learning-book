"""code/part05-dp-ml/public_data.py runs, writes its figures, and reproduces the numbers in book-full/part05-dp-ml/06-public-data.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part05-dp-ml"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_public_data_matches_chapter(tmp_path: Path) -> None:
    out = run(CODE / "public_data.py", tmp_path)
    for expected in (
        # The public task and the pretrained model
        "public records=50000  public teacher agrees with private labels on 0.9004",
        "pretrained: public training accuracy 0.9861  private report accuracy 0.9035",
        "non-private from scratch, 1000 steps: report accuracy 0.9606",
        "non-private fine-tuning of all weights, 1000 steps: report accuracy 0.9704",
        "non-private last layer (exact logistic regression): report accuracy 0.9536",
        # Why dimension costs
        "lora: k=525",
        "lora rank 1: k=180",
        "lora rank 16: k=1905",
        "feature norm: largest 7.035  bound sqrt(65)=8.062",
        "eps=0.5 sigma=7.72: noise length k=3329: 2.227  k=525: 0.884  k=65: 0.311",
        "eps=1 sigma=4.13: noise length k=3329: 1.191  k=525: 0.473  k=65: 0.166",
        "eps=2 sigma=2.28: noise length k=3329: 0.658  k=525: 0.261  k=65: 0.092",
        "eps=8 sigma=0.917: noise length k=3329: 0.265  k=525: 0.105  k=65: 0.037",
        "signal at start, scratch: length of the clipped mean 0.0618",
        "signal at start, full: length of the clipped mean 0.0130  eps=1 noise/signal 92.0  break-even steps 8455",
        "signal at start, last: length of the clipped mean 0.0250  eps=1 noise/signal 6.7  break-even steps 44",
        "signal at start, lora: length of the clipped mean 0.0207  eps=1 noise/signal 22.9  break-even steps 523",
        "all weights: largest eigenvalue 0.723  trace 16.310",
        "last layer: largest eigenvalue 0.142  smallest 2.96e-04  trace 0.681",
        "eta=0.1, T=10000: d_eff all weights 451  last layer 60.2  eigenvalues above 1/(2 eta T): 406",
        "eps=1, eta=0.1: predicted noise cost last layer 0.00064",
        "eps=1 (s=0.0206), eta=0.1: predicted noise cost all weights 0.0048  stationary bound p*eta*s^2/4=0.0355",
        # The public subspace
        "private clipped mean: length 0.0130",
        "k=1: mean captured 0.596  random 0.001",
        "k=4: mean captured 0.623  random 0.001",
        "k=16: mean captured 0.655  random 0.004",
        "k=64: mean captured 0.732  random 0.022",
        "k=256: mean captured 0.887  random 0.080",
        "eps=1: break-even steps all weights 8455  k=1: 4  k=4: 16  k=16: 62  k=64: 222  k=256: 733",
        # Low-rank adapters
        "256 examples: largest relative difference ghost vs explicit = 3.3e-16",
        # The running example
        "scratch: lr=0.25: 0.9318  lr=0.5: 0.9303  lr=1: 0.8582  -> lr=0.25",
        "full: lr=0.03: 0.9559  lr=0.1: 0.9567  lr=0.3: 0.9433  -> lr=0.1",
        "last: lr=0.1: 0.9479  lr=0.3: 0.9499  lr=1: 0.9460  -> lr=0.3",
        "lora: lr=0.03: 0.9556  lr=0.1: 0.9619  lr=0.3: 0.9472  -> lr=0.1",
        "eps=0.5: scratch 0.8589 +- 0.0075  full 0.9395 +- 0.0006  last 0.9478 +- 0.0010  lora 0.9464 +- 0.0005",
        "eps=1: scratch 0.9246 +- 0.0050  full 0.9599 +- 0.0013  last 0.9497 +- 0.0000  lora 0.9616 +- 0.0001",
        "eps=2: scratch 0.9518 +- 0.0010  full 0.9663 +- 0.0001  last 0.9519 +- 0.0005  lora 0.9653 +- 0.0002",
        "eps=8: scratch 0.9611 +- 0.0012  full 0.9677 +- 0.0001  last 0.9514 +- 0.0005  lora 0.9655 +- 0.0001",
        "rank 1: k=180  report accuracy 0.9590 +- 0.0038",
        "rank 4: k=525  report accuracy 0.9624 +- 0.0023",
        "rank 16: k=1905  report accuracy 0.9586 +- 0.0031",
    ):
        assert expected in out, expected
    for name in ("pub_accuracy.svg", "pub_noise_signal.svg", "pub_subspace.svg"):
        assert (tmp_path / name).exists()
