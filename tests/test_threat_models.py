"""code/part02-attacks/threat_models.py runs, writes its figures, and reproduces the numbers in book-full/part02-attacks/01-threat-models.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part02-attacks"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_threat_models_matches_section(tmp_path: Path) -> None:
    out = run(CODE / "threat_models.py", tmp_path)
    for expected in (
        # Threat models: the centroid detector
        "white-box=0.2789  black-box, calibrated=0.2789  black-box, raw loss=0.0407  label-only=0.0400",
        "d=1: white-box=0.0326",
        "d=500: white-box=0.7407",
        # Advantage and the trade-off curve: the noisy count, rest known
        "TV=0.3829  simulated TPR=0.6901 FPR=0.3092 advantage=0.3809  win probability=0.6905 (1/2 + TV/2 = 0.6915)",
        "worst case at FPR 1e-3: TPR=0.0183 (simulated TPR=0.0184 at FPR=0.00109)",
        "precision at FPR 1e-3, prior 0.5: 0.948",
        "precision at FPR 1e-3, prior 0.01: 0.156",
        # Average case and worst case
        "TV=0.0779  normal approximation=0.0781  simulated advantage=0.0751",
        "average case at FPR 1e-3: TPR=0.0019 (simulated 0.0016)",
        "random target bit: advantage TV/2=0.0390  simulated=0.0398",
        "average-case TV=0.0796  sqrt(2/(pi m))=0.0798",
        "TPR=0.00098  FPR=0.00000  advantage=0.00098  (1/n = 0.00100)",
        # Privacy games: attribute and reconstruction
        "prior 0.5: baseline=0.5000  best accuracy=0.6915  E_gamma=0.3829  simulated=0.6925",
        "prior 0.1: baseline=0.9000  best accuracy=0.9013  E_gamma=0.0134  simulated=0.9028",
        "m=10: fraction recovered=0.6230  1/2 + 1/sqrt(2 pi m)=0.6262",
        "m=100: fraction recovered=0.5398  1/2 + 1/sqrt(2 pi m)=0.5399",
        "m=1000: fraction recovered=0.5126  1/2 + 1/sqrt(2 pi m)=0.5126",
        "m=10 simulated: 0.6236",
    ):
        assert expected in out
    assert {f.name for f in tmp_path.iterdir()} == {
        "threat_game.svg",
        "threat_access.svg",
        "threat_tradeoff.svg",
        "threat_average_worst.svg",
    }
