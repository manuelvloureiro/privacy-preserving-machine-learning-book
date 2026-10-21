"""code/appendix-derivations/apx_cryptography.py runs, writes its figures, and reproduces the numbers in
book-full/appendix-derivations/09-cryptography.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "appendix-derivations"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_appendix_cryptography_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "apx_cryptography.py", tmp_path)
    for expected in (
        "[GGM H_0] Pr[A=1] simulated=0.0000  exact=0.0000",
        "[GGM H_1] Pr[A=1] simulated=0.4994  exact=0.4980",
        "[GGM H_2] Pr[A=1] simulated=0.4985  exact=0.4980",
        "[GGM H_3] Pr[A=1] simulated=0.9962  exact=0.9961",
        "[GGM B_1] on PRG pairs=0.0000 (H_0=0.0000)  on uniform pairs=0.5005",
        "[GGM B_2] on PRG pairs=0.4979 (H_1=0.4994)  on uniform pairs=0.4993",
        "[GGM B_3] on PRG pairs=0.4983 (H_2=0.4985)  on uniform pairs=0.9964",
        "Pr[B=1] on G(U)=0.5535  on uniform=0.6641  advantage=0.1105  predicted Adv(A)/(nq)=0.1107  nq*advantage=0.995",
        "g=3] residuals over r=0..4: [1, 1, 1, 1, 1]",
        "g=1] residuals over r=0..4: [1, 4, 2, 0, 3]",
        "correct guess=0.760 (exact 0.757)  wrong guess=0.071 (exact 0.072)",
        "secret=[6, 27, 27, 28]  recovered=[6, 27, 27, 28]  Pr[accept LWE]=0.7567  eps=0.0684  runs per guess N=19,230",
        "oracle calls=7,480,470",
        "threshold eps/2=0.0342  smallest correct gap=0.0708  largest wrong gap=0.0166",
        "10/10 random secrets recovered exactly",
    ):
        assert expected in out
    assert {f.name for f in tmp_path.iterdir()} == {
        "apx_crypto_ggm_hybrids.svg",
        "apx_crypto_ggm_check.svg",
        "apx_crypto_lwe_residuals.svg",
        "apx_crypto_lwe_search.svg",
    }
