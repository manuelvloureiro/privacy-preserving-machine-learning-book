"""code/part01-prerequisites/crypto_basics.py runs, writes its figures, and reproduces the numbers in book-full/part01-prerequisites/08-crypto-basics.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "code" / "part01-prerequisites"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_crypto_basics_matches_section(tmp_path: Path) -> None:
    out = run(CODE / "crypto_basics.py", tmp_path)
    for expected in (
        "2^-lam=2^-128.00 (2.9e-39)  lam^-log lam=2^-49.00 (1.8e-15)  "
        "2^-sqrt lam=2^-11.31 (3.9e-04)  lam^-10=2^-70.00 (8.5e-22)",
        "first below lam^-10 at lambda=20520",
        "= 2^84.7 keys, success=9.3e-14; exhausting 2^128 takes 1.08e+13 years",
        "[Birthday q=23 N=365] exact=0.507  union=0.693  1-exp=0.500",
        "[Birthday q=1000 N=1048576] exact=0.379  union=0.476  1-exp=0.379",
        "[Birthday q=100000 N=4294967296] exact=0.688  union=1.164  1-exp=0.688",
        "exact=0.3927  empirical=0.3920  union=0.4980  half at q=301",
        "m=1] majority empirical=0.008  best (TV)=0.010  hybrid bound=0.010",
        "m=100] majority empirical=0.082  best (TV)=0.080  hybrid bound=1.000",
        "m=10000] majority empirical=0.481  best (TV)=0.683  hybrid bound=1.000",
        "[IND-CPA deterministic] advantage=1.000",
        "empirical=0.0605  exact=0.0607  bound q/|R|=0.0625",
        "sees each pair exactly once: True",
        "Z_11 empirical TV=0.0045 (exact 0)  integers R=16 empirical TV=0.4423 (exact 0.4375)",
        "recovers 16/16 coordinates without noise, 0/16 with e in {-1,0,1}",
        "b1=3030 dec=1  b2=1969 dec=1  sum b=903 <a,s>=898 dec=0",
        "worst-case safe for k < 32",
        "[Regev k=1] empirical failure=0.0000  normal approx=0.0000",
        "[Regev k=31] empirical failure=0.0000  normal approx=0.0000",
        "[Regev k=256] empirical failure=0.0007  normal approx=0.0006",
        "[Regev k=1024] empirical failure=0.0868  normal approx=0.0881",
        "[Regev k=4096] empirical failure=0.3814  normal approx=0.3833",
    ):
        assert expected in out
    assert {f.name for f in tmp_path.iterdir()} == {
        "crypto_basics_negligible.svg",
        "crypto_basics_hybrid.svg",
        "crypto_basics_birthday.svg",
        "crypto_basics_shares.svg",
        "crypto_basics_regev_noise.svg",
    }
