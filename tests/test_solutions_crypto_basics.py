"""code/solutions_crypto_basics.py runs and reproduces the numbers in solutions/08-crypto-basics.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "book" / "part01-prerequisites" / "code"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_solutions_crypto_basics_match_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_crypto_basics.py", tmp_path)
    for expected in (
        "[Ex1] lam^-loglog lam at lambda=128: 2^-19.65",
        "[Ex2 m=1] TV=0.0100  normal=0.0080  linear=0.0080  hybrid=0.0100",
        "[Ex2 m=100] TV=0.0799  normal=0.0797  linear=0.0798  hybrid=1.0000",
        "[Ex2 m=400] TV=0.1584  normal=0.1585  linear=0.1596  hybrid=1.0000",
        "[Ex2 m=2500] TV=0.3830  normal=0.3829  linear=0.3989  hybrid=1.0000",
        "[Ex2 m=10000] TV=0.6827  normal=0.6827  linear=0.7979  hybrid=1.0000",
        "crossing k*=5050.0033 (m/2 + m delta/2 = 5050)  threshold advantage=0.6827  TV=0.6827  "
        "majority advantage=0.4807",
        "[Ex4] sqrt(2 N ln 2)=77162.7  smallest q with exact >= 1/2: 77164",
        "c1^c2 = m1^m2: True  crib 'the ' readable at positions [0, 11, 17]: 0->'meet'  11->'d is'  17->'jih '",
        "[Ex6 n=8] calls per evaluation=8  distinct outputs=256/256  fraction of 1 bits=0.5001  consistent=True",
        "x=2 shares (s1,s2,s3) for c=0..4: (2, 2, 2) (3, 4, 0) (4, 1, 3) (0, 3, 1) (1, 0, 4)",
        "single share uniform p=5: True, p=7: True; pairs reconstruct p=5: True, p=7: True; "
        "holders 1,3 with s1=4, s3=3: x=2",
        "[Ex8 uniform B=32] var=352  z=2.576  k=449.0  wrapped k=449",
        "simulated failure at that k=0.0109",
        "[Ex8 rounded normal sigma=3.2] var=10.3233  k=15309  wrapped k=15309  simulated failure at that k=0.0111",
        "[Ex9 k=1] q=4093: drift=0.00 (predicted 0.00)  failure=0.0000 (normal 0.0000)  q=4096: drift=0.00 failure=0.0000",
        "[Ex9 k=31] q=4093: drift=-7.51 (predicted -7.50)  failure=0.0000 (normal 0.0000)  "
        "q=4096: drift=0.00 failure=0.0000",
        "[Ex9 k=256] q=4093: drift=-63.79 (predicted -63.75)  failure=0.0010 (normal 0.0008)  "
        "q=4096: drift=0.00 failure=0.0003",
        "[Ex9 k=1024] q=4093: drift=-255.81 (predicted -255.75)  failure=0.1167 (normal 0.1171)  "
        "q=4096: drift=0.00 failure=0.0873",
        "[Ex9 k=4096] q=4093: drift=-1023.82 (predicted -1023.75)  failure=0.5007 (normal 0.5001)  "
        "q=4096: drift=0.00 failure=0.3780",
    ):
        assert expected in out, expected
