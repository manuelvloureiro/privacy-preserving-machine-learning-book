"""code/memorization.py runs, writes its figures, and reproduces the numbers in 05-memorization-extraction.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "book" / "part02-attacks" / "code"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_memorization_matches_section(tmp_path: Path) -> None:
    out = run(CODE / "memorization.py", tmp_path)
    for expected in (
        # Corpus and model sizes
        "background lines=5000  secrets=600 (100 per copies group (1, 2, 4, 8, 16, 32))  training lines=11300  "
        "symbols=146900  example line 'toneka 14524'",
        "n=3: 2734  n=4: 19129  n=5: 42407  n=6: 52610  n=7: 54334  n=8: 54614",
        # Toolkit: add-alpha at a context seen only in the copies
        "k=1: p(true)=0.2245  p(other)=0.0204  ratio=11.0  bits per digit=3.46",
        "k=4: p(true)=0.5190  p(other)=0.0127  ratio=41.0  bits per digit=5.36",
        "k=16: p(true)=0.8090  p(other)=0.0050  ratio=161.0  bits per digit=7.33",
        # One secret
        "'migule 16430' (inserted 4 times)",
        "n=3: k*=151  greedy from 'migule ' -> '07<EOL>'",
        "n=4: k*=43  greedy from 'migule ' -> '81554'",
        "n=5: k*=9  greedy from 'migule ' -> '14032'",
        "n=6: k*=1  greedy from 'migule ' -> '16430'",
        "n=8: k*=1  greedy from 'migule ' -> '16430'",
        # Discoverable memorization
        "n=3: k=1: 0.00/0.00/0.00  k=2: 0.00/0.03/0.00  k=4: 0.00/0.00/0.00  k=8: 0.00/0.00/0.00  "
        "k=16: 0.00/0.02/0.00  k=32: 0.00/0.05/0.00",
        "n=4: k=1: 0.00/0.00/0.00  k=2: 0.00/0.00/0.00  k=4: 0.00/0.01/0.00  k=8: 0.00/0.00/0.00  "
        "k=16: 0.00/0.00/0.00  k=32: 0.13/0.17/0.13",
        "n=5: k=1: 0.00/0.00/0.00  k=2: 0.00/0.00/0.00  k=4: 0.05/0.06/0.05  k=8: 0.11/0.15/0.11  "
        "k=16: 0.42/0.42/0.42  k=32: 0.70/0.70/0.70",
        "n=6: k=1: 0.08/0.13/0.08  k=2: 0.65/0.69/0.65  k=4: 0.76/0.78/0.76  k=8: 0.80/0.85/0.80  "
        "k=16: 0.90/0.91/0.90  k=32: 0.96/0.97/0.96",
        "n=7: k=1: 0.70/0.78/0.70  k=2: 0.95/0.96/0.95  k=4: 0.98/0.98/0.98  k=8: 0.98/0.99/0.98  "
        "k=16: 0.99/0.99/0.99  k=32: 0.97/0.98/0.97",
        "n=8: k=1: 0.98/0.98/0.98  k=2: 1.00/1.00/1.00  k=4: 1.00/1.00/1.00  k=8: 0.99/0.99/0.99  "
        "k=16: 1.00/1.00/1.00  k=32: 1.00/1.00/1.00",
        "groups where the k* prediction matches greedy exactly: 36 of 36",
        # Sampling from the true prefix
        "k=4: mean p=1.42e-03  samples for 99%=3252",
        "k=16: mean p=4.42e-02  samples for 99%=102",
        "k=32: mean p=1.42e-01  samples for 99%=31",
        # Exposure null
        "n=2: mean exposure of a non-member=1.443  Pr[exposure >= 5]=0.03125 (bound 0.03125)",
        "n=3: mean exposure of a non-member=1.443  Pr[exposure >= 5]=0.03125 (bound 0.03125)",
        "n=4: mean exposure of a non-member=1.443  Pr[exposure >= 5]=0.03125 (bound 0.03125)",
        "n=8 (ties: unseen contexts give equal scores): mean=0.000  max=0.00",
        "E[exposure]=1.4426  log2 e=1.4427  log2|R|=16.61",
        # Canary exposure
        "'my pin is 58745'",
        "n=2: k=0: 0.40  k=1: 0.41  k=2: 0.43  k=4: 0.47  k=8: 0.55  k=16: 0.73  k=32: 1.21  k=64: 2.54",
        "n=3: k=0: 0.89  k=1: 4.05  k=2: 4.44  k=4: 4.71  k=8: 5.24  k=16: 6.59  k=32: 9.69  k=64: 15.61",
        "n=4: k=0: 3.76  k=1: 9.98  k=2: 12.15  k=4: 15.02  k=8: 15.61  k=16: 16.61  k=32: 16.61  k=64: 16.61",
        # Estimating exposure
        "k=1: exact=4.05  rank=6018  sampling=4.13  normal fit=4.47",
        "k=2: exact=4.44  rank=4606  sampling=4.43  normal fit=5.57",
        "k=4: exact=4.71  rank=3823  sampling=4.68  normal fit=6.81",
        "k=8: exact=5.24  rank=2648  sampling=5.35  normal fit=8.28",
        # Extraction without the prefix
        "samples=20000  distinct=14852  private training lines among them=197  public lines among them=50",
        "k=1: 0.00  k=2: 0.00  k=4: 0.00  k=8: 0.11  k=16: 0.86  k=32: 1.00",
        "precision of top 100: by likelihood=0.45  by ratio to reference=0.85",
    ):
        assert expected in out, expected
    assert {f.name for f in tmp_path.iterdir()} == {
        "memo_copies_needed.svg",
        "memo_exposure_histogram.svg",
        "memo_exposure_vs_copies.svg",
        "memo_discoverable.svg",
        "memo_extraction.svg",
    }
