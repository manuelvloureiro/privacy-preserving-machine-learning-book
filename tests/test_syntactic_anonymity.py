"""code/part02-attacks/syntactic_anonymity.py runs, writes its figures, and reproduces the numbers in book-full/part02-attacks/08-syntactic-anonymity.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part02-attacks"


def test_syntactic_anonymity_matches_section(tmp_path: Path) -> None:
    out = run(CODE / "syntactic_anonymity.py", tmp_path)
    for expected in (
        # Toy release
        "class (130**, < 30): size=3  posterior={'heart disease': 1.0}  distinct l=1  entropy l=1.00",
        "distinct l=3  entropy l=3.00",
        "class (130**, 3*): size=3  posterior={'cancer': 0.667, 'flu': 0.333}  distinct l=2  entropy l=1.89",
        "Bob, after ruling out flu: {'cancer': 1.0}",
        "TV to table=0.556",
        "TV to table=0.111",
        "TV to table=0.444",
        "c=1: bound=0.500",
        "c=2: bound=0.667",
        "c=3: bound=0.750",
        "class with 50% HIV: TV=0.49  posterior ratio=50",
        "class with 10% HIV: TV=0.09  posterior ratio=10",
        # Uniform cells
        "log2 n = 13.29",
        "k=2: people per cell=4.61  bits allowed=11.08  below log2 n by 2.20  (log2 k=1.00)",
        "k=5: people per cell=10.05  bits allowed=9.96  below log2 n by 3.33  (log2 k=2.32)",
        "k=10: people per cell=17.40  bits allowed=9.17  below log2 n by 4.12  (log2 k=3.32)",
        "k=20: people per cell=30.58  bits allowed=8.35  below log2 n by 4.93  (log2 k=4.32)",
        "k=50: people per cell=66.74  bits allowed=7.23  below log2 n by 6.06  (log2 k=5.64)",
        # Hospital table
        "HIV=0.010",
        "n=10000  exact quasi-identifiers: linkage=0.933",
        "baseline: guess the most common diagnosis=0.185  imputation from exact age=0.536  "
        "largest population posterior=0.705",
        "k=2: classes=4091  sizes 2..5  mean size=2.6  width=0.005  linkage=0.4091  certain=0.2797  guess=0.675",
        "k=5: classes=1542  sizes 5..11  mean size=6.8  width=0.016  linkage=0.1542  certain=0.0232  guess=0.593  "
        "max TV=0.937",
        "k=10: classes=748  sizes 10..20  mean size=14.1  width=0.028  linkage=0.0748  certain=0.0000  guess=0.562",
        "k=20: classes=366  sizes 20..41  mean size=29.2  width=0.043  linkage=0.0366  certain=0.0000  guess=0.540",
        "k=50: classes=129  sizes 50..99  mean size=78.4  width=0.075  linkage=0.0129  certain=0.0000  guess=0.506",
        "k=5, distinct l=3: classes=1132  sizes 5..33  mean size=10.4  width=0.022  linkage=0.1132  certain=0.0000  "
        "guess=0.574",
        "k=5, recursive (c=2, l=3): classes=389  sizes 5..641  mean size=114.9  width=0.113  linkage=0.0389  "
        "certain=0.0000  guess=0.417",
        "k=5, t=0.3: classes=495  sizes 6..60  mean size=24.9  width=0.291  linkage=0.0495  certain=0.0000  "
        "guess=0.275  max TV=0.300",
        "k=5, t=0.15: classes=136  sizes 14..304  mean size=96.6  width=0.324  linkage=0.0136  certain=0.0000  "
        "guess=0.222  max TV=0.149",
        # Composition
        "k=5: people in both=6649  pinned by A=0.0289  by B=0.0262  by A and B=0.0985 (655 people)  "
        "mean candidates A=2.79  A and B=2.38",
        "k=5, distinct l=3: people in both=6649  pinned by A=0.0000  by B=0.0000  by A and B=0.0027 (18 people)  "
        "mean candidates A=3.22  A and B=2.92",
        "k=2: pinned exact=0.8734",
        "union bound=0.8734",
        "k=5: pinned exact=0.1615",
        "k=7: pinned exact=0.0288",
        "k=10: pinned exact=0.0015",
    ):
        assert expected in out, expected
    assert {f.name for f in tmp_path.iterdir()} == {
        "anon_bits.svg",
        "anon_tradeoff.svg",
        "anon_skew.svg",
        "anon_composition.svg",
    }
