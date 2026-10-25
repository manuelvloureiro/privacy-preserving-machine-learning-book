"""code/part02-attacks/solutions_gradient_inversion.py runs and reproduces the numbers in book-exercises/part02-attacks/06-gradient-inversion.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part02-attacks"


def test_solutions_gradient_inversion_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_gradient_inversion.py", tmp_path)
    for expected in (
        # Exercise 1
        "grad w=(-0.5379, -0.2689)  grad b=-0.2689  ratio=(2.0000, 1.0000)",
        # Exercise 2
        "|cos(top singular vector, x)| = 1.000000",
        "recovered scale c=-3.3719  ||x||=3.3719  relative error 4.0e-16",
        "mismatch at -c: 1.08e+01",
        # Exercise 3
        "tanh, 1000 trials: argmin rule 0.488  odd-one-out rule 1.000",
        # Exercise 4
        "B=16, 200 batches: largest count error, median: uniform 1.102  auxiliary 0.233",
        "share of batches with every error below 1/2: uniform 0.105  auxiliary 0.950",
        # Exercise 5
        "B=4: independent 2^-B=6.25e-02  uniform fraction 1/(B(B+1))=5.00e-02  measured on images 5.27e-02",
        "B=10: independent 2^-B=9.77e-04  uniform fraction 1/(B(B+1))=9.09e-03  measured on images 8.09e-03",
        "B=16: independent 2^-B=1.53e-05  uniform fraction 1/(B(B+1))=3.68e-03  measured on images 2.52e-03",
        "width for probability 1/2 at B=10: independent 710  uniform fraction 76",
        # Exercise 6
        "5 local steps, lr 0.5: relative error of update row ratio 1.4e-16",
        "singular values of the update: s2/s1=1.4e-16",
        "batch of 2, 1 local step(s), 200 batches: examples recovered exactly 0.963",
        "batch of 2, 5 local step(s), 200 batches: examples recovered exactly 0.917",
    ):
        assert expected in out
    assert list(tmp_path.iterdir()) == []
