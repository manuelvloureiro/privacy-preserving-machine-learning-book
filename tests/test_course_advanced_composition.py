"""code/part02-dp-basics/advanced_composition.py runs, writes its figure, and reproduces the numbers in book-course/part02-dp-basics/04-advanced-composition.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part02-dp-basics"


def test_advanced_composition_matches_chapter(tmp_path: Path) -> None:
    out = run(CODE / "advanced_composition.py", tmp_path)
    for expected in (
        "eps=0.1 k=100 delta'=1e-06: basic=10.00  lead=5.257  second=0.500  advanced=5.756",
        "eps=0.1 k=1000 delta'=1e-06: basic=100.00  lead=16.623  second=4.996  advanced=21.618",
        "ln(1/delta')=13.8155  tanh(eps/2)=0.04996",
        "crossover eps=0.1 delta'=1e-06: k=31",
        "advanced/exact for k in [100, 10000]: max=1.207  at k=10000=1.062",
        "per mechanism=0.00301  advanced=0.5045",
        "largest per mechanism=0.00591  basic would give=5.911",
        "d=1000 n=100000 eps=1: basic=0.4472  corollary=0.1487",
        "d=10000 n=100000 eps=1: basic=14.1421  corollary=1.4868",
    ):
        assert expected in out
    assert (tmp_path / "ac-composition.svg").exists()
