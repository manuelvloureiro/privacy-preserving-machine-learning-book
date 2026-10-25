"""code/part01-privacy-failures/reconstruction.py runs, writes its figure, and reproduces the numbers in book-course/part01-privacy-failures/02-reconstruction.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part01-privacy-failures"


def test_reconstruction_matches_chapter(tmp_path: Path) -> None:
    out = run(CODE / "reconstruction.py", tmp_path)
    for expected in (
        "sorted triples=341376  with median 35=3276  with mean 47 too=36",
        "first=(0, 35, 106)  last=(35, 35, 71)",
        "4/401=0.00998",
        "(32/3) 2 ln 2=14.79",
        "n=1000 beta=0.01: linear attack m=7443  ratio=7.44",
        "(3*7)^0.5=4.58257569  second decimal=8",
        "id=2000: (2 id)^0.7=332.23241  third decimal=2  selected=True",
        "id=2001: (2 id)^0.7=332.34868  third decimal=8  selected=False",
        "ids 2000..3000: kept=501 of 1001  share=0.500",
        "n=400 m=3200 E/sqrt(n)=0.5: wrong=0.0000",
        "n=400 m=3200 E/sqrt(n)=2: wrong=0.1675",
    ):
        assert expected in out
    figure_lines = [line for line in out.splitlines() if line.startswith("figure n=")]
    assert len(figure_lines) == 3
    for line in figure_lines:  # the caption says every run ends within the tolerance 0.01
        assert float(line.split("largest violation=")[1].split()[0]) <= 0.01
    assert (tmp_path / "rec-attack.svg").exists()
