"""code/part05-dp-ml/solutions_adaptive_data_analysis.py runs and reproduces the numbers in book-exercises/part05-dp-ml/11-adaptive-data-analysis.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part05-dp-ml"


def test_solutions_adaptive_data_analysis_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_adaptive_data_analysis.py", tmp_path)
    for expected in (
        # Exercise 1
        "[Ex1 n=1000 N=2000] E population=0.3935  n/N=0.5000  error=0.6065",
        "[Ex1 n=1000 N=10000] E population=0.0952  n/N=0.1000  error=0.9048",
        "[Ex1 n=1000] smallest N with error > 1/2: 1444  queries=1445  approximation n/ln 2=1442.7",
        # Exercise 2
        "[Ex2 eps=0.1] bias=0.02498  tanh(eps/2)/2=0.02498  eps/4=0.02500  sharper (e^eps-1)/2=0.05259  lemma=0.10517",
        "[Ex2 eps=1] bias=0.23106  tanh(eps/2)/2=0.23106  eps/4=0.25000  sharper (e^eps-1)/2=0.85914  lemma=1.71828",
        # Exercise 4
        "[Ex4 k=1000 alpha=0.05] optimum delta=0.0019  formula n=697435  numerical n=764337",
        "[Ex4 n=764337] sigma=0.00620  rho=2.224e-05  eps=0.0236  noise=0.0242  transfer=0.0258",
        # Exercise 5
        "[Ex5 k=1000 r=1] n=1521",
        "[Ex5 k=1000 r=10] n=15202",
        "[Ex5 k=1000 r=1000] n=1520181",
        # Exercise 6
        "[Ex6 n=1000 k=100] simulated overfit=0.0681±0.0021  approx=0.0712",
        "[Ex6 n=4000 k=400] simulated overfit=0.0715±0.0012  approx=0.0712",
        # Exercise 7
        "[Ex7 n=2000] eps per detection=0.0894  detections within eps=1: 11",
        "[Ex7 B=300 eps=0.1] sqrt(n)=12000  n=1.440e+08",
    ):
        assert expected in out, expected
