"""code/part01-prerequisites/solutions_stability.py runs and reproduces the numbers in book-exercises/part01-prerequisites/07-stability-generalization.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part01-prerequisites"


def test_solutions_stability_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_stability.py", tmp_path)
    for expected in (
        # Exercise 2
        "[Ex2 p=0.1] E train=0.0250  E test=0.1150  E renamed=0.1150  gap=0.0900  p(1-p)=0.0900",
        "[Ex2 p=0.3] E train=0.0750  E test=0.2850  E renamed=0.2850  gap=0.2100  p(1-p)=0.2100",
        "[Ex2 p=0.5] E train=0.1250  E test=0.3750  E renamed=0.3750  gap=0.2500  p(1-p)=0.2500",
        "[Ex2 p=0.7] E train=0.1750  E test=0.3850  E renamed=0.3850  gap=0.2100  p(1-p)=0.2100",
        "[Ex2 p=0.9] E train=0.2250  E test=0.3150  E renamed=0.3150  gap=0.0900  p(1-p)=0.0900",
        "[Ex2] argmax p=0.500  max gap=0.2500  beta=0.75",
        # Exercises 3 and 4
        "[Ex3 n=10 fair coins] train=0.2250  test=0.2750  gap=0.0500  2sigma^2/n=0.0500  bound 2/n=0.2000  exact sup=0.1900",
        "[Ex3 n=2 fair coins] train=0.1250  test=0.3750  gap=0.2500",
        "[Ex4 n=10 fair coins] LOO=0.27778  test(n-1)=0.27778  test(n)=0.27500  train=0.22500",
        # Exercise 5
        "q=0.98056  limit=0.5143",
        "[Ex5 epoch=1] measured=0.0451  strongly convex bound=0.4421  convex bound=1.0",
        "[Ex5 epoch=5] measured=0.0591  strongly convex bound=0.5143  convex bound=5.0",
        "[Ex5 epoch=20] measured=0.0528  strongly convex bound=0.5143  convex bound=20.0",
        "[Ex5 epoch=40] measured=0.0584  strongly convex bound=0.5143  convex bound=40.0",
        "[Ex5 epoch=100] measured=0.0554  strongly convex bound=0.5143  convex bound=100.0",
        # Exercise 6
        "[Ex6 n=200] 1-NN train=0.000 test=0.379 gap=0.379  15-NN train=0.246 test=0.316 gap=0.070",
        "[Ex6 n=1000] 1-NN train=0.000 test=0.369 gap=0.369  15-NN train=0.243 test=0.289 gap=0.046",
        "[Ex6 worst case k=15] loss before=0  after=1",
        # Exercise 7
        "[Ex7 c=1 T=n=10000] sum eta=9.7876  1+ln T=10.2103  HRS=0.001958  bound=0.002042",
        "[Ex7 n=1000 beta=2/sqrt(n)=0.0632] BE width=4.997  FV2018 rate=0.439  BKZ rate=1.364",
        "[Ex7 n=10000 beta=2/sqrt(n)=0.0200] BE width=4.928  FV2018 rate=0.245  BKZ rate=0.569",
        "[Ex7 n=1000000 beta=2/sqrt(n)=0.0020] BE width=4.899  FV2018 rate=0.077  BKZ rate=0.085",
        "[Ex7] limit of the stability term 2*sqrt(2 ln(1/delta))=4.895",
        # Exercise 8
        "[Ex8 signal=5] P(against trend)=Bayes error=0.250",
        "[Ex8 signal=20] P(against trend)=Bayes error=0.079",
        (
            "gap=0.0896±0.0051  mean influence=0.0918±0.0016  against trend=0.254  with trend=0.077  "
            "share against=0.082  train=0.3083  test=0.3979"
        ),
        # Exercise 9
        (
            "[Ex9 k=100 m=1000 p=0.8] bound sqrt(ln k/(2m))=0.0480  simulated bias=0.0313±0.0001  "
            "normal approx=2.5076*0.01265=0.0317  fresh test bias=-0.0001±0.0002"
        ),
    ):
        assert expected in out, expected
