"""code/part05-dp-ml/solutions_matrix_mechanisms.py runs and reproduces the numbers in
book-exercises/part05-dp-ml/04-matrix-mechanisms.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part05-dp-ml"


def test_solutions_matrix_mechanisms_matches_page(tmp_path: Path) -> None:
    out = run(CODE / "solutions_matrix_mechanisms.py", tmp_path)
    for expected in (
        "T=8: tree rmse=2.550  identity rmse=2.121",
        "smallest T with tree below identity: 28",
        "T=28: tree rmse=3.789  identity rmse=3.808",
        "max |R R - A| at T=4: 0.0e+00",
        "T=4: sum r_k^2=1.4883  sens=1.2200",
        "h_0..h_3: 1.0000, -0.5000, -0.1250, -0.0625",
        "variance of the step-4 noise / s^2 = 1.2695  sd ratio = 1.1267",
        "T=4: square-root max error=1.4883  identity max error=2.0000",
        "rho*=0.3820  objective=2.6180  (3+sqrt5)/2=2.6180",
        "fixed point: X_12=0.3820  ||L||_F^2=2.6180",
        "singular values of A (T=2): 1.6180, 0.6180  trace-norm bound ||A||_*^2/T=2.5000",
        "square root: sens^2 * ||L||_F^2 = 2.8125  identity: 3.0000",
        "T=100: lower bound=2.178  square root=2.371  ratio=1.089",
        "T=1000: lower bound=2.902  square root=3.102  ratio=1.069",
        "T=10000: lower bound=3.634  square root=3.836  ratio=1.056",
        "ln(10)/pi=0.733",
        "T=200, columns 0 and 100: ||r_0||^2=2.7524  ||r_100||^2=2.5314  cross=1.1211  total=6.4048",
        "square-root sensitivity=2.5308",
        "banded sensitivity=2.2500  sqrt(2)*||r||=2.2500",
        "eps=2, E=1: final prefix noise dpsgd=8.27",
        "eps=2, E=1: final prefix noise identity=19.94",
        "eps=2, E=1: final prefix noise sqrt=5.05",
        "DP-SGD sigma matching the square root: 0.505  eps=7.82",
        "DP-SGD sigma at eps=2, T=100: 0.827",
    ):
        assert expected in out, expected
