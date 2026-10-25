"""code/part05-dp-ml/matrix_mechanisms.py runs, writes its figures, and reproduces the numbers in
book-full/part05-dp-ml/04-matrix-mechanisms.md."""

from pathlib import Path

from helpers import run

CODE = Path(__file__).resolve().parent.parent / "code" / "part05-dp-ml"


def test_matrix_mechanisms_matches_chapter(tmp_path: Path) -> None:
    out = run(CODE / "matrix_mechanisms.py", tmp_path)
    for expected in (
        # Checks of the formulas
        "T=256: largest relative difference = 3.3e-16",
        "T=256: singular values closed form vs SVD, largest difference = 2.0e-13",
        "square-root coefficients r_0..r_5: 1.0000, 0.5000, 0.3750, 0.3125, 0.2734, 0.2461",
        "T=4: max |C C - A| = 0.0e+00",
        "inverse coefficients h_0..h_5: 1.0000, -0.5000, -0.1250, -0.0625, -0.0391, -0.0273",
        "sum of h_k^2 over 100000 terms = 1.2732  (4/pi = 1.2732)",
        # One participation
        "T=100 identity: sensitivity=1.000  rmse=7.11  max error=10.00",
        "T=100 prefix: sensitivity=10.000  rmse=10.00  max error=10.00",
        "T=100 tree: sensitivity=2.828  rmse=5.05  max error=6.93",
        "T=100 sqrt: sensitivity=1.591  rmse=2.37  max error=2.53",
        "T=100 optimal: sensitivity=1.000  rmse=2.24  max error=2.45",
        "T=100 lower bound on rmse (trace norm) = 2.18",
        "T=100 square root: sum r_k^2 = 2.531  bounds [2.338, 2.647]  asymptotic 2.532  printed Psi(T) = 2.392",
        "T=256 sqrt: sensitivity=1.683  rmse=2.67  max error=2.83",
        "T=256 optimal: sensitivity=1.000  rmse=2.52  max error=2.75",
        "T=10000 identity: sensitivity=1.000  rmse=70.71  max error=100.00",
        "T=10000 prefix: sensitivity=100.000  rmse=100.00  max error=100.00",
        "T=10000 tree: sensitivity=3.873  rmse=9.84  max error=13.96",
        "T=10000 sqrt: sensitivity=2.000  rmse=3.84  max error=4.00",
        "T=10000 lower bound on rmse (trace norm) = 3.63",
        "T=100 optimal factorization: 155 fixed-point iterations",
        "T=256 optimal factorization: 193 fixed-point iterations",
        "tree levels: T=100: 8  T=10000: 15",
        # Several participations
        (
            "E=1 T=100: identity sens=1.000 last=10.00  tree sens=2.828 last=4.90  sqrt sens=1.591 last=2.53  "
            "banded sens=1.591 last=2.53"
        ),
        (
            "E=2 T=200: identity sens=1.414 last=20.00  tree sens=4.690 last=8.12  sqrt sens=2.531 last=4.20  "
            "banded sens=2.250 last=4.16"
        ),
        (
            "E=5 T=500: identity sens=2.236 last=50.00  tree sens=8.944 last=21.91  sqrt sens=5.080 last=8.86  "
            "banded sens=3.558 last=8.56"
        ),
        (
            "E=10 T=1000: identity sens=3.162 last=100.00  tree sens=16.186 last=39.65  sqrt sens=9.154 last=16.54  "
            "banded sens=5.031 last=15.69"
        ),
        (
            "E=20 T=2000: identity sens=4.472 last=200.00  tree sens=30.430 last=74.54  sqrt sens=17.191 last=32.09  "
            "banded sens=7.115 last=29.85"
        ),
        "banded check: sqrt(20)*||c||=7.115",
        # Noise at equal epsilon
        "eps=2: mu=0.502",
        "eps=8: mu=1.666",
        "E=1 T=100: dpsgd sigma=0.827  identity s=1.994  tree s=5.639  sqrt s=3.172  banded s=3.172",
        "E=1 noise in the final prefix sum: dpsgd=8.27  identity=19.94  tree=9.77  sqrt=5.05  banded=5.05",
        "E=5 T=500: dpsgd sigma=0.928  identity s=4.458  tree s=17.833  sqrt s=10.129  banded s=7.093",
        "E=5 noise in the final prefix sum: dpsgd=20.75  identity=99.69  tree=43.68  sqrt=17.67  banded=17.06",
        "E=10 T=1000: dpsgd sigma=1.03  identity s=6.305  tree s=32.273  sqrt s=18.251  banded s=10.031",
        "E=10 noise in the final prefix sum: dpsgd=32.57  identity=199.38  tree=79.05  sqrt=32.98  banded=31.28",
        "E=1 T=100: dpsgd sigma=0.501  identity s=0.600  tree s=1.698  sqrt s=0.955  banded s=0.955",
        "E=1 noise in the final prefix sum: dpsgd=5.01  identity=6.00  tree=2.94  sqrt=1.52  banded=1.52",
        "E=5 T=500: dpsgd sigma=0.573  identity s=1.342  tree s=5.369  sqrt s=3.049  banded s=2.135",
        "E=5 noise in the final prefix sum: dpsgd=12.81  identity=30.01  tree=13.15  sqrt=5.32  banded=5.14",
        "E=10 T=1000: dpsgd sigma=0.616  identity s=1.898  tree s=9.716  sqrt s=5.495  banded s=3.020",
        "E=10 noise in the final prefix sum: dpsgd=19.48  identity=60.02  tree=23.80  sqrt=9.93  banded=9.42",
        # Training
        (
            "eps=2 E=1: dpsgd 0.9343+-0.0044 (lr=2)  identity 0.9066+-0.0031 (lr=2)  tree 0.9134+-0.0038 (lr=2)  "
            "sqrt 0.9361+-0.0046 (lr=2)  banded 0.9358+-0.0019 (lr=2)"
        ),
        (
            "eps=2 E=5: dpsgd 0.9460+-0.0021 (lr=0.5)  identity 0.8941+-0.0071 (lr=0.5)  tree 0.9183+-0.0084 (lr=0.5)  "
            "sqrt 0.9430+-0.0011 (lr=0.5)  banded 0.9456+-0.0019 (lr=0.5)"
        ),
        (
            "eps=2 E=10: dpsgd 0.9493+-0.0012 (lr=0.5)  identity 0.8641+-0.0130 (lr=0.5)  tree 0.9096+-0.0074 (lr=0.5)  "
            "sqrt 0.9398+-0.0029 (lr=0.5)  banded 0.9491+-0.0024 (lr=0.5)"
        ),
        (
            "eps=8 E=1: dpsgd 0.9371+-0.0016 (lr=2)  identity 0.9395+-0.0018 (lr=2)  tree 0.9412+-0.0006 (lr=2)  "
            "sqrt 0.9461+-0.0022 (lr=2)  banded 0.9434+-0.0017 (lr=2)"
        ),
        (
            "eps=8 E=5: dpsgd 0.9524+-0.0026 (lr=2)  identity 0.9420+-0.0029 (lr=0.5)  tree 0.9461+-0.0012 (lr=0.5)  "
            "sqrt 0.9497+-0.0025 (lr=2)  banded 0.9542+-0.0014 (lr=2)"
        ),
        (
            "eps=8 E=10: dpsgd 0.9520+-0.0006 (lr=0.5)  identity 0.9410+-0.0032 (lr=0.5)  tree 0.9484+-0.0004 (lr=0.5)  "
            "sqrt 0.9530+-0.0015 (lr=0.5)  banded 0.9549+-0.0024 (lr=2)"
        ),
        "no noise, clipped, fixed order, E=1: 0.9420",
        "no noise, clipped, fixed order, E=5: 0.9580",
        "no noise, clipped, fixed order, E=10: 0.9606",
    ):
        assert expected in out, expected
    for name in ("mfac_noise_paths.svg", "mfac_errors.svg", "mfac_matrices.svg", "mfac_accuracy.svg"):
        assert (tmp_path / name).exists()
