"""code/gradient_inversion.py runs, writes its figures, and reproduces the numbers in 06-gradient-inversion.md."""

import subprocess
import sys
from pathlib import Path

CODE = Path(__file__).resolve().parent.parent / "book" / "part02-attacks" / "code"


def run(script: Path, out: Path) -> str:
    return subprocess.run(
        [sys.executable, str(script), "--out", str(out)], capture_output=True, text=True, check=True
    ).stdout


def test_gradient_inversion_matches_section(tmp_path: Path) -> None:
    out = run(CODE / "gradient_inversion.py", tmp_path)
    for expected in (
        # Worked example
        "a=(0.1200, 0.4700)  h=(0.1200, 0.4700)  z=(-0.3500, 0.2950)  p=(0.3441, 0.6559)",
        "d2=p-e_y=(-0.6559, 0.6559)  W2^T d2=(-0.3279, 0.9838)  d1=(-0.3279, 0.9838)",
        "grad W1 row 1=(-0.1640, -0.3279, -0.0656)  row 2=(0.4919, 0.9838, 0.1968)  grad b1=(-0.3279, 0.9838)",
        "ratios: unit 1 (0.5000, 1.0000, 0.2000)  unit 2 (0.5000, 1.0000, 0.2000)",
        "grad b2=(-0.6559, 0.6559)  label from bias=0  grad W2 row sums=(-0.3870, 0.3870)",
        # Section 14.3
        "units with nonzero grad b1: 13 of 32",
        "largest relative error over those units: 6.7e-17",
        "singular values of grad W1: s1=1.6155  s2/s1=9.5e-17",
        "gradient entries: 2212  input pixels: 64",
        # Section 14.4
        "single example, 1000 trials: bias rule 1.000  weight rule 1.000",
        "batch of 16: true counts [4, 2, 4, 6]  uniform guess (2.12, 3.01, 4.27, 6.60)  "
        "auxiliary guess (3.88, 2.00, 4.19, 5.92)  true p (4.00, 2.00, 4.00, 6.00)",
        "batch 4: all counts exact, uniform guess 0.925  auxiliary guess 1.000",
        "batch 16: all counts exact, uniform guess 0.095  auxiliary guess 0.975",
        "batch 64: all counts exact, uniform guess 0.000  auxiliary guess 0.525",
        # Section 14.5
        "iteration 0: relative error 1.6e+00  matching loss 1.3e+00",
        "iteration 10: relative error 6.7e-01  matching loss 2.0e-01",
        "iteration 50: relative error 7.5e-02  matching loss 2.5e-03",
        "iteration 100: relative error 5.7e-03  matching loss 1.3e-05",
        "iteration 200: relative error 3.4e-05  matching loss 4.6e-10",
        "iteration 400: relative error 7.8e-10  matching loss 2.5e-19",
        "iteration 600: relative error 5.8e-14  matching loss 1.3e-27",
        "B=1: median relative error 0.000  recovered (<0.05) 1.00  median matching loss 1.5e-27",
        "B=2: median relative error 0.000  recovered (<0.05) 0.67  median matching loss 1.3e-27",
        "B=4: median relative error 0.363  recovered (<0.05) 0.29  median matching loss 3.3e-05",
        "B=8: median relative error 0.711  recovered (<0.05) 0.00  median matching loss 7.7e-05",
        # Section 14.6
        "B=4: singular values / s1: 1.0e+00 8.4e-01 6.9e-01 3.9e-02 1.5e-16 1.3e-16",
        "B=4: largest distance of a row of grad W1 from span(x): 4.6e-16",
        "B=4: examples recovered exactly by unit ratios: 4 of 4",
        "m=64 B=4: formula 0.984  gaussian inputs 0.975  images 0.838",
        "m=64 B=8: formula 0.222  gaussian inputs 0.188  images 0.550",
        "m=64 B=16: formula 0.001  gaussian inputs 0.000  images 0.144",
        "m=256 B=8: formula 0.633  gaussian inputs 0.562  images 0.769",
        "m=256 B=12: formula 0.061  gaussian inputs 0.046  images 0.542",
        "m=256 B=16: formula 0.004  gaussian inputs 0.006  images 0.375",
        "m=1024 B=12: formula 0.221  gaussian inputs 0.196  images 0.850",
        "m=1024 B=16: formula 0.016  gaussian inputs 0.009  images 0.691",
        "width for isolation probability 1/2 at B=10: 710",
        "double backprop against finite differences, batch of 2: largest gap 1.2e-10",
    ):
        assert expected in out
    assert {f.name for f in tmp_path.iterdir()} == {
        "gradinv_rank_one.svg",
        "gradinv_labels.svg",
        "gradinv_dlg_progress.svg",
        "gradinv_dlg_batch.svg",
        "gradinv_isolation.svg",
    }
