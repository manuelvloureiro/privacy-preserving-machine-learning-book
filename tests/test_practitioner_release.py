"""code/part05-dp-ml/practitioner_release.py runs and prints the numbers that the practitioner release chapter quotes."""

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "code" / "part05-dp-ml" / "practitioner_release.py"


def run(*args: str) -> str:
    return subprocess.run([sys.executable, str(SCRIPT), *args], capture_output=True, text=True, check=True).stdout


@pytest.mark.libs
def test_practitioner_release_adult_run() -> None:
    out = run("--built-in")
    for expected in (
        "record: built-in copy of the Adult run (practitioner_dp_sgd.py, seed 0)",
        "record problems: 0",
        "record limitations: 1",
        "  - secure_mode=False: Opacus drew the noise from a seeded, non-cryptographic generator so the numbers "
        "reproduce; floating-point attacks on the noise are not excluded. Retrain with secure_mode=True to ship",
        "secure_mode: False",
        "seed: 0",
        "Noise generator: Opacus secure_mode=False, seed 0.",
        "Tuning: learning rate 2 = 10x the non-private 0.2, the best of 6 non-private runs trained and scored only "
        "on the 2,000 public stand-in rows; batch 1,000, 10 epochs, momentum 0.9, C=1 fixed in advance; every use "
        "of private rows is in epsilon.",
        # Recomputed guarantee
        "eps PLD at delta=1e-05: 3.981",
        "eps RDP at delta=1e-05: 4.503",
        "eps PLD at delta=1e-07: 5.191",
        "eps RDP at delta=1e-07: 5.736",
        "TPR bound at FPR 1e-03: 0.0123 (PLD curve)  0.0536 (pair eps, delta only)",
        "TPR bound at FPR 1e-02: 0.0652 (PLD curve)  0.5355 (pair eps, delta only)",
        "record's PRV eps: reported 3.99, recomputed 3.98: consistent",
        "record's PLD eps: reported 3.98, recomputed 3.98: consistent",
        "record's RDP eps: reported 4.50, recomputed 4.50: consistent",
        # Statement
        "Guarantee: (3.98, 1e-05)-DP per person (one row of Adult), add/remove neighbours,",
        "  Also (5.19, 1e-07)-DP. Any membership test has TPR at most 0.0123 at FPR 1e-3.",
        "Mechanism: DP-SGD, poisson sampling q = 0.0333, noise multiplier 0.9729,",
        "  clipping norm 1, 300 steps (10 epochs).",
        "Accounting: prv, Opacus 1.6.0; recomputed with dp_accounting PLD (RDP gives 4.50).",
        "Not covered: data outside the training set, API outputs that use other private data, other releases.",
        # Review of the flawed claim
        "findings: 10",
        "  - missing field: unit",
        "  - missing field: neighbours",
        "  - missing field: max_grad_norm",
        "  - missing field: secure_mode",
        "  - sampler 'shuffled fixed-size batches' is not Poisson subsampling: Poisson-subsampled accounting does not "
        "apply",
        "  - delta=1e-03 is above 1/n^1.1 (use 1e-06)",
        "  - tuning not reported: the cost of choosing hyperparameters is unknown",
        "  - eps=1.0 does not follow: PLD gives 2.17 at its delta",
        "  - eps=1.0 does not follow: RDP, the accountant named, gives 2.55 at its delta",
        "  - at delta=1e-06 (1/n^1.1) PLD gives eps=3.75",
    ):
        assert expected in out, expected


@pytest.mark.libs
def test_practitioner_release_reads_a_record(tmp_path: Path) -> None:
    # A record as practitioner_dp_sgd.py writes it, here with an under-reported ε and no tuning entry.
    record = {
        "unit": "person",
        "n": 30_000,
        "delta": 1e-5,
        "sigma": 0.972900390625,
        "sample_rate": 1 / 30,
        "steps": 300,
        "epochs": 10,
        "max_grad_norm": 1.0,
        "accountant": "Opacus PRV",
        "library_versions": {"opacus": "1.6.0"},
        "epsilon_prv": 3.5,
        "epsilon_pld": None,
        "epsilon_rdp": None,
    }
    path = tmp_path / "run.json"
    path.write_text(json.dumps(record))
    out = run("--record", str(path))
    assert f"record: {path}" in out
    assert "record problems: 3" in out
    assert "  - missing field: sampler" in out  # never defaulted to Poisson
    assert "  - missing field: secure_mode" in out
    assert "sampler: not stated" in out
    assert "Mechanism: DP-SGD, not stated sampling" in out
    assert "eps PLD at delta=1e-05: 3.981" in out
    assert "record's PRV eps: reported 3.50 is BELOW the recomputed 3.98" in out
    assert "record's PLD eps: not reported" in out


@pytest.mark.libs
def test_review_flags_a_hidden_sampler_and_the_named_accountant() -> None:
    spec = importlib.util.spec_from_file_location("practitioner_release", SCRIPT)
    assert spec is not None and spec.loader is not None
    release = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(release)
    # The walkthrough's claim: sampler not stated, accountant "Opacus RDP", epsilon 8 at delta 1e-7.
    claim = {
        "epsilon": 8.0, "delta": 1e-7, "n": 2_000_000, "unit": "example (one message)", "neighbours": "add/remove",
        "sigma": 0.55, "sample_rate": 4096 / 2_000_000, "steps": 7_324, "max_grad_norm": 1.0,
        "sampler": "not stated", "accountant": "Opacus RDP", "secure_mode": None, "tuning": "grid of 8, not counted",
    }  # fmt: skip
    findings: list[str] = release.review(claim)
    assert "missing field: sampler" in findings
    assert "missing field: secure_mode" in findings
    assert any(f.startswith("eps=8.0 does not follow: RDP, the accountant named,") for f in findings), findings


@pytest.mark.libs
def test_practitioner_release_rejects_a_missing_record(tmp_path: Path) -> None:
    missing = tmp_path / "no_such_run.json"
    result = subprocess.run([sys.executable, str(SCRIPT), "--record", str(missing)], capture_output=True, text=True)
    assert result.returncode != 0
    assert "cannot read the run record" in result.stderr
    assert "Guarantee:" not in result.stdout  # no statement from the built-in record
