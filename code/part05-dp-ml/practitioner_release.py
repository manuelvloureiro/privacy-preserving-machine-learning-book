"""Prepare the privacy part of a model release: check the run record, recompute ε, write the statement, review a claim.

    uv run python code/part05-dp-ml/practitioner_release.py                   # the built-in copy of the Adult run
    uv run python code/part05-dp-ml/practitioner_release.py --record _build/practitioner/dp_sgd_run.json

The workflow script of the practitioner edition's release chapter (book-practitioner/part05-dp-ml/
12-release-governance.md). It does four things, and prints each:

1. Assemble the run record: load the JSON that code/part05-dp-ml/practitioner_dp_sgd.py writes at the end of its
   run (default _build/practitioner/dp_sgd_run.json; fields unit, n, delta, sigma, sample_rate, steps, epochs,
   max_grad_norm, accountant, library_versions, epsilon_prv, epsilon_pld, epsilon_rdp, tuning), check that the
   fields a release needs are there, and that δ ≤ 1/n^1.1.
2. Recompute the guarantee independently of the training code, from (σ, q, steps) alone, with dp_accounting 0.6.0:
   the privacy loss distribution (PLD) and the Rényi (RDP) accountant, each at δ and at δ/100, and the largest true
   positive rate any membership test can have at false positive rates 1e-3 and 1e-2, read from the PLD's privacy
   profile as TPR(α) ≤ min over ε of e^ε α + δ(ε) (the lower envelope of the profile's lines, full edition
   book-full/part05-dp-ml/09-choosing-epsilon.md, #eq-epsc-duality; a grid of ε gives a valid upper bound).
3. Format the privacy statement for the model card from the record and the recomputed numbers.
4. Review someone else's claim: a deliberately flawed vendor claim (made up for the book; no real product) is
   recomputed from its own (σ, q, steps, δ), and the script prints every finding: an ε that does not follow from the
   parameters, δ above 1/n^1.1, a sampler that does not match the accountant, a missing privacy unit, uncounted
   tuning.

The built-in record ADULT_RUN_RECORD is, value for value, the record that practitioner_dp_sgd.py writes with
seed 0 (Adult, n = 30,000, δ = 1e-5, q = 1/30, 300 steps, σ = 0.9729, C = 1, ε = 3.99 by Opacus's PRV accountant, secure_mode off, seed 0);
tests/test_practitioner_dp_sgd.py checks that the two agree. It is used only when --record is not given (or with
--built-in), so this script and its test run on their own; a --record path that does not exist or does not parse
is an error, never a silent fallback to the built-in record. RELEASE_CONTEXT adds what a training script does not
record (dataset, neighbouring relation, preprocessing, public data); the record's own fields take precedence.

No dataset is read: the record names the data (Adult, UCI dataset 2, CC BY 4.0) but the guarantee is recomputed from
the mechanism's parameters only. Runs in a few seconds.
"""

import argparse
import json
import math
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import dp_accounting
import numpy as np
from dp_accounting.pld import privacy_loss_distribution

FPRS: tuple[float, ...] = (1e-3, 1e-2)
EPS_GRID = np.linspace(0.0, 40.0, 4001)  # where the TPR bound looks for the best line of the profile


# --- 1. The run record ------------------------------------------------------------------------------------


@dataclass(frozen=True)
class RunRecord:
    """What a training run must write down for its privacy claim to be checked."""

    dataset: str
    unit: str | None  # the privacy unit: what one "record" is (a person, a user, a document)
    neighbours: str  # add/remove or replace-one; "not stated" if the record lacks it
    n: int  # number of privacy units in the training data
    delta: float
    sigma: float  # noise multiplier
    sample_rate: float  # q = expected batch / n
    sampler: str  # how batches are drawn in the code, as read from the loader; "not stated" if missing
    secure_mode: bool | None  # Opacus's cryptographically secure noise; None if not stated
    seed: int | None
    steps: int
    epochs: int
    max_grad_norm: float
    accountant: str
    library_versions: dict[str, str]
    epsilon_prv: float | None
    epsilon_pld: float | None
    epsilon_rdp: float | None
    tuning: dict[str, Any] | str | None  # how hyperparameters were chosen, and whether that cost is counted
    preprocessing: str = "public facts only"
    public_data: str = "none"
    audit: dict[str, Any] = field(default_factory=lambda: {})  # a lambda, not dict, so strict pyright sees the type


# The record practitioner_dp_sgd.py writes with seed 0, copied exactly (tests/test_practitioner_dp_sgd.py checks that
# they agree). Its values are the book's Adult run.
ADULT_RUN_RECORD: dict[str, Any] = {
    "unit": "person (one row of Adult)",
    "n": 30000,
    "delta": 1e-05,
    "sigma": 0.972900390625,
    "sample_rate": 0.03333333333333333,
    "steps": 300,
    "epochs": 10,
    "max_grad_norm": 1.0,
    "accountant": "prv",
    "sampler": "poisson",
    "secure_mode": False,
    "seed": 0,
    "library_versions": {
        "opacus": "1.6.0",
        "torch": "2.14.1+cpu",
        "dp-accounting": "0.6.0",
        "numpy": "2.5.4",
    },
    "epsilon_prv": 3.990840824374009,
    "epsilon_pld": 3.980570279139719,
    "epsilon_rdp": 4.502817840760661,
    "tuning": {
        "choices": (
            "learning rate 2 = 10x the non-private 0.2, the best of 6 non-private runs trained and "
            "scored only on the 2,000 public stand-in rows; batch 1,000, 10 epochs, momentum 0.9, C=1 "
            "fixed in advance"
        ),
        "private_runs": (
            "one run at epsilon=4, fixed in advance, is released; the runs at the other budgets were "
            "trained only to draw the utility curve, and nothing (not epsilon, not a hyperparameter) "
            "was chosen from them or from the baselines, none of which is released"
        ),
        "counted_in_epsilon": True,
        "not_counted": "",
    },
}

# What the release owner adds to the training record: facts about the run that the training script does not know.
RELEASE_CONTEXT: dict[str, Any] = {
    "dataset": "Adult (UCI 2), 30,000 training rows",
    "neighbours": "add/remove",
    "preprocessing": "none fitted on private rows (codebook categories, public numeric bounds)",
    "public_data": "none; 2,000 held-out Adult rows stand in for public data (clipping norm, learning-rate sweep)",
}

REQUIRED = (
    "unit", "neighbours", "n", "delta", "sigma", "sample_rate", "sampler", "steps", "max_grad_norm", "accountant",
    "secure_mode",
)  # fmt: skip


def delta_for(n: int) -> float:
    """The book's default δ: at most 1/n^1.1, rounded down to a power of ten."""
    return 10.0 ** math.floor(-1.1 * math.log10(n))


def load_record(data: dict[str, Any]) -> RunRecord:
    """Build the record from the JSON fields; a field the training script did not write becomes None or a default."""
    return RunRecord(
        dataset=str(data.get("dataset", "not stated")),
        unit=data.get("unit"),
        neighbours=str(data.get("neighbours") or "not stated"),
        n=int(data["n"]),
        delta=float(data["delta"]),
        sigma=float(data["sigma"]),
        sample_rate=float(data["sample_rate"]),
        sampler=str(data.get("sampler") or "not stated"),
        secure_mode=data.get("secure_mode"),
        seed=data.get("seed"),
        steps=int(data["steps"]),
        epochs=int(data.get("epochs", 0)),
        max_grad_norm=float(data["max_grad_norm"]),
        accountant=str(data["accountant"]),
        library_versions=dict(data.get("library_versions", data.get("versions", {}))),
        epsilon_prv=data.get("epsilon_prv"),
        epsilon_pld=data.get("epsilon_pld"),
        epsilon_rdp=data.get("epsilon_rdp"),
        tuning=data.get("tuning"),
        preprocessing=str(data.get("preprocessing", "public facts only")),
        public_data=str(data.get("public_data", "none")),
        audit=dict(data.get("audit", {})),
    )


def record_problems(record: dict[str, Any]) -> list[str]:
    """What is missing or out of the book's defaults in a record, before any number is recomputed."""
    problems = [f"missing field: {name}" for name in REQUIRED if record.get(name) in (None, "", "not stated")]
    sampler = record.get("sampler")
    if sampler not in (None, "", "not stated") and "poisson" not in str(sampler).lower():
        problems.append(f"sampler '{sampler}' is not Poisson subsampling: Poisson-subsampled accounting does not apply")
    if "n" in record and "delta" in record and record["delta"] > delta_for(int(record["n"])):
        problems.append(f"delta={record['delta']:.0e} is above 1/n^1.1 (use {delta_for(int(record['n'])):.0e})")
    tuning = record.get("tuning")
    if tuning in (None, ""):
        problems.append("tuning not reported: the cost of choosing hyperparameters is unknown")
    elif isinstance(tuning, dict) and tuning.get("counted_in_epsilon") is False:
        problems.append(f"tuning not counted in epsilon: {tuning.get('not_counted', 'no reason given')}")
    return problems


def record_limitations(record: dict[str, Any]) -> list[str]:
    """What the record states honestly but a release must disclose: a stated limitation is not a pass."""
    limitations: list[str] = []
    if record.get("secure_mode") is False:
        limitations.append(
            "secure_mode=False: Opacus drew the noise from a seeded, non-cryptographic generator so the numbers "
            "reproduce; floating-point attacks on the noise are not excluded. Retrain with secure_mode=True to ship"
        )
    return limitations


# --- 2. Recompute the guarantee -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Recomputed:
    eps_pld: float
    eps_rdp: float
    eps_pld_small_delta: float  # at δ/100
    eps_rdp_small_delta: float
    tpr_bounds: dict[float, float]  # FPR -> largest TPR of any membership test, from the PLD profile
    tpr_pair_bounds: dict[float, float]  # FPR -> the same from the single pair (ε_PLD, δ)


def dp_sgd_event(sigma: float, sample_rate: float, steps: int) -> dp_accounting.DpEvent:
    return dp_accounting.SelfComposedDpEvent(
        dp_accounting.PoissonSampledDpEvent(sample_rate, dp_accounting.GaussianDpEvent(sigma)), steps
    )


def rdp_epsilon(sigma: float, sample_rate: float, steps: int, delta: float) -> float:
    accountant = dp_accounting.rdp.RdpAccountant()
    accountant.compose(dp_sgd_event(sigma, sample_rate, steps))
    return accountant.get_epsilon(delta)


def tpr_bound(pld: privacy_loss_distribution.PrivacyLossDistribution, fpr: float) -> float:
    """Largest TPR at this FPR: the lowest line e^ε FPR + δ(ε) of the privacy profile, over a grid of ε."""
    deltas = pld.get_delta_for_epsilon(list(EPS_GRID))
    return float(min(1.0, np.min(np.exp(EPS_GRID) * fpr + deltas)))


def pair_tpr_bound(eps: float, delta: float, fpr: float) -> float:
    """Largest TPR that the single pair (ε, δ) allows: both lines of the hypothesis-testing reading."""
    return min(1.0, math.exp(eps) * fpr + delta, 1 - math.exp(-eps) * (1 - delta - fpr))


def recompute(sigma: float, sample_rate: float, steps: int, delta: float) -> Recomputed:
    # excerpt: release-recompute
    pld = privacy_loss_distribution.from_gaussian_mechanism(sigma, sampling_prob=sample_rate).self_compose(steps)
    eps_pld = pld.get_epsilon_for_delta(delta)
    eps_pld_small = pld.get_epsilon_for_delta(delta / 100)
    eps_rdp = rdp_epsilon(sigma, sample_rate, steps, delta)
    eps_rdp_small = rdp_epsilon(sigma, sample_rate, steps, delta / 100)
    tpr = {fpr: tpr_bound(pld, fpr) for fpr in FPRS}  # largest TPR of any membership test
    # end excerpt
    pair = {fpr: pair_tpr_bound(eps_pld, delta, fpr) for fpr in FPRS}
    return Recomputed(eps_pld, eps_rdp, eps_pld_small, eps_rdp_small, tpr, pair)


def reported_matches(reported: float | None, recomputed: float, tolerance: float = 0.05) -> str:
    """Compare a reported ε with an independent recomputation: a lower report than the recomputed value is a flag."""
    if reported is None:
        return "not reported"
    if reported < recomputed - tolerance:
        return f"reported {reported:.2f} is BELOW the recomputed {recomputed:.2f}"
    return f"reported {reported:.2f}, recomputed {recomputed:.2f}: consistent"


# --- 3. The privacy statement -------------------------------------------------------------------------------


def describe_tuning(tuning: dict[str, Any] | str | None) -> str:
    if not isinstance(tuning, dict):
        return tuning or "not reported"
    counted = (
        "every use of private rows is in epsilon" if tuning.get("counted_in_epsilon") else "NOT counted in epsilon"
    )
    return f"{tuning.get('choices', 'not described')}; {counted}: {tuning.get('not_counted', '')}".rstrip(": ")


# excerpt: release-statement
def format_statement(rec: RunRecord, check: Recomputed) -> str:
    """The privacy section of the model card: every number a reader needs to check the claim."""
    return "\n".join(
        [
            f"Guarantee: ({check.eps_pld:.2f}, {rec.delta:.0e})-DP per {rec.unit}, {rec.neighbours} neighbours,",
            f"  for the whole training run on {rec.dataset} (n = {rec.n:,}).",
            (
                f"  Also ({check.eps_pld_small_delta:.2f}, {rec.delta / 100:.0e})-DP. Any membership test has TPR at most "
                f"{check.tpr_bounds[1e-3]:.4f} at FPR 1e-3."
            ),
            f"Mechanism: DP-SGD, {rec.sampler} sampling q = {rec.sample_rate:.4f}, noise multiplier {rec.sigma:.4f},",
            f"  clipping norm {rec.max_grad_norm:g}, {rec.steps} steps ({rec.epochs} epochs).",
            (
                f"Accounting: {rec.accountant}, Opacus {rec.library_versions.get('opacus', '?')}; "
                f"recomputed with dp_accounting PLD (RDP gives {check.eps_rdp:.2f})."
            ),
            f"Tuning: {describe_tuning(rec.tuning)}.",
            f"Preprocessing on private data: {rec.preprocessing}. Public data: {rec.public_data}.",
            f"Noise generator: Opacus secure_mode={rec.secure_mode}, seed {rec.seed}.",
            *(f"Limitation: {item}." for item in record_limitations(asdict(rec))),
            "Not covered: data outside the training set, API outputs that use other private data, other releases.",
        ]
    )


# end excerpt


# --- 4. Review someone else's claim -------------------------------------------------------------------------

# Made up for the book: what a vendor's model card might say. Every flaw in it is one the review must catch.
FLAWED_CLAIM: dict[str, Any] = {
    "claim": "Our model is trained with differential privacy, epsilon = 1.",
    "epsilon": 1.0,
    "delta": 1e-3,
    "n": 50_000,
    "unit": None,  # not stated
    "sigma": 0.8,
    "sample_rate": 256 / 50_000,
    "steps": 5_000,
    "sampler": "shuffled fixed-size batches",
    "accountant": "RDP, Poisson-subsampled Gaussian",
    "tuning": None,  # not stated
}


# excerpt: release-review
def review(claim: dict[str, Any]) -> list[str]:
    """Recompute a claim from its own parameters and list what does not hold up."""
    findings = record_problems(claim)  # includes a sampler that is missing or not Poisson
    sigma, q, steps, delta = claim["sigma"], claim["sample_rate"], claim["steps"], claim["delta"]
    pld = privacy_loss_distribution.from_gaussian_mechanism(sigma, sampling_prob=q).self_compose(steps)
    recomputed = {"PLD": pld.get_epsilon_for_delta(delta)}
    if "rdp" in str(claim.get("accountant")).lower():  # also the accountant the claim names
        recomputed["RDP, the accountant named,"] = rdp_epsilon(sigma, q, steps, delta)
    for name, eps in recomputed.items():
        if eps > claim["epsilon"] + 0.05:
            findings.append(f"eps={claim['epsilon']} does not follow: {name} gives {eps:.2f} at its delta")
    delta_ok = delta_for(claim["n"])
    findings.append(f"at delta={delta_ok:.0e} (1/n^1.1) PLD gives eps={pld.get_epsilon_for_delta(delta_ok):.2f}")
    return findings


# end excerpt


# --- Main -----------------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0, help="unused: nothing here is random; kept for uniformity")
    parser.add_argument("--record", type=Path, default=None, help="the JSON run record of the training run")
    parser.add_argument("--built-in", action="store_true", help="use the built-in copy of the Adult run's record")
    args = parser.parse_args()

    print("== 1. Run record ==")
    record_path: Path | None = args.record
    if record_path is None:
        print("record: built-in copy of the Adult run (practitioner_dp_sgd.py, seed 0)")
        data = ADULT_RUN_RECORD
    else:
        if args.built_in:
            parser.error("--built-in and --record exclude each other")
        try:
            data = json.loads(record_path.read_text())
        except (OSError, json.JSONDecodeError) as err:
            sys.exit(f"error: cannot read the run record {record_path}: {err}")
        print(f"record: {record_path}")
    # excerpt: release-record-load
    merged = RELEASE_CONTEXT | data  # the training record, plus what the release owner adds
    rec = load_record(merged)
    for key, value in asdict(rec).items():
        print(f"{key}: {value}")
    problems = record_problems(merged)  # missing fields, sampler not Poisson, delta above 1/n^1.1, tuning
    limitations = record_limitations(merged)  # stated, so not a problem, but the statement must disclose them
    print(f"record problems: {len(problems)}")
    for problem in problems:
        print(f"  - {problem}")
    print(f"record limitations: {len(limitations)}")
    for limitation in limitations:
        print(f"  - {limitation}")
    # end excerpt

    print("== 2. Recomputed guarantee ==")
    check = recompute(rec.sigma, rec.sample_rate, rec.steps, rec.delta)
    print(f"eps PLD at delta={rec.delta:.0e}: {check.eps_pld:.3f}")
    print(f"eps RDP at delta={rec.delta:.0e}: {check.eps_rdp:.3f}")
    print(f"eps PLD at delta={rec.delta / 100:.0e}: {check.eps_pld_small_delta:.3f}")
    print(f"eps RDP at delta={rec.delta / 100:.0e}: {check.eps_rdp_small_delta:.3f}")
    for fpr in FPRS:
        print(
            f"TPR bound at FPR {fpr:.0e}: {check.tpr_bounds[fpr]:.4f} (PLD curve)  "
            f"{check.tpr_pair_bounds[fpr]:.4f} (pair eps, delta only)"
        )
    print(f"record's PRV eps: {reported_matches(rec.epsilon_prv, check.eps_pld)}")
    print(f"record's PLD eps: {reported_matches(rec.epsilon_pld, check.eps_pld)}")
    print(f"record's RDP eps: {reported_matches(rec.epsilon_rdp, check.eps_rdp)}")

    print("== 3. Privacy statement ==")
    print(format_statement(rec, check))

    print("== 4. Review of a vendor claim ==")
    print(f"claim: {FLAWED_CLAIM['claim']}")
    findings = review(FLAWED_CLAIM)
    print(f"findings: {len(findings)}")
    for finding in findings:
        print(f"  - {finding}")


if __name__ == "__main__":
    main()
