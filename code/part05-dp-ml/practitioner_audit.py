"""Audit a DP-SGD implementation and a trained model: a CI test with Opacus and a one-run audit on Adult.

    uv run python code/part05-dp-ml/practitioner_audit.py
    uv run python code/part05-dp-ml/practitioner_audit.py --seed 1

Two audits for the practitioner edition's chapter on auditing (book-practitioner/part05-dp-ml/10-auditing.md),
whose results the release chapter (12-release-governance.md) attaches as evidence.

1. A unit test for continuous integration. Opacus's `DPOptimizer` wraps a one-layer `nn.Linear`. At every step the
   test writes per-example gradients straight into `p.grad_sample`: a batch of ordinary rows that never touch one
   coordinate, plus, in one of two runs, a Dirac canary, a row that is zero except on that coordinate, where it is
   ten times the clipping norm C. After `optimizer.step()` the noisy clipped sum on the canary coordinate, divided by
   C, is one observation: N(0, σ²) without the canary and N(1, σ²) with it, if the optimizer clips and adds the
   noise it claims, so each step is exactly μ-GDP with μ = 1/σ. The test chooses a threshold on the first half of
   the steps, counts false and true positives on the second half, and bounds μ from below with one-sided
   Clopper–Pearson intervals at β/2 each, μ_lo = Φ⁻¹(TPR_lo) − Φ⁻¹(FPR_hi) (the full edition's per-step audit,
   book-full/part05-dp-ml/10-auditing.md, #sec-audit-bugs). It fails when μ_lo > 1/σ, which a correct optimizer
   does with probability at most β = 5%. It runs twice: once correct, once with a planted bug, the noise variance
   σ² passed where Opacus expects the noise multiplier σ (σ = 0.7 becomes 0.49).

2. A one-run black-box audit of the released training pipeline on Adult (Steinke, Nasr & Jagielski 2023,
   Corollary 5.4, as in the full edition's #sec-audit-onerun and #sec-audit-blackbox). It imports
   practitioner_dp_sgd.py's split, public-fact preprocessing, model (91 features, 64 ReLU units, 6,018 weights),
   learning-rate choice on the public stand-in rows, σ calibration (ε = 4 at δ = 1e-5), batch 1,000, 10 epochs and
   Opacus training loop, and adds only the canaries: the first 1,000 of the release's test rows with their labels
   flipped, each joining the 30,000 training rows by a fair coin. Three models are trained once each: DP-SGD exactly
   as released, the release's non-private baseline, and a control built to memorize (non-private Adam, batch 128,
   60 epochs) that shows what the audit can detect. Each canary is scored by minus its loss; the auditor guesses
   "in" for the r/2 highest scores and "out" for the r/2 lowest, for r in (20, 50, 100, 200, 500, 1000), and
   certifies the largest ε for which the number of correct guesses is too high, with the δ term of the corollary
   and a Bonferroni correction over the six r (each test at β/6). The bound uses `scipy.stats.binom.sf`. With the
   canaries the training set has 30,537 rows, so Opacus samples at q = 1/31 for 310 steps (ε 3.91, against 3.98
   for the release's 30,000 rows at the same σ).

Data: Adult (UCI Machine Learning Repository, dataset 2, "Census Income", Becker & Kohavi 1996, CC BY 4.0), the
48,842 rows that OpenML publishes as data_id 1590, through practitioner_dp_sgd.load_splits (seed 0): 30,000
training rows, 2,000 public stand-in validation rows, and 16,842 test rows, of which 1,000 become canaries and
15,842 measure test accuracy.

Prints, with seed 0, every number the pages quote: the CI test's μ_lo and verdict for each implementation, and,
for each trained model, its test accuracy, the canaries' mean loss in and out, the correct guesses per r, and the
certified ε lower bound. Runs in about 1 minute on this machine's CPU (the CI test about 5 s, the three trainings
and the public learning-rate sweep the rest; up to 2 minutes under load). Opacus 1.6.0, SciPy 1.18.1, torch 2.14.1
(CPU).
"""

import argparse
import math
import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch
from numpy.typing import NDArray
from opacus.optimizers import DPOptimizer
from practitioner_dp_sgd import (
    BATCH,
    CLIP,
    DP_RATE_FACTOR,
    EPOCHS,
    N_TRAIN,
    TARGET_EPSILON,
    Split,
    accuracy,
    calibrate,
    load_splits,
    make_model,
    public_sweep,
    train_non_private,
    train_private,
)
from scipy import stats

Array = NDArray[np.float64]
Mask = NDArray[np.bool_]

BETA = 0.05  # every bound holds with probability at least 1 - β = 95%

# The CI test.
SIGMA_CI, CLIP_CI, BATCH_CI, STEPS_CI, DIM_CI = 0.7, 1.0, 64, 2_000, 20
CANARY_COORD = 0  # the coordinate only the canary touches
NOISE_TOLERANCE = 0.1  # relative; the sample std of 2,000 normals is within 3% of σ with overwhelming probability

# The one-run audit: the release's pipeline (practitioner_dp_sgd.py) plus canaries and a memorizing control.
N_CANARIES = 1_000
DELTA = 1e-5  # the release's delta: 1/n^1.1 = 1.2e-5 for n = 30,000, rounded down to a power of ten
MEMORIZE_EPOCHS, MEMORIZE_LR, MEMORIZE_BATCH = 60, 3e-3, 128
GUESS_GRID: tuple[int, ...] = (20, 50, 100, 200, 500, 1000)


# --- Confidence bounds on rates (Clopper–Pearson, one-sided) ------------------------------------------


def cp_upper(k: int, n: int, a: float) -> float:
    """One-sided Clopper–Pearson upper bound at level a: the rate u with P[Bin(n, u) ≤ k] = a."""
    return 1.0 if k >= n else stats.beta.ppf(1 - a, k + 1, n - k)


def cp_lower(k: int, n: int, a: float) -> float:
    """One-sided Clopper–Pearson lower bound at level a: the rate l with P[Bin(n, l) ≥ k] = a."""
    return 0.0 if k <= 0 else stats.beta.ppf(a, k, n - k + 1)


# --- 1. The CI test: Dirac canary in every step of Opacus's DPOptimizer --------------------------------


BUGS = ("correct", "variance for std", "no noise")  # the implementations the demo runs


def make_optimizer(layer: torch.nn.Linear, sigma: float, planted_bug: str) -> DPOptimizer:
    """The optimizer under test, correct or with a planted bug: σ² passed for σ, or the noise left out."""
    noise_multiplier = {"variance for std": sigma**2, "no noise": 0.0}.get(planted_bug, sigma)
    return DPOptimizer(
        torch.optim.SGD(layer.parameters(), lr=0.0),  # lr 0: the weights stay put, we only read p.grad
        noise_multiplier=noise_multiplier,
        max_grad_norm=CLIP_CI,
        expected_batch_size=BATCH_CI,
        loss_reduction="sum",  # p.grad is then the noisy clipped sum itself
    )


def batch_grad_samples(with_canary: bool) -> torch.Tensor:
    """Per-example gradients of one batch: ordinary rows that avoid the canary coordinate, plus the canary."""
    rows = 3.0 * torch.randn(BATCH_CI, 1, DIM_CI)  # norms around 13: every row gets clipped
    rows[:, :, CANARY_COORD] = 0.0
    rows[0] = 0.0
    if with_canary:
        rows[0, 0, CANARY_COORD] = 10 * CLIP_CI  # a Dirac canary, clipped to C if clipping works
    return rows


# excerpt: audit-ci-observe
def canary_coordinate(sigma: float, planted_bug: str, with_canary: bool, steps: int) -> Array:
    """Run the optimizer for `steps` steps; return the canary coordinate of each noisy sum, in units of C."""
    layer = torch.nn.Linear(DIM_CI, 1, bias=False)
    optimizer = make_optimizer(layer, sigma, planted_bug)
    observations = np.empty(steps)
    for t in range(steps):
        setattr(layer.weight, "grad_sample", batch_grad_samples(with_canary))  # as Opacus's hooks would
        optimizer.step()  # clip each row to C, sum, add N(0, σ²C²)
        assert layer.weight.grad is not None
        observations[t] = layer.weight.grad[0, CANARY_COORD].item() / CLIP_CI
        optimizer.zero_grad()
    return observations


# end excerpt


@dataclass(frozen=True)
class MuAudit:
    threshold: float
    false_pos: int
    true_pos: int
    trials: int  # per hypothesis, in the evaluation half
    mu_lo: float


def mu_lower(fp: int, n_out: int, tp: int, n_in: int, beta: float = BETA) -> float:
    """Rigorous lower bound on μ: Φ⁻¹(TPR_lo) − Φ⁻¹(FPR_hi), each rate bounded at β/2."""
    fpr_hi, tpr_lo = cp_upper(fp, n_out, beta / 2), cp_lower(tp, n_in, beta / 2)
    if tpr_lo <= 0.0:
        return 0.0
    return max(0.0, stats.norm.ppf(tpr_lo) - stats.norm.ppf(fpr_hi))


def per_step_audit(x_in: Array, x_out: Array) -> MuAudit:
    """Choose the threshold on the first half of the steps, then count and bound on the second half."""
    h_in, h_out = x_in.size // 2, x_out.size // 2
    sel_in, sel_out, ev_in, ev_out = x_in[:h_in], x_out[:h_out], x_in[h_in:], x_out[h_out:]
    candidates = [float(np.quantile(sel_out, 1 - a)) for a in (0.5, 0.2, 0.1, 0.05, 0.02, 0.01, 0.005, 0.002)]

    # Say "canary in" when the reading is strictly above the threshold: with no noise every canary-free reading is
    # exactly 0, the threshold is 0, and ">=" would flag them all as false positives and hide the bug.
    def mu_at(thr: float, s_in: Array, s_out: Array) -> float:
        return mu_lower(int(np.sum(s_out > thr)), s_out.size, int(np.sum(s_in > thr)), s_in.size)

    thr = max(candidates, key=lambda c: mu_at(c, sel_in, sel_out))
    fp, tp = int(np.sum(ev_out > thr)), int(np.sum(ev_in > thr))
    return MuAudit(thr, fp, tp, ev_in.size, mu_at(thr, ev_in, ev_out))


# excerpt: audit-ci-test
def ci_test(sigma: float, planted_bug: str, steps: int) -> tuple[MuAudit, float, bool]:
    """Fail if the canary-free noise is not σ·C, or if the measured μ exceeds the claimed 1/σ."""
    x_in = canary_coordinate(sigma, planted_bug, with_canary=True, steps=steps)
    x_out = canary_coordinate(sigma, planted_bug, with_canary=False, steps=steps)
    noise_std = float(np.std(x_out))  # in units of C: should be σ on the noisy sum
    noise_ok = abs(noise_std / sigma - 1) <= NOISE_TOLERANCE
    audit = per_step_audit(x_in, x_out)
    passed = noise_ok and audit.mu_lo <= 1 / sigma  # a correct optimizer fails with probability ≤ β
    return audit, noise_std, passed


# end excerpt


# --- 2. One-run audit of the released pipeline: practitioner_dp_sgd.py's data, model and training, plus canaries


@dataclass(frozen=True)
class CanarySplit:
    train: Split  # the release's 30,000 training rows plus the canaries whose coin came up "in"
    validation: Split  # the public stand-in rows the release chose its learning rate on
    canaries: Split  # held-out test rows with their labels flipped
    test: Split  # the rest of the release's test rows
    included: Mask


def canary_split(seed: int, n_canaries: int) -> CanarySplit:
    """The release's split (same seed), with the first test rows turned into mislabeled canaries."""
    train, validation, test = load_splits(seed)
    canaries = Split(test.x[:n_canaries], 1 - test.y[:n_canaries], test.sex[:n_canaries])
    rest = Split(test.x[n_canaries:], test.y[n_canaries:], test.sex[n_canaries:])
    included = np.random.default_rng(seed).integers(0, 2, size=n_canaries).astype(bool)  # a fair coin each
    positions = [int(i) for i in np.flatnonzero(included)]
    rows = torch.tensor(positions, dtype=torch.int64)
    joined = Split(
        torch.cat([train.x, canaries.x[rows]]),
        torch.cat([train.y, canaries.y[rows]]),
        train.sex + [canaries.sex[i] for i in positions],
    )
    return CanarySplit(joined, validation, canaries, rest, included)


def train_memorizing(train: Split, seed: int) -> torch.nn.Module:
    """A control that is built to memorize: the release's model, non-private Adam, small batches, many epochs."""
    model = make_model(train.x.shape[1], seed)
    optimizer = torch.optim.Adam(model.parameters(), lr=MEMORIZE_LR)
    loss_fn = torch.nn.CrossEntropyLoss()
    generator = torch.Generator().manual_seed(seed)
    for _ in range(MEMORIZE_EPOCHS):
        order = torch.randperm(train.y.shape[0], generator=generator)
        for start in range(0, len(order), MEMORIZE_BATCH):
            batch = order[start : start + MEMORIZE_BATCH]
            optimizer.zero_grad()
            loss_fn(model(train.x[batch]), train.y[batch]).backward()
            optimizer.step()
    return model


def per_example_loss(model: torch.nn.Module, split: Split) -> Array:
    model.eval()
    with torch.no_grad():
        loss = torch.nn.functional.cross_entropy(model(split.x), split.y, reduction="none")
    return loss.double().numpy()


# --- 2. One-run audit: guesses and the certified bound ------------------------------------------------


def correct_guesses(scores: Array, included: Mask, r: int) -> int:
    """Guess "in" for the r/2 highest scores and "out" for the r/2 lowest; count the correct guesses."""
    order = np.argsort(-scores, kind="stable")
    guessed_in, guessed_out = order[: r // 2], order[scores.size - r // 2 :]
    return int(np.count_nonzero(included[guessed_in]) + np.count_nonzero(~included[guessed_out]))


def one_run_pvalue(m: int, r: int, v: int, eps: float, delta: float) -> float:
    """P[W ≥ v] under (ε, δ)-DP with m canaries: f(v) + 2mδ max_i (f(v − i) − f(v))/i, f the binomial tail."""
    p = 1 / (1 + math.exp(-eps))
    tail = stats.binom.sf(v - 1, r, p)  # f(v) = P[Bin(r, p) ≥ v]
    i = np.arange(1, v + 1, dtype=np.float64)
    gaps = (stats.binom.sf(v - i - 1, r, p) - tail) / i
    return min(1.0, tail + 2 * m * delta * float(gaps.max()))


def one_run_eps(m: int, r: int, v: int, delta: float, beta: float) -> float:
    """The largest ε rejected at level β by v correct guesses out of r: training is not (ε, δ)-DP for smaller ε."""
    if v <= 0 or one_run_pvalue(m, r, v, 0.0, delta) >= beta:
        return 0.0
    lo, hi = 0.0, 1.0
    while one_run_pvalue(m, r, v, hi, delta) < beta and hi < 64:
        lo, hi = hi, 2 * hi
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if one_run_pvalue(m, r, v, mid, delta) < beta else (lo, mid)
    return lo


@dataclass(frozen=True)
class OneRun:
    rows: list[tuple[int, int, float]]  # (guesses r, correct v, ε lower bound at β / len(grid))

    @property
    def best(self) -> tuple[int, int, float]:
        return max(self.rows, key=lambda row: row[2])


# excerpt: audit-onerun-bound
def one_run_audit(scores: Array, included: Mask, grid: Sequence[int] = GUESS_GRID) -> OneRun:
    """Certified ε for each number of guesses r; trying len(grid) values costs a Bonferroni correction."""
    m = scores.size
    level = BETA / len(grid)
    rows: list[tuple[int, int, float]] = []
    for r in (r for r in grid if r <= m):
        v = correct_guesses(scores, included, r)
        rows.append((r, v, one_run_eps(m, r, v, DELTA, level)))
    return OneRun(rows)


# end excerpt


@dataclass(frozen=True)
class ModelAudit:
    name: str
    test_acc: float
    loss_in: float
    loss_out: float
    audit: OneRun


def audit_model(name: str, model: torch.nn.Module, data: CanarySplit) -> ModelAudit:
    losses = per_example_loss(model, data.canaries)
    return ModelAudit(
        name,
        accuracy(model, data.test),
        float(losses[data.included].mean()),
        float(losses[~data.included].mean()),
        one_run_audit(-losses, data.included),
    )


def print_model_audit(res: ModelAudit) -> None:
    print(f"{res.name}: test accuracy {res.test_acc:.4f}")
    print(f"{res.name}: mean canary loss in {res.loss_in:.3f}  out {res.loss_out:.3f}")
    for r, v, eps in res.audit.rows:
        print(f"{res.name}: r={r}  correct={v}  eps_lo={eps:.2f}")
    r, v, eps = res.audit.best
    print(f"{res.name}: certified eps lower bound {eps:.2f} (r={r}, {v} correct)")


# --- Main -----------------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=STEPS_CI, help="optimizer steps per run in the CI test")
    parser.add_argument("--canaries", type=int, default=N_CANARIES)
    args = parser.parse_args()
    torch.set_num_threads(1)  # a tiny model: one thread is as fast, and reproducible
    warnings.filterwarnings("ignore", message="Full backward hook")  # Opacus's hooks on inputs without grad

    print("== CI test: Dirac canary in every step of Opacus DPOptimizer ==")
    print(
        f"sigma={SIGMA_CI}  C={CLIP_CI}  B={BATCH_CI}  steps per run={args.steps}  claim mu=1/sigma={1 / SIGMA_CI:.3f}"
    )
    # excerpt: audit-ci-run
    for bug in BUGS:
        torch.manual_seed(args.seed)
        audit, noise_std, passed = ci_test(SIGMA_CI, bug, args.steps)
        print(
            f"{bug}: noise std {noise_std:.3f} (claim {SIGMA_CI})  FP {audit.false_pos}/{audit.trials}  "
            f"TP {audit.true_pos}/{audit.trials}"
        )
        print(f"{bug}: mu_lo={audit.mu_lo:.3f}  {'pass' if passed else 'FAIL'}")
    # end excerpt

    print("== One-run audit of the released pipeline on Adult: mislabeled canaries ==")
    steps = EPOCHS * math.ceil(N_TRAIN / BATCH)
    # excerpt: audit-onerun
    data = canary_split(args.seed, args.canaries)  # the release's split; canaries join by fair coins
    sweep = public_sweep(data.validation, steps=steps, seed=args.seed)  # the release's learning-rate choice
    rate = max(sweep, key=lambda lr: sweep[lr])
    sigma = calibrate(TARGET_EPSILON, DELTA, 1 / math.ceil(N_TRAIN / BATCH), EPOCHS)  # the release's sigma
    dp = train_private(data.train, sigma=sigma, delta=DELTA, epochs=EPOCHS, lr=DP_RATE_FACTOR * rate, seed=args.seed)
    plain = train_non_private(data.train, epochs=EPOCHS, lr=rate, seed=args.seed)
    memorizer = train_memorizing(data.train, seed=args.seed)
    for name, model in (("DP-SGD eps=4", dp.model), ("non-private", plain), ("memorizing control", memorizer)):
        print_model_audit(audit_model(name, model, data))
    # end excerpt
    n_in = int(data.included.sum())
    print(
        f"train n={data.train.y.shape[0]} ({N_TRAIN} + {n_in} canaries of {data.included.size})  "
        f"features={data.train.x.shape[1]}  test n={data.test.y.shape[0]}  delta={DELTA:.0e}"
    )
    print(
        f"DP-SGD: sigma={sigma:.4f}  lr={DP_RATE_FACTOR * rate:g}  epochs={EPOCHS}  batch={BATCH}  C={CLIP:g}  "
        f"q={dp.sample_rate:.4f}  steps={dp.steps}  eps (PRV)={dp.epsilon:.2f}"
    )
    best = max(r for r in GUESS_GRID if r <= data.included.size)
    ceiling = one_run_eps(data.included.size, best, best, DELTA, BETA / len(GUESS_GRID))
    print(f"ceiling: all {best} guesses right certifies {ceiling:.2f}")


if __name__ == "__main__":
    main()
