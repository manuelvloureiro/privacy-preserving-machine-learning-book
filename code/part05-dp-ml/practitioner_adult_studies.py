"""Run the Adult experiments that the practitioner edition's DP-SGD topic pages quote, beside the workflow's run.

    uv run python code/part05-dp-ml/practitioner_adult_studies.py                 # every study, about 25 minutes
    uv run python code/part05-dp-ml/practitioner_adult_studies.py --study users   # one study
    uv run python code/part05-dp-ml/practitioner_adult_studies.py --check         # the accounting and five runs

Every study trains the model of practitioner_dp_sgd.py (91 inputs -> 64 ReLU units -> 2 logits) on its split of
Adult (30,000 private training rows, 2,000 public stand-in rows, 16,842 test rows, split seed 0), with its
schedule: sampling rate q = 1/30, 10 epochs (300 steps), clipping norm C = 1, SGD with momentum 0.9 at the
DP-SGD learning rate 2, and delta = 1e-5. Each study prints one line per training run, then the averages and
the accounting its page quotes:

  dpsgd     02-dp-sgd.md: test accuracy at five budgets over three seeds; the learning rate times C held fixed
            or not; the drift of the gradient norms; microbatches of 4; the epsilon of fixed-size batches.
  public    06-public-data.md: pretraining on the 1,500 public stand-in rows that practitioner_dp_sgd.py tunes
            on, then fine-tuning the last layer, a rank-4 adapter plus the last layer, or every weight, against
            training from scratch. Each method's learning rate is chosen at epsilon = 1 on half of the test rows
            and scored on the other half.
  users     07-user-level.md: the 30,000 rows given to 3,000 made-up households, the user accountant, and three
            routes to a household-level guarantee.
  groups    08-disparate-impact.md: accuracy by race, a group's share of the training rows, and oversampling it.
  curve     09-choosing-epsilon.md: the trade-off curve of the runs at epsilon 4 and 8.

Training goes through Opacus (practitioner_dp_sgd.train_private) when Opacus can do what the study needs. The
runs that clip something other than one row (microbatches, households) or sample rows at different rates
(oversampling) use train_units below, the same algorithm written out with torch.func. The household accountant
is the full edition's (code/part05-dp-ml/user_level.py), since no library has one.

The runs are independent, so they go to a process pool of PRIVACY_WORKERS processes (default 4), one CPU thread
each. Every run repeats exactly. --check prints the accounting and five representative runs, for the test.
"""

import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")

import argparse
import math
import statistics
import sys
import warnings
from collections.abc import Callable
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from dp_accounting.pld import privacy_loss_distribution as pld
from numpy.typing import NDArray
from opacus.grad_sample import GradSampleModule
from scipy.stats import binomtest
from torch import nn
from torch.func import functional_call, grad, vmap  # pyright: ignore[reportUnknownVariableType]

sys.path.insert(0, str(Path(__file__).resolve().parent))
import practitioner_dp_sgd as wf
import user_level

IntArray = NDArray[np.int64]
FloatArray = NDArray[np.float64]

SEEDS = (0, 1, 2)
LR = 2.0  # the workflow's DP-SGD learning rate, ten times the best non-private one on the public rows
NON_PRIVATE_LR = 0.2  # the best non-private learning rate on the public rows
DELTA = 1e-5
Q = 1 / 30
STEPS = 300
BUDGETS = (0.25, 1.0, 2.0, 4.0, 8.0)
WORKERS = int(os.environ.get("PRIVACY_WORKERS") or min(4, os.cpu_count() or 1))

torch.set_num_threads(1)


@dataclass(frozen=True)
class Data:
    train: wf.Split
    validation: wf.Split
    test: wf.Split
    race: tuple[NDArray[np.str_], NDArray[np.str_]]  # of the training rows and of the test rows


_DATA: list[Data] = []


def data() -> Data:
    """The workflow's split (seed 0), loaded once per process."""
    if not _DATA:
        adult = wf.practitioner_datasets.adult()
        order = np.random.default_rng(0).permutation(adult.height)  # the permutation of wf.load_splits(0)
        race: NDArray[np.str_] = adult["race"].to_numpy()[order]
        train, validation, test = wf.load_splits(0)
        _DATA.append(Data(train, validation, test, (race[: wf.N_TRAIN], race[wf.N_TRAIN + wf.N_VALIDATION :])))
    return _DATA[0]


def sigma_for(epsilon: float, q: float = Q) -> float:
    """The noise multiplier for (epsilon, DELTA) after STEPS steps at rate q, from Opacus's PRV accountant.

    Opacus counts steps in epochs of 1/q steps, so STEPS steps at rate q are STEPS * q epochs.
    """
    return wf.calibrate(epsilon, DELTA, q, round(STEPS * q))


def subset(split: wf.Split, rows: IntArray) -> wf.Split:
    index = torch.from_numpy(rows)
    return wf.Split(split.x[index], split.y[index], [split.sex[i] for i in rows.tolist()])


def opacus(
    epsilon: float | None,
    seed: int,
    *,
    clip: float = wf.CLIP,
    lr: float = LR,
    model: nn.Module | None = None,
    train: wf.Split | None = None,
) -> nn.Module:
    """The workflow's Opacus run at a budget (None: clipping only, no noise), returning the trained model."""
    sigma = 0.0 if epsilon is None else sigma_for(epsilon)
    run = wf.train_private(
        train or data().train, sigma=sigma, delta=DELTA, epochs=wf.EPOCHS, lr=lr, seed=seed, clip=clip, model=model
    )
    wrapped = run.model
    assert isinstance(wrapped, GradSampleModule)
    return wrapped.to_standard_module()  # removes Opacus's hooks, so the model can be wrapped again


def non_private(seed: int, train: wf.Split | None = None) -> nn.Module:
    return wf.train_non_private(train or data().train, epochs=wf.EPOCHS, lr=NON_PRIVATE_LR, seed=seed)


# ---------------------------------------------------------------------------------------------------------------
# DP-SGD written out, for units other than one row


def train_units(
    train: wf.Split,
    *,
    unit: IntArray,
    rate: FloatArray,
    sigma: float,
    lr: float = LR,
    microbatch: int = 1,
    seed: int = 0,
) -> nn.Module:
    """DP-SGD over units: each step samples unit u with probability rate[u] (Poisson), averages the gradients of
    each sampled unit's rows, clips each average to C, adds Gaussian noise and divides by the expected number of
    units, rate.sum(); then a step of SGD with momentum 0.9, as torch.optim.SGD takes it.

    unit[i] is the unit of row i: the row itself for ordinary DP-SGD, its household for household-level DP-SGD.
    With microbatch = m > 1 the units are rows, and the sampled rows are cut at random into groups of m whose
    average gradient is clipped. Adding a row changes its group's clipped average by up to 2C, so the noise
    doubles, and the update divides by the expected number of groups.
    """
    model = wf.make_model(train.x.shape[1], seed)
    params = {name: p.detach().clone() for name, p in model.named_parameters()}
    velocity = {name: torch.zeros_like(p) for name, p in params.items()}
    criterion = nn.CrossEntropyLoss()

    def loss(p: dict[str, torch.Tensor], x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
        logits: torch.Tensor = functional_call(model, p, (x.unsqueeze(0),))  # pyright: ignore[reportUnknownVariableType]
        return criterion(logits, y.unsqueeze(0))

    per_row: Callable[..., dict[str, torch.Tensor]] = vmap(grad(loss), in_dims=(None, 0, 0))  # pyright: ignore[reportUnknownVariableType]
    rng = np.random.default_rng(seed)
    noise = torch.Generator().manual_seed(seed)
    expected = float(rate.sum()) / microbatch
    scale = sigma * wf.CLIP * (2.0 if microbatch > 1 else 1.0)
    for _ in range(STEPS):
        chosen = rng.random(rate.size) < rate
        rows = np.flatnonzero(chosen[unit])
        if microbatch > 1:
            groups = rng.permutation(rows.size) // microbatch
        else:
            groups = np.unique(unit[rows], return_inverse=True)[1]
        total = {name: torch.zeros_like(p) for name, p in params.items()}
        if rows.size:
            index = torch.from_numpy(rows)
            g = per_row(params, train.x[index], train.y[index])
            group = torch.from_numpy(groups.astype(np.int64))
            count = torch.bincount(group).to(torch.float32)
            means = {
                name: torch.zeros((len(count), *t.shape[1:])).index_add_(0, group, t)
                / count.view(-1, *[1] * (t.dim() - 1))
                for name, t in g.items()
            }
            squares = torch.stack([m.flatten(start_dim=1).pow(2).sum(dim=1) for m in means.values()]).sum(dim=0)
            factor = torch.clamp(wf.CLIP / (squares.sqrt() + 1e-12), max=1.0)
            total = {name: (m * factor.view(-1, *[1] * (m.dim() - 1))).sum(dim=0) for name, m in means.items()}
        for name, p in params.items():
            update = (total[name] + scale * torch.randn(p.shape, generator=noise)) / expected
            velocity[name] = 0.9 * velocity[name] + update
            params[name] = p - lr * velocity[name]
    with torch.no_grad():
        for name, p in model.named_parameters():
            p.copy_(params[name])
    return model


def rows_as_units(n: int) -> IntArray:
    return np.arange(n, dtype=np.int64)


# ---------------------------------------------------------------------------------------------------------------
# A run is a job: a function of plain arguments, run in a worker, that returns its key and its accuracies.

Result = tuple[tuple[object, ...], dict[str, float]]
Job = tuple[Callable[..., Result], tuple[object, ...]]


def test_accuracy(model: nn.Module) -> dict[str, float]:
    return {"test": wf.accuracy(model, data().test)}


# ---------------------------------------------------------------------------------------------------------------
# 02-dp-sgd.md


def job_budget(epsilon: float | str, seed: int) -> Result:
    """epsilon is a budget, "clip-only" (clipping, no noise) or "non-private"."""
    if epsilon == "non-private":
        model = non_private(seed)
    else:
        model = opacus(None if epsilon == "clip-only" else float(epsilon), seed)
    return ("budget", epsilon, seed), test_accuracy(model)


def job_clip(clip: float, lr: float) -> Result:
    return ("clip", clip, lr), test_accuracy(opacus(4.0, 0, clip=clip, lr=lr))


def job_drift() -> Result:
    d = data()
    start = wf.per_example_grad_norms(wf.make_model(d.train.x.shape[1], 0), d.validation)
    end = wf.per_example_grad_norms(opacus(4.0, 0), d.validation)
    values = {"median start": float(np.median(start)), "median end": float(np.median(end))}
    values |= {"clipped start": float(np.mean(start > wf.CLIP)), "clipped end": float(np.mean(end > wf.CLIP))}
    return ("drift",), values


def job_microbatch(m: int) -> Result:
    n = len(data().train.y)
    model = train_units(data().train, unit=rows_as_units(n), rate=np.full(n, Q), sigma=sigma_for(4.0), microbatch=m)
    return ("microbatch", m), test_accuracy(model)


def dpsgd_jobs() -> list[Job]:
    found: list[Job] = [(job_budget, (e, s)) for e in (*BUDGETS, "clip-only", "non-private") for s in SEEDS]
    found += [(job_clip, (c, LR / c)) for c in (0.1, 0.3, 1.0, 3.0, 10.0)] + [(job_clip, (10.0, LR))]
    return found + [(job_drift, ())] + [(job_microbatch, (m,)) for m in (1, 4)]


def dpsgd_accounting_lines() -> list[str]:
    sigma = sigma_for(4.0)
    poisson, fixed = epsilon_pld(sigma), epsilon_pld(sigma / 2)
    return [f"epsilon at sigma={sigma:.3f}, Poisson: {poisson:.2f}; fixed-size batches, sigma/2: {fixed:.2f} (PLD)"]


def dpsgd_lines(results: dict[tuple[object, ...], dict[str, float]]) -> list[str]:
    lines: list[str] = []
    for e in (*BUDGETS, "clip-only", "non-private"):
        runs = [results[("budget", e, s)]["test"] for s in SEEDS if ("budget", e, s) in results]
        if len(runs) == len(SEEDS):
            label = f"epsilon={e:g} (sigma={sigma_for(e):.3f})" if isinstance(e, float) else str(e)
            lines.append(f"{label}: test accuracy {statistics.mean(runs):.4f} +- {statistics.stdev(runs):.4f}")
    if ("drift",) in results:
        r = results[("drift",)]
        lines.append(
            f"median gradient norm on the public rows: {r['median start']:.3f} at initialisation, {r['median end']:.3f} "
            f"after {STEPS} steps; fraction clipped at C=1: {r['clipped start']:.3f} -> {r['clipped end']:.3f}"
        )
    return lines


def epsilon_pld(sigma: float) -> float:
    return pld.from_gaussian_mechanism(sigma, sampling_prob=Q).self_compose(STEPS).get_epsilon_for_delta(DELTA)


# ---------------------------------------------------------------------------------------------------------------
# 06-public-data.md


class LowRank(nn.Module):
    """A frozen linear layer plus a trainable correction (up @ down) of rank r, up starting at zero (LoRA)."""

    def __init__(self, base: nn.Linear, rank: int, seed: int) -> None:
        super().__init__()
        self.base = base
        for p in base.parameters():
            p.requires_grad_(False)
        torch.manual_seed(1_000 + seed)  # the adapter's draw depends on the seed only, never on the data
        self.down = nn.Linear(base.in_features, rank, bias=False)
        self.up = nn.Linear(rank, base.out_features, bias=False)
        nn.init.zeros_(self.up.weight)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out: torch.Tensor = self.base(x) + self.up(self.down(x))
        return out


PUBLIC_EPOCHS = 200  # 200 epochs of the 1,500 public rows: 400 steps of batch 1,000
METHODS = ("scratch", "full", "last layer", "adapter")
PUBLIC_RATES = (0.1, 0.3, 1.0, 3.0)
PUBLIC_BUDGETS = (0.25, 0.5, 1.0, 4.0)
PUBLIC_SEEDS = (0, 1)
CHOOSE_AT = 1.0  # the budget at which each method's learning rate is chosen


def pretrained() -> nn.Module:
    """The running example trained without privacy on the 1,500 public stand-in rows the workflow tunes on."""
    fit = subset(data().validation, np.arange(wf.N_SWEEP_FIT, dtype=np.int64))
    return wf.train_non_private(fit, epochs=PUBLIC_EPOCHS, lr=NON_PRIVATE_LR, seed=0)


def prepared(method: str, seed: int, rank: int = 4) -> nn.Module:
    """The model a method fine-tunes, with the weights it freezes frozen."""
    if method == "scratch":
        return wf.make_model(data().train.x.shape[1], seed)
    model = pretrained()
    first, relu, last = list(model.children())
    assert isinstance(first, nn.Linear) and isinstance(last, nn.Linear)
    if method == "last layer":
        for p in first.parameters():
            p.requires_grad_(False)
    elif method == "adapter":
        model = nn.Sequential(LowRank(first, rank, seed), relu, last)
    return model


def trainable(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def halves() -> tuple[wf.Split, wf.Split]:
    """The test rows cut in two: the first chooses each method's learning rate, the second scores it."""
    test = data().test
    rows = np.random.default_rng(1).permutation(len(test.y))
    return subset(test, rows[: rows.size // 2]), subset(test, rows[rows.size // 2 :])


def job_public(method: str, epsilon: float | None, lr: float, seed: int, rank: int = 4) -> Result:
    """Fine-tune with DP-SGD at a budget, or without privacy (epsilon None), and score on both halves."""
    model = prepared(method, seed, rank)
    if epsilon is None:
        model.train()
        optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=lr, momentum=0.9)
        torch.manual_seed(seed)
        for _ in range(wf.EPOCHS):
            for x, y in wf.loader(data().train):
                optimizer.zero_grad()
                nn.CrossEntropyLoss()(model(x), y).backward()
                optimizer.step()
    else:
        model = opacus(epsilon, seed, lr=lr, model=model)
    choose, score = halves()
    return ("public", method, rank, epsilon, lr, seed), {
        "choose": wf.accuracy(model, choose),
        "score": wf.accuracy(model, score),
    }


def clipped_signal(model: nn.Module, split: wf.Split, chunk: int = 2_000) -> float:
    """The length of the average clipped per-example gradient over the trainable weights."""
    total: torch.Tensor | None = None
    for start in range(0, len(split.y), chunk):
        part = subset(split, np.arange(start, min(start + chunk, len(split.y)), dtype=np.int64))
        wrapped = GradSampleModule(model)
        wrapped.train()
        logits: torch.Tensor = wrapped(part.x)
        nn.CrossEntropyLoss()(logits, part.y).backward()
        samples: list[torch.Tensor] = [
            vars(p)["grad_sample"].flatten(start_dim=1) for p in wrapped.parameters() if p.requires_grad
        ]
        grads = torch.cat(samples, dim=1)
        norms = grads.pow(2).sum(dim=1).sqrt()
        factor = torch.clamp(wf.CLIP / (norms + 1e-12), max=1.0)
        summed = (grads * factor.unsqueeze(1)).sum(dim=0)
        total = summed if total is None else total + summed
        wrapped.to_standard_module()
        model.zero_grad(set_to_none=True)
    assert total is not None
    return float(total.pow(2).sum().sqrt()) / len(split.y)


def hessian_trace(model: nn.Module, split: wf.Split, probes: int = 200) -> float:
    """Hutchinson's estimate of the trace of the loss's Hessian over the trainable weights."""
    weights = [p for p in model.parameters() if p.requires_grad]
    loss = nn.CrossEntropyLoss()(model(split.x), split.y)
    grads = torch.autograd.grad(loss, weights, create_graph=True)
    gen = torch.Generator().manual_seed(0)
    total = 0.0
    for _ in range(probes):
        v = [torch.randint(0, 2, w.shape, generator=gen).float() * 2 - 1 for w in weights]
        hv = torch.autograd.grad(grads, weights, grad_outputs=v, retain_graph=True)
        total += float(torch.stack([(a * b).sum() for a, b in zip(hv, v, strict=True)]).sum())
    return total / probes


def job_curvature(method: str) -> Result:
    """The noise raises the loss by about (noise per weight)^2 / 2 times the Hessian's trace over the trained weights."""
    return ("curvature", method), {"trace": hessian_trace(prepared(method, 0), data().train)}


def public_jobs() -> list[Job]:
    found: list[Job] = [
        (job_public, (m, e, lr, s))
        for m in METHODS
        for e in PUBLIC_BUDGETS
        for lr in PUBLIC_RATES
        for s in PUBLIC_SEEDS
    ]
    found += [(job_public, (m, None, lr, 0)) for m in ("scratch", "last layer") for lr in PUBLIC_RATES]
    found += [
        (job_public, ("adapter", CHOOSE_AT, lr, s, r)) for r in (1, 16) for lr in PUBLIC_RATES for s in PUBLIC_SEEDS
    ]
    return found + [(job_curvature, (m,)) for m in ("full", "last layer")]


def public_accounting_lines() -> list[str]:
    d = data()
    _, score = halves()
    lines = [f"pretrained on the public rows, no fine-tuning: {wf.accuracy(pretrained(), score):.4f}"]
    lines += [f"trainable weights, {m}: {trainable(prepared(m, 0)):,}" for m in METHODS]
    sigma = sigma_for(CHOOSE_AT)
    for method in ("full", "last layer"):
        model = prepared(method, 0)
        noise = sigma * wf.CLIP * math.sqrt(trainable(model)) / (Q * len(d.train.y))
        signal = clipped_signal(model, d.train)
        lines.append(
            f"epsilon=1 (sigma={sigma:.3f}), {method}: noise length {noise:.4f}  signal length {signal:.4f}  "
            f"ratio {noise / signal:.1f}"
        )
    return lines


def public_lines(results: dict[tuple[object, ...], dict[str, float]]) -> list[str]:
    def mean(method: str, rank: int, epsilon: float | None, lr: float, key: str) -> float:
        seeds = PUBLIC_SEEDS if epsilon is not None else (0,)
        return statistics.mean(results[("public", method, rank, epsilon, lr, s)][key] for s in seeds)

    lines: list[str] = []
    for method, rank in [(m, 4) for m in METHODS] + [("adapter", 1), ("adapter", 16)]:
        best = max(PUBLIC_RATES, key=lambda lr: mean(method, rank, CHOOSE_AT, lr, "choose"))
        budgets = PUBLIC_BUDGETS if rank == 4 else (CHOOSE_AT,)
        cells = "  ".join(f"epsilon={e:g}: {mean(method, rank, e, best, 'score'):.4f}" for e in budgets)
        lines.append(f"{method} rank={rank}, learning rate {best:g} (chosen at epsilon=1): {cells}")
        if rank == 4:
            sweep = "  ".join(f"lr={lr:g}: {mean(method, rank, CHOOSE_AT, lr, 'score'):.4f}" for lr in PUBLIC_RATES)
            lines.append(f"{method} at epsilon=1, scored: {sweep}")
    for method in ("scratch", "last layer"):
        best = max(PUBLIC_RATES, key=lambda lr: mean(method, 4, None, lr, "choose"))
        lines.append(f"{method} without privacy, learning rate {best:g}: {mean(method, 4, None, best, 'score'):.4f}")
    traces = {m: results[("curvature", m)]["trace"] for m in ("full", "last layer")}
    lines.append(
        f"Hessian trace: full {traces['full']:.4f}, last layer {traces['last layer']:.4f}, "
        f"ratio {traces['full'] / traces['last layer']:.1f}"
    )
    return lines


# ---------------------------------------------------------------------------------------------------------------
# 07-user-level.md


HOUSEHOLDS = 3_000
UNITS_PER_BATCH = 100
HOUSEHOLD_LR = 0.2  # the best of 0.1, 0.2, 0.5 and 1 for household-level DP-SGD at epsilon = 4
HOUSEHOLD_BUDGETS = (1.0, 2.0, 4.0, 8.0)
CAP = 2
STRATEGIES = ("all rows", "households", f"cap={CAP}")


def households() -> IntArray:
    """The household of each training row: 3,000 households of very uneven size, each a cluster of similar rows."""
    rng = np.random.default_rng(user_level.USER_SEED)
    counts = user_level.user_counts(rng, wf.N_TRAIN, HOUSEHOLDS, user_level.LOG_SPREAD)
    return user_level.assign_users(rng, data().train.x.numpy().astype(np.float64), counts)


def capped(owner: IntArray, k: int) -> IntArray:
    """At most k rows of each household, chosen at random without looking at any other household."""
    taken = np.zeros(HOUSEHOLDS, dtype=np.int64)
    keep: list[int] = []
    for r in np.random.default_rng(7).permutation(owner.size).tolist():
        if taken[owner[r]] < k:
            taken[owner[r]] += 1
            keep.append(r)
    return np.sort(np.array(keep, dtype=np.int64))


def household_plan(strategy: str, epsilon: float, owner: IntArray) -> tuple[float, float, IntArray]:
    """(sigma, sampling rate, rows trained on) for a household-level (epsilon, DELTA) guarantee in 300 steps."""
    if strategy == "all rows":
        q = UNITS_PER_BATCH / owner.size
        largest = int(np.bincount(owner).max())
        return user_level.sigma_for_user(q, STEPS, largest, epsilon, DELTA), q, rows_as_units(owner.size)
    if strategy == "households":
        q = UNITS_PER_BATCH / HOUSEHOLDS
        return user_level.sigma_for_user(q, STEPS, 1, epsilon, DELTA), q, rows_as_units(owner.size)
    k = int(strategy.removeprefix("cap="))
    rows = capped(owner, k)
    q = UNITS_PER_BATCH / rows.size
    return user_level.sigma_for_user(q, STEPS, k, epsilon, DELTA), q, rows


def job_households(strategy: str, epsilon: float, lr: float = HOUSEHOLD_LR) -> Result:
    owner = households()
    sigma, q, rows = household_plan(strategy, epsilon, owner)
    train = subset(data().train, rows)
    if strategy == "households":
        model = train_units(train, unit=owner, rate=np.full(HOUSEHOLDS, q), sigma=sigma, lr=lr)
    else:
        model = train_units(train, unit=rows_as_units(rows.size), rate=np.full(rows.size, q), sigma=sigma, lr=lr)
    return ("households", strategy, epsilon, lr), test_accuracy(model) | {"sigma": sigma, "rows": rows.size}


def users_jobs() -> list[Job]:
    found: list[Job] = [(job_households, ("households", 4.0, lr)) for lr in (0.1, 0.5, 1.0)]
    found += [(job_households, (st, e)) for st in STRATEGIES for e in HOUSEHOLD_BUDGETS]
    return found + [(job_households, (f"cap={k}", 4.0)) for k in (1, 50)]


def users_accounting_lines() -> list[str]:
    owner = households()
    counts = np.bincount(owner)
    sigma = sigma_for(4.0)
    row_epsilon = user_level.record_epsilon(Q, sigma, STEPS, DELTA)
    lines = [
        f"households {HOUSEHOLDS:,}: median {int(np.median(counts))} rows, largest {int(counts.max())}",
        f"delta for {HOUSEHOLDS:,} households: {wf.delta_for(HOUSEHOLDS):.0e}",
        f"row-level epsilon at sigma={sigma:.3f} (the full edition's RDP accountant): {row_epsilon:.2f}",
    ]
    lines += [
        f"group privacy, k={k}: epsilon {user_level.generic_group_epsilon(Q, sigma, STEPS, k, DELTA)[0]:.2f}"
        for k in (2, 10)
    ]
    for k in (1, 2, 5, 10, 20):
        exact = user_level.user_epsilon(Q, sigma, STEPS, k, DELTA)[0]
        screen = user_level.record_epsilon(k * Q, sigma, STEPS, DELTA)
        lines.append(
            f"household of k={k} rows: exact {exact:.2f}  screen at rate kq {screen:.2f}  ({screen / exact - 1:+.0%})"
        )
    for name, size in (("rows", wf.N_TRAIN), ("households", HOUSEHOLDS)):
        q = UNITS_PER_BATCH / size
        sigma_units = user_level.sigma_for_user(q, STEPS, 1, 4.0, DELTA)
        lines.append(f"epsilon=4 with {UNITS_PER_BATCH} {name} per batch (q={q:.4g}): sigma {sigma_units:.3g}")
    for k in (1, CAP, 50):
        sigma_cap, _, rows = household_plan(f"cap={k}", 4.0, owner)
        lines.append(f"cap k={k}: {rows.size:,} rows kept, sigma for household-level epsilon=4 {sigma_cap:.3g}")
    return lines


# ---------------------------------------------------------------------------------------------------------------
# 08-disparate-impact.md


RACES = wf.CATEGORIES["race"]
GROUP = "Asian-Pac-Islander"  # the group that pays most at epsilon = 1 among those with 500 test rows or more
SHARE = 0.01
OVERSAMPLE = 4
GROUP_TRAININGS = ("non-private", "clip-only", 4.0, 1.0)


def by_race(model: nn.Module) -> dict[str, float]:
    right = (wf.predictions(model, data().test) == data().test.y).numpy()
    return {race: float(right[data().race[1] == race].mean()) for race in RACES}


def job_race(training: float | str, seed: int) -> Result:
    if training == "non-private":
        model = non_private(seed)
    else:
        model = opacus(None if training == "clip-only" else float(training), seed)
    return ("race", training, seed), by_race(model)


def share_rows(share: float) -> IntArray:
    """Training rows with GROUP cut down to `share` of them."""
    race = data().race[0]
    group, rest = np.flatnonzero(race == GROUP), np.flatnonzero(race != GROUP)
    keep = round(share * rest.size / (1 - share))
    return np.sort(np.concatenate([rest, np.random.default_rng(3).permutation(group)[:keep]]))


def job_share(epsilon: float | None, seed: int) -> Result:
    train = subset(data().train, share_rows(SHARE))
    model = non_private(seed, train) if epsilon is None else opacus(epsilon, seed, train=train)
    return ("share", epsilon, seed), by_race(model)


def job_rate(oversample: int, epsilon: float, seed: int) -> Result:
    """The written-out DP-SGD with GROUP's rows sampled at oversample * q; their rate sets epsilon, so sigma."""
    n = len(data().train.y)
    rate = np.where(data().race[0] == GROUP, oversample * Q, Q)
    sigma = sigma_for(epsilon, q=oversample * Q)
    model = train_units(data().train, unit=rows_as_units(n), rate=rate, sigma=sigma, seed=seed)
    return ("rate", oversample, epsilon, seed), by_race(model)


def groups_jobs() -> list[Job]:
    found: list[Job] = [(job_race, (t, s)) for t in GROUP_TRAININGS for s in SEEDS]
    found += [(job_share, (e, s)) for e in (None, 1.0) for s in SEEDS]
    return found + [(job_rate, (o, 1.0, s)) for o in (1, OVERSAMPLE) for s in SEEDS]


def groups_accounting_lines() -> list[str]:
    race_train, race_test = data().race
    lines = [
        f"{race}: {int((race_train == race).sum()):,} training rows, {int((race_test == race).sum()):,} test rows"
        for race in RACES
    ]
    rows = share_rows(SHARE)
    lines.append(f"{GROUP} cut to {SHARE:.0%}: {int((race_train[rows] == GROUP).sum()):,} of {rows.size:,} rows")
    lines.append(
        f"oversampling x{OVERSAMPLE} at epsilon=1: sigma {sigma_for(1.0):.3f} -> {sigma_for(1.0, q=OVERSAMPLE * Q):.3f}"
    )
    for race in (GROUP, "Other"):
        total = int((race_test == race).sum())
        interval = binomtest(round(0.85 * total), total).proportion_ci(confidence_level=0.95, method="exact")
        lines.append(
            f"exact 95% interval at accuracy 0.85, {race} ({total} test rows): [{interval.low:.3f}, {interval.high:.3f}]"
        )
    return lines


def groups_lines(results: dict[tuple[object, ...], dict[str, float]]) -> list[str]:
    def mean(*key: object) -> dict[str, float]:
        runs = [results[(*key, s)] for s in SEEDS]
        return {race: statistics.mean(r[race] for r in runs) for race in RACES}

    rows = [(f"race, {t}", mean("race", t)) for t in GROUP_TRAININGS]
    rows += [(f"{GROUP} cut to {SHARE:.0%}, {e or 'non-private'}", mean("share", e)) for e in (None, 1.0)]
    rows += [(f"written-out DP-SGD at epsilon=1, {GROUP} sampled x{o}", mean("rate", o, 1.0)) for o in (1, OVERSAMPLE)]
    return [f"{label}: " + "  ".join(f"{race} {acc:.4f}" for race, acc in accs.items()) for label, accs in rows]


# ---------------------------------------------------------------------------------------------------------------
# 09-choosing-epsilon.md


EPS_GRID = np.linspace(0.0, 20.0, 2001)


def tpr_bound(sigma: float, fpr: float, pessimistic: bool) -> float:
    """The trade-off curve at fpr: the lowest of the lines e^eps * fpr + delta(eps) of the run's PLD."""
    run = pld.from_gaussian_mechanism(
        sigma, sampling_prob=Q, pessimistic_estimate=pessimistic, use_connect_dots=pessimistic
    ).self_compose(STEPS)
    deltas = np.array(run.get_delta_for_epsilon(list(EPS_GRID)))
    return float(np.min(np.exp(EPS_GRID) * fpr + deltas))


def curve_lines() -> list[str]:
    lines: list[str] = []
    for epsilon in (4.0, 8.0):
        sigma = sigma_for(epsilon)
        pair = min(1.0, math.exp(epsilon) * 1e-3 + DELTA, 1 - math.exp(-epsilon) * (1 - DELTA - 1e-3))
        cells = [f"pair at FPR 1e-3: {pair:.4g}"]
        cells += [
            f"curve at FPR {fpr:g}: [{tpr_bound(sigma, fpr, False):.4g}, {tpr_bound(sigma, fpr, True):.4g}]"
            for fpr in (1e-3, 1e-2)
        ]
        lines.append(f"epsilon={epsilon:g} (sigma={sigma:.3f}): " + "  ".join(cells))
    upper, prior = tpr_bound(sigma_for(8.0), 1e-2, True), 0.01
    lines.append(
        f"right flags at prior 1%, FPR 1e-2, epsilon=8: {prior * upper / (prior * upper + (1 - prior) * 1e-2):.4f}"
    )
    return lines


# ---------------------------------------------------------------------------------------------------------------


def run(job: Job) -> Result:
    warnings.filterwarnings("ignore", category=UserWarning)
    function, args = job
    return function(*args)


def show(key: tuple[object, ...], values: dict[str, float]) -> str:
    return " ".join(str(k) for k in key) + ": " + "  ".join(f"{name} {v:.4f}" for name, v in values.items())


CHECK_JOBS: list[Job] = [
    (job_budget, (4.0, 1)),
    (job_microbatch, (4,)),
    (job_public, ("adapter", 1.0, 0.3, 0)),
    (job_households, ("households", 4.0)),
    (job_race, (1.0, 0)),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--study", choices=["dpsgd", "public", "users", "groups", "curve", "all"], default="all")
    parser.add_argument("--check", action="store_true", help="the accounting and five runs only")
    args = parser.parse_args()
    study: str = "all" if args.check else args.study
    warnings.filterwarnings("ignore", category=UserWarning)
    studies: dict[str, tuple[Callable[[], list[str]], Callable[[], list[Job]]]] = {
        "dpsgd": (dpsgd_accounting_lines, dpsgd_jobs),
        "public": (public_accounting_lines, public_jobs),
        "users": (users_accounting_lines, users_jobs),
        "groups": (groups_accounting_lines, groups_jobs),
        "curve": (curve_lines, list),
    }
    chosen = list(studies) if study == "all" else [study]
    for name in chosen:
        print(f"== {name}: accounting")
        print("\n".join(studies[name][0]()), flush=True)
    jobs = CHECK_JOBS if args.check else [job for name in chosen for job in studies[name][1]()]
    results: dict[tuple[object, ...], dict[str, float]] = {}
    print("== runs")
    with ProcessPoolExecutor(WORKERS) as pool:
        for key, values in pool.map(run, jobs):
            results[key] = values
            print(show(key, values), flush=True)
    if args.check:
        return
    summaries = {"dpsgd": dpsgd_lines, "public": public_lines, "groups": groups_lines}
    for name in chosen:
        if name in summaries:
            print(f"== {name}: averages")
            print("\n".join(summaries[name](results)))


if __name__ == "__main__":
    main()
