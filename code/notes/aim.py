"""The worked example of the study note notes/aim/: AIM, run end to end on a four-attribute toy dataset.

    uv run python code/notes/aim.py [--out DIR]

AIM (McKenna, Mullins, Sheldon and Miklau, VLDB 2022) generates differentially private synthetic data by repeatedly
choosing a marginal (a contingency table over a few attributes), measuring it with Gaussian noise, and fitting a
model of the data to every noisy marginal measured so far. The note explains the algorithm to a reader who knows
probability and nothing else, and this script carries its running example:

- A toy dataset of 1,000 people with four attributes (smoker, age group, exercise, condition), drawn with seed 0
  from a fixed population in which smoking and the condition depend on age and on each other. Its full table has
  2 x 3 x 2 x 2 = 24 cells, small enough to print, which is what lets the note show every step.
- The workload: every two-way marginal, six of them, with equal weight. The candidates are the ten marginals in its
  downward closure (the four one-way and the six two-way marginals).
- Private-PGM in miniature: the model p_theta(x) proportional to exp(sum of one parameter table per measured
  marginal), fitted by minimizing the weighted squared error between its marginals and the noisy measurements.
  Rather than a junction tree, the script computes the model's marginals from its full 24-cell table, which is
  exact on a domain this small; Private-PGM computes the same thing without ever building the table.
- AIM itself with the paper's defaults (T = 16d rounds planned, 90% of each round's budget on measuring), under
  rho-zCDP for the rho that gives (epsilon, delta) = (1, 1e-6): initialization on the one-way marginals, selection
  by the exponential mechanism with the noise-aware score, Gaussian measurement, refitting, budget annealing and a
  final round that spends what is left.

Output: two SVG figures (scores.svg, rounds.svg) and five Markdown snippets (_marginal.md, _init.md, _scores.md,
_schedule.md, _rounds.md) that the note includes, written to _build/figures/notes/aim/ (or --out), and every number the note
quotes, printed. Seed 0 draws the data and seed 1 the mechanisms' noise, so the note and the script agree.
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt

matplotlib.rcParams["svg.hashsalt"] = "aim"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

NOTE = "aim"
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / "notes" / NOTE

Array = npt.NDArray[np.float64]
Attrs = tuple[int, ...]  # a marginal is named by the attributes it keeps, as sorted axis indices

ATTRIBUTES = ("smoker", "age", "exercise", "condition")
VALUES = (("no", "yes"), ("under 40", "40 to 64", "65 and over"), ("rarely", "often"), ("no", "yes"))
SIZES = tuple(len(v) for v in VALUES)
N = 1_000

# The paper's defaults: T = 16 d rounds planned, a 0.9 share of each round's budget on the measurement.
ROUNDS_PER_ATTRIBUTE = 16
MEASURE_SHARE = 0.9
EPSILON, DELTA = 1.0, 1e-6
SQRT_2_OVER_PI = math.sqrt(2.0 / math.pi)


# --- The toy population and its marginals ----------------------------------------------------------------------


def draw_records(n: int, seed: int) -> npt.NDArray[np.int64]:
    """n records from a fixed population: age first, then smoking and exercise given age, then the condition
    given age and smoking, so that the two-way marginals carry real structure for AIM to find."""
    rng = np.random.default_rng(seed)
    age = rng.choice(3, size=n, p=[0.45, 0.35, 0.20])
    smoker = (rng.random(n) < np.array([0.30, 0.25, 0.15])[age]).astype(np.int64)
    exercise = (rng.random(n) < np.array([0.60, 0.45, 0.30])[age]).astype(np.int64)
    condition = (rng.random(n) < np.array([0.05, 0.15, 0.30])[age] + 0.20 * smoker).astype(np.int64)
    return np.stack([smoker, age, exercise, condition], axis=1)


def full_table(records: npt.NDArray[np.int64]) -> Array:
    """The count of records in every cell of the domain, as an array with one axis per attribute."""
    table = np.zeros(SIZES)
    np.add.at(table, tuple(records[:, i] for i in range(len(SIZES))), 1.0)
    return table


def marginal(table: Array, attrs: Attrs) -> Array:
    """The marginal over `attrs`: the table summed over every other attribute, flattened to a vector."""
    other = tuple(i for i in range(table.ndim) if i not in attrs)
    return np.asarray(table.sum(axis=other), dtype=np.float64).reshape(-1)


def cells(attrs: Attrs) -> int:
    """The number of cells of the marginal over `attrs`."""
    return math.prod(SIZES[i] for i in attrs)


def name(attrs: Attrs) -> str:
    return " × ".join(ATTRIBUTES[i] for i in attrs)


def downward_closure(workload: list[Attrs]) -> list[Attrs]:
    """Every non-empty subset of a workload marginal, in a fixed order: the candidates AIM may measure."""
    found: set[Attrs] = set()
    for attrs in workload:
        for mask in range(1, 2 ** len(attrs)):
            found.add(tuple(a for i, a in enumerate(attrs) if mask >> i & 1))
    return sorted(found, key=lambda a: (len(a), a))


def candidate_weight(attrs: Attrs, workload: list[Attrs], weights: dict[Attrs, float]) -> float:
    """w_r = sum over workload marginals s of c_s |r ∩ s|: how much of the workload the candidate touches."""
    return sum(weights[s] * len(set(attrs) & set(s)) for s in workload)


# --- Private-PGM in miniature -------------------------------------------------------------------------------------


@dataclass
class Measurement:
    attrs: Attrs
    noisy: Array  # the marginal's counts plus Gaussian noise
    sigma: float  # the noise's standard deviation


def model_table(theta: dict[Attrs, Array], total: float) -> Array:
    """The model's full table: `total` times the softmax of the sum of one parameter table per measured marginal,
    each broadcast over the attributes it does not mention. This is the graphical model Private-PGM fits."""
    log_p = np.zeros(SIZES)
    for attrs, param in theta.items():
        shape = tuple(SIZES[i] if i in attrs else 1 for i in range(len(SIZES)))
        log_p = log_p + param.reshape(shape)
    log_p = log_p - log_p.max()
    p = np.exp(log_p)
    return total * p / p.sum()


def fit(measurements: list[Measurement], total: float, steps: int = 3000, lr: float = 0.05) -> Array:
    """The table of the model that minimizes sum_i (1/sigma_i) ||marginal_i(model) - noisy_i||_2^2, by Adam on the
    parameters, from zero. Returns the fitted full table."""
    theta: dict[Attrs, Array] = {m.attrs: np.zeros(cells(m.attrs)) for m in measurements}
    first = {a: np.zeros_like(p) for a, p in theta.items()}
    second = {a: np.zeros_like(p) for a, p in theta.items()}
    for step in range(1, steps + 1):
        p = model_table(theta, total)
        # d loss / d p: each measurement's residual, broadcast back over the full table.
        grad_p = np.zeros(SIZES)
        for m in measurements:
            residual = 2.0 / m.sigma * (marginal(p, m.attrs) - m.noisy)
            shape = tuple(SIZES[i] if i in m.attrs else 1 for i in range(len(SIZES)))
            grad_p = grad_p + residual.reshape(shape)
        # Through the softmax: d loss / d log-odds = p * (grad_p - weighted mean of grad_p).
        grad_eta = p * (grad_p - (p * grad_p).sum() / total)
        for attrs in list(theta):
            g = marginal(grad_eta, attrs)
            first[attrs] = 0.9 * first[attrs] + 0.1 * g
            second[attrs] = 0.999 * second[attrs] + 0.001 * g * g
            m_hat = first[attrs] / (1 - 0.9**step)
            v_hat = second[attrs] / (1 - 0.999**step)
            theta[attrs] = theta[attrs] - lr * m_hat / (np.sqrt(v_hat) + 1e-8)
    return model_table(theta, total)


def round_to_records(table: Array) -> Array:
    """A table of whole counts with the same total: round every cell down, then give the leftover records to the
    cells with the largest fractional parts. Private-PGM's randomized rounding does this with a random tie-break."""
    floors = np.floor(table)
    leftover = round(float(table.sum() - floors.sum()))
    order = np.argsort(-(table - floors).reshape(-1), kind="stable")[:leftover]
    flat = floors.reshape(-1).copy()
    flat[order] += 1.0
    return flat.reshape(SIZES)


# --- zCDP arithmetic -------------------------------------------------------------------------------------------------


def delta_of(rho: float, eps: float) -> float:
    """The (eps, delta)-DP guarantee of rho-zCDP (Canonne, Kamath and Steinke, 2020):
    delta = min over alpha > 1 of exp((alpha - 1)(alpha rho - eps)) / (alpha - 1) * (1 - 1/alpha)^alpha."""
    alphas = 1.0 + np.logspace(-4, 3, 20_001)
    log_delta = (alphas - 1) * (alphas * rho - eps) - np.log(alphas - 1) + alphas * np.log1p(-1 / alphas)
    return float(np.exp(log_delta.min()))


def rho_for(eps: float, delta: float) -> float:
    """The largest rho whose zCDP guarantee converts to (eps, delta)-DP, by bisection (delta grows with rho)."""
    lo, hi = 0.0, eps
    for _ in range(100):
        mid = 0.5 * (lo + hi)
        if delta_of(mid, eps) <= delta:
            lo = mid
        else:
            hi = mid
    return lo


def round_cost(eps: float, sigma: float) -> float:
    """The zCDP cost of one round: eps^2/8 for the exponential mechanism plus 1/(2 sigma^2) for the Gaussian."""
    return eps**2 / 8 + 1 / (2 * sigma**2)


# --- AIM ------------------------------------------------------------------------------------------------------------


@dataclass
class Round:
    t: int
    attrs: Attrs
    sigma: float
    eps: float
    cost: float
    model_error: float  # ||marginal(D) - marginal(model before)||_1 on the chosen marginal
    penalty: float  # sqrt(2/pi) sigma cells: the L1 noise the measurement adds in expectation
    change: float  # ||marginal(model after) - marginal(model before)||_1
    annealed: bool
    workload_error: float  # the normalized workload error of the model after the round


def workload_error(data: Array, model: Array, workload: list[Attrs], weights: dict[Attrs, float], n: int) -> float:
    """The paper's metric: the weighted L1 error over the workload, divided by the number of marginals and by n."""
    total = sum(weights[s] * float(np.abs(marginal(data, s) - marginal(model, s)).sum()) for s in workload)
    return total / (len(workload) * n)


# excerpt: score
def score(
    data: Array,
    model: Array,
    attrs: Attrs,
    weight: float,
    sigma: float,
) -> float:
    """AIM's quality score of a candidate marginal: how far the model is from the data
    on it, minus the L1 error that measuring it with noise of standard deviation sigma
    would add in expectation, times the weight."""
    model_error = float(np.abs(marginal(data, attrs) - marginal(model, attrs)).sum())
    penalty = SQRT_2_OVER_PI * sigma * cells(attrs)
    return weight * (model_error - penalty)


# end excerpt


def selection_probabilities(scores: Array, eps: float, sensitivity: float) -> Array:
    """The exponential mechanism's probabilities: proportional to exp(eps * score_i / (2 * sensitivity))."""
    logits = eps * scores / (2 * sensitivity)
    probabilities = np.exp(logits - logits.max())
    return probabilities / probabilities.sum()


def exponential_mechanism(scores: Array, eps: float, sensitivity: float, rng: np.random.Generator) -> int:
    """Choose index i with probability proportional to exp(eps * score_i / (2 * sensitivity))."""
    return int(rng.choice(len(scores), p=selection_probabilities(scores, eps, sensitivity)))


# excerpt: anneal
def anneal(
    eps: float, sigma: float, change: float, penalty: float, remaining: float
) -> tuple[float, float, bool, bool]:
    """The next round's (eps, sigma). If the last measurement moved the model by less
    than the noise it carried, double eps and halve sigma. Then, if what is left cannot
    pay for two such rounds, make the next round the last one and let it spend exactly
    the remaining budget. Also returns whether it annealed and whether it is the last."""
    annealed = change <= penalty
    if annealed:
        eps, sigma = 2 * eps, sigma / 2
    last = remaining <= 2 * round_cost(eps, sigma)
    if last:
        eps = math.sqrt(8 * (1 - MEASURE_SHARE) * remaining)
        sigma = math.sqrt(1 / (2 * MEASURE_SHARE * remaining))
    return eps, sigma, annealed, last


# end excerpt


@dataclass
class Run:
    rho: float
    sigma0: float
    eps1: float
    init_cost: float
    init_model: Array
    first_scores: list[
        tuple[Attrs, float, float, float, float]
    ]  # (candidate, model error, penalty, score, probability)
    rounds: list[Round]
    model: Array
    synthetic: Array


def aim(data: Array, workload: list[Attrs], weights: dict[Attrs, float], rho: float, seed: int) -> Run:
    rng = np.random.default_rng(seed)
    candidates = downward_closure(workload)
    d = len(SIZES)
    planned = ROUNDS_PER_ATTRIBUTE * d
    sigma0 = math.sqrt(planned / (2 * MEASURE_SHARE * rho))
    eps1 = math.sqrt(8 * (1 - MEASURE_SHARE) * rho / planned)

    # Initialization: every one-way marginal, with the starting noise.
    measurements: list[Measurement] = []
    for attrs in candidates:
        if len(attrs) == 1:
            noisy = marginal(data, attrs) + rng.normal(0.0, sigma0, cells(attrs))
            measurements.append(Measurement(attrs, noisy, sigma0))
    used = len(measurements) / (2 * sigma0**2)
    init_cost = used
    # Under add/remove-one-record privacy the number of records is itself private, so the model's total is
    # estimated from the noisy one-way marginals, each of which sums to a noisy copy of it.
    total = float(np.mean([m.noisy.sum() for m in measurements]))
    model = fit(measurements, total)
    init_model = model

    cand_weight = {r: candidate_weight(r, workload, weights) for r in candidates}
    sensitivity = max(cand_weight.values())
    eps, sigma = eps1, sigma0
    rounds: list[Round] = []
    first_scores: list[tuple[Attrs, float, float, float, float]] = []
    t = 0
    while used < rho:
        t += 1
        used += round_cost(eps, sigma)
        scores = np.array([score(data, model, r, cand_weight[r], sigma) for r in candidates])
        if t == 1:
            probabilities = selection_probabilities(scores, eps, sensitivity)
            for r, s, pr in zip(candidates, scores, probabilities, strict=True):
                err = float(np.abs(marginal(data, r) - marginal(model, r)).sum())
                first_scores.append((r, err, SQRT_2_OVER_PI * sigma * cells(r), float(s), float(pr)))
        chosen = candidates[exponential_mechanism(scores, eps, sensitivity, rng)]
        before = marginal(model, chosen)
        model_error = float(np.abs(marginal(data, chosen) - before).sum())
        noisy = marginal(data, chosen) + rng.normal(0.0, sigma, cells(chosen))
        measurements.append(Measurement(chosen, noisy, sigma))
        model = fit(measurements, total)
        change = float(np.abs(marginal(model, chosen) - before).sum())
        penalty = SQRT_2_OVER_PI * sigma * cells(chosen)
        annealed = change <= penalty
        if used < rho:  # the final round leaves nothing to anneal
            next_eps, next_sigma, annealed, _ = anneal(eps, sigma, change, penalty, rho - used)
        else:
            next_eps, next_sigma = eps, sigma
        rounds.append(
            Round(
                t=t,
                attrs=chosen,
                sigma=sigma,
                eps=eps,
                cost=round_cost(eps, sigma),
                model_error=model_error,
                penalty=penalty,
                change=change,
                annealed=annealed,
                workload_error=workload_error(data, model, workload, weights, N),
            )
        )
        eps, sigma = next_eps, next_sigma
    assert abs(used - rho) < 1e-9, "the final round spends exactly what is left"
    synthetic = round_to_records(model)
    return Run(rho, sigma0, eps1, init_cost, init_model, first_scores, rounds, model, synthetic)


def measure_everything(
    data: Array, workload: list[Attrs], weights: dict[Attrs, float], rho: float, seed: int
) -> tuple[float, float]:
    """The plain alternative to AIM: measure every workload marginal once, with the whole budget split evenly, and
    fit the model to all of them. Returns the noise's standard deviation and the model's workload error."""
    rng = np.random.default_rng(seed)
    sigma = math.sqrt(len(workload) / (2 * rho))
    measurements = [Measurement(s, marginal(data, s) + rng.normal(0.0, sigma, cells(s)), sigma) for s in workload]
    total = float(np.mean([m.noisy.sum() for m in measurements]))
    model = fit(measurements, total)
    return sigma, workload_error(data, model, workload, weights, N)


# --- Figures and tables -----------------------------------------------------------------------------------------------


def two_way_table(vector: Array, attrs: Attrs, caption: str, label: str, decimals: int = 0) -> str:
    """A two-way marginal as a Markdown table with the first attribute down the rows."""
    a, b = attrs
    grid = vector.reshape(SIZES[a], SIZES[b])
    lines = [f"| {ATTRIBUTES[a]} \\ {ATTRIBUTES[b]} | " + " | ".join(VALUES[b]) + " |", "|:---|" + "---:|" * SIZES[b]]
    for i, row in enumerate(VALUES[a]):
        lines.append(f"| {row} | " + " | ".join(f"{grid[i, j]:,.{decimals}f}" for j in range(SIZES[b])) + " |")
    return "\n".join(lines) + f"\n\n: {caption} {{#{label}}}\n"


def marginal_snippet(data: Array) -> str:
    attrs: Attrs = (0, 3)
    return two_way_table(
        marginal(data, attrs),
        attrs,
        "The marginal of the toy dataset over smoker and condition: how many of its 1,000 people fall in each of "
        "the four cells. Its one-way marginals are the row and column sums.",
        "tbl-marginal",
    )


def init_snippet(data: Array, init_model: Array) -> str:
    """The true marginal over smoker and condition beside the one the model fitted to the one-way marginals gives."""
    attrs: Attrs = (0, 3)
    true = marginal(data, attrs).reshape(2, 2)
    model = marginal(init_model, attrs).reshape(2, 2)
    lines = [
        "| smoker | true: no condition | true: condition | model: no condition | model: condition |",
        "|:---|---:|---:|---:|---:|",
    ]
    for i, row in enumerate(VALUES[0]):
        lines.append(f"| {row} | {true[i, 0]:,.0f} | {true[i, 1]:,.0f} | {model[i, 0]:,.1f} | {model[i, 1]:,.1f} |")
    lines.append("")
    lines.append(
        ": The true marginal over smoker and condition, and what the model fitted to the noisy one-way marginals "
        "alone gives for it. The model keeps the row and column sums, within the noise, but it spreads each row in "
        "the same proportions, because nothing it has measured says that smoking and the condition go together. "
        "{#tbl-init}"
    )
    return "\n".join(lines) + "\n"


def scores_snippet(run: Run, cand_weight: dict[Attrs, float]) -> str:
    lines = [
        "| Candidate | $w_r$ | Model error on it | Noise penalty | Score | Chance of being chosen |",
        "|:---|---:|---:|---:|---:|---:|",
    ]
    for r, err, pen, sc, pr in run.first_scores:
        lines.append(f"| {name(r)} | {cand_weight[r]:.0f} | {err:.0f} | {pen:.0f} | {sc:.0f} | {pr:.0%} |")
    lines.append("")
    lines.append(
        ": The first round's selection on the toy dataset. The model error is the $L_1$ distance between the "
        "true marginal and the model's, the noise penalty is $\\sqrt{2/\\pi}\\,\\sigma_1$ times the marginal's cells, "
        "the score is the weight times their difference, and the last column is the exponential mechanism's "
        "probability of choosing that candidate at the first round's $\\varepsilon_1$. {#tbl-scores}"
    )
    return "\n".join(lines) + "\n"


def schedule_snippet(run: Run) -> str:
    d = len(SIZES)
    planned = ROUNDS_PER_ATTRIBUTE * d
    rows = [
        ("budget $\\rho$, from $(\\varepsilon, \\delta) = (1, 10^{-6})$", f"{run.rho:.4f}"),
        ("rounds planned, $T = 16d$", f"{planned}"),
        ("starting noise $\\sigma_0 = \\sqrt{T/(2 \\cdot 0.9\\rho)}$", f"{run.sigma0:.1f}"),
        ("starting selection budget $\\varepsilon_1 = \\sqrt{8 \\cdot 0.1\\rho/T}$", f"{run.eps1:.4f}"),
        ("cost of the initialization, $d/(2\\sigma_0^2)$", f"{run.init_cost:.4f}"),
        (
            "cost of one round at the start, $\\varepsilon_1^2/8 + 1/(2\\sigma_0^2)$",
            f"{round_cost(run.eps1, run.sigma0):.4f}",
        ),
        ("cost of a round after one annealing", f"{round_cost(2 * run.eps1, run.sigma0 / 2):.4f}"),
        ("cost of a round after two annealings", f"{round_cost(4 * run.eps1, run.sigma0 / 4):.4f}"),
    ]
    lines = ["| Quantity | Value |", "|:---|---:|"] + [f"| {k} | {v} |" for k, v in rows]
    lines.append("")
    lines.append(
        ": AIM's budget schedule on the toy dataset, with the paper's defaults. Each annealing doubles "
        "$\\varepsilon$ and halves $\\sigma$, so it multiplies the cost of a round by four. {#tbl-schedule}"
    )
    return "\n".join(lines) + "\n"


def rounds_snippet(run: Run) -> str:
    lines = [
        "| Round | Marginal chosen | $\\sigma_t$ | $\\varepsilon_t$ | Cost | Model error on it | Noise penalty | Change | Annealed | Workload error after |",
        "|---:|:---|---:|---:|---:|---:|---:|---:|:---|---:|",
    ]
    for r in run.rounds:
        lines.append(
            f"| {r.t} | {name(r.attrs)} | {r.sigma:.1f} | {r.eps:.3f} | {r.cost:.4f} | {r.model_error:.0f} | "
            f"{r.penalty:.0f} | {r.change:.0f} | {'yes' if r.annealed else ''} | {r.workload_error:.4f} |"
        )
    lines.append("")
    lines.append(
        ": Every round of AIM on the toy dataset. The model error is the $L_1$ distance between the true marginal "
        "and the model's before the measurement, the noise penalty is $\\sqrt{2/\\pi}\\,\\sigma_t$ times the "
        "marginal's cells, and the change is how far the model moved on that marginal after refitting. A round "
        "anneals when the change is at most the penalty. The workload error is the paper's metric, the "
        "weighted $L_1$ error over the workload divided by the number of marginals and by $n$. {#tbl-rounds}"
    )
    return "\n".join(lines) + "\n"


def figure_scores(run: Run, out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    labels = [name(r) for r, *_ in run.first_scores]
    errors = [e for _, e, *_ in run.first_scores]
    penalties = [p for _, _, p, *_ in run.first_scores]
    x = np.arange(len(labels))
    ax.bar(x - 0.2, errors, width=0.4, color="#1f77b4", label="model error on the marginal, $L_1$")
    ax.bar(x + 0.2, penalties, width=0.4, color="#bbbbbb", label=r"noise penalty $\sqrt{2/\pi}\,\sigma_1 \times$ cells")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=35, ha="right", fontsize=8)
    ax.set_ylabel("records")
    ax.set_title("Round 1: what each candidate would gain against what its noise would cost", fontsize=10)
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(True, axis="y", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "scores.svg")
    plt.close(fig)


def figure_rounds(run: Run, data: Array, workload: list[Attrs], weights: dict[Attrs, float], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    ts = [0] + [r.t for r in run.rounds]
    errors = [workload_error(data, run.init_model, workload, weights, N)] + [r.workload_error for r in run.rounds]
    ax.plot(ts, errors, marker="o", markersize=4, color="#1f77b4", label="workload error of the model")
    annealed = [r.t for r in run.rounds if r.annealed]
    ax.plot(
        annealed,
        [errors[t] for t in annealed],
        linestyle="none",
        marker="o",
        markersize=9,
        markerfacecolor="none",
        markeredgecolor="#d62728",
        label="round that annealed (next round pays 4× more)",
    )
    ax.axhline(
        workload_error(data, run.synthetic, workload, weights, N),
        color="grey",
        linestyle=":",
        linewidth=1,
        label="error of the synthetic dataset rounded from the final model",
    )
    ax.set_xlabel("round (0 is the model fitted to the one-way marginals)")
    ax.set_ylabel("workload error")
    ax.set_title("AIM on the toy dataset: the model's workload error, round by round", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "rounds.svg")
    plt.close(fig)


# --- Main ----------------------------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=FIGURES)
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    records = draw_records(N, seed=0)
    data = full_table(records)
    workload = [(i, j) for i in range(len(SIZES)) for j in range(i + 1, len(SIZES))]
    weights = {s: 1.0 for s in workload}
    candidates = downward_closure(workload)

    print(f"== The toy dataset: n={N}  cells={cells(tuple(range(len(SIZES))))}  attributes={len(SIZES)}")
    smoker_condition = marginal(data, (0, 3)).reshape(2, 2)
    print(f"smoker x condition: {smoker_condition.astype(int).tolist()}")
    print(
        f"condition rate: non-smokers={smoker_condition[0, 1] / smoker_condition[0].sum():.3f}  "
        f"smokers={smoker_condition[1, 1] / smoker_condition[1].sum():.3f}"
    )
    print(f"workload marginals={len(workload)}  candidates={len(candidates)}")
    cand_weight = {r: candidate_weight(r, workload, weights) for r in candidates}
    print(
        f"weights: one-way={cand_weight[(0,)]:.0f}  two-way={cand_weight[(0, 3)]:.0f}  sensitivity={max(cand_weight.values()):.0f}"
    )
    print(f"sqrt(2/pi)={SQRT_2_OVER_PI:.3f}")

    rho = rho_for(EPSILON, DELTA)
    print(f"== Budget: rho={rho:.4f} for eps={EPSILON} delta={DELTA:.0e}  (delta back={delta_of(rho, EPSILON):.2e})")
    run = aim(data, workload, weights, rho, seed=1)
    print(
        f"T={ROUNDS_PER_ATTRIBUTE * len(SIZES)}  sigma0={run.sigma0:.1f}  eps1={run.eps1:.4f}  init cost={run.init_cost:.4f}  "
        f"round cost={round_cost(run.eps1, run.sigma0):.4f}"
    )
    print(f"total estimated from the noisy one-way marginals: {np.mean([m for m in [run.init_model.sum()]]):.1f}")
    init_sc = marginal(run.init_model, (0, 3)).reshape(2, 2)
    print(f"init model smoker x condition: {np.round(init_sc, 1).tolist()}")
    print(f"init workload error={workload_error(data, run.init_model, workload, weights, N):.4f}")
    print("== Round 1 scores")
    for r, err, pen, sc, pr in run.first_scores:
        print(f"  {name(r):22s} error={err:7.1f}  penalty={pen:7.1f}  score={sc:8.1f}  probability={pr:.3f}")
    print("== Rounds")
    for r in run.rounds:
        print(
            f"  t={r.t} {name(r.attrs):22s} sigma={r.sigma:.1f} eps={r.eps:.3f} cost={r.cost:.4f} "
            f"error={r.model_error:.0f} penalty={r.penalty:.0f} change={r.change:.0f} "
            f"annealed={'yes' if r.annealed else 'no'} workload error={r.workload_error:.4f}"
        )
    print(f"rounds={len(run.rounds)}  annealings={sum(r.annealed for r in run.rounds)}")
    print(
        f"final model workload error={run.rounds[-1].workload_error:.4f}  "
        f"synthetic dataset workload error={workload_error(data, run.synthetic, workload, weights, N):.4f}  "
        f"synthetic records={run.synthetic.sum():.0f}"
    )
    naive_sigma, naive_error = measure_everything(data, workload, weights, rho, seed=2)
    print(f"measure everything: sigma={naive_sigma:.1f}  workload error={naive_error:.4f}")
    adult_3way = math.comb(15, 3)
    print(
        f"all 3-way marginals of 15 attributes: {adult_3way}  sigma if all measured={math.sqrt(adult_3way / (2 * rho)):.1f}"
    )
    final_sc = marginal(run.model, (0, 3)).reshape(2, 2)
    print(f"final model smoker x condition: {np.round(final_sc, 1).tolist()}")

    (out / "_marginal.md").write_text(marginal_snippet(data))
    (out / "_init.md").write_text(init_snippet(data, run.init_model))
    (out / "_scores.md").write_text(scores_snippet(run, cand_weight))
    (out / "_schedule.md").write_text(schedule_snippet(run))
    (out / "_rounds.md").write_text(rounds_snippet(run))
    figure_scores(run, out)
    figure_rounds(run, data, workload, weights, out)


if __name__ == "__main__":
    main()
