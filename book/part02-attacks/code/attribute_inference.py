"""Check the attribute inference and model inversion chapter and draw its figures.

    uv run python book/part02-attacks/code/attribute_inference.py
    uv run python book/part02-attacks/code/attribute_inference.py --seed 1 --out /tmp/figures

Prints every number quoted in 03-attribute-inference.md (they match exactly with the default
seed), then saves the chapter's figures.

Two worked examples:

- The smoking table: a sensitive bit s (smoker), a public feature x (age band) and an outcome y
  (disease), given by an exact joint table J[s, x, y]. Everything about it is computed exactly.
- The toy model: s ~ Bernoulli(0.3); x in R^d with x_1 shifted by `shift` when s = 1; y drawn from
  a logistic model whose weights depend on s. A logistic regression with one weight vector per
  value of s is trained on n records, and the attacker, who knows (x, y) of a target record and
  the population law of (s, x), guesses s from the model's confidences (MAP model inversion).
  Its success on training records (members) and fresh records (non-members) is compared with the
  Bayes imputation baseline, which knows the true population law of (s, x, y) and no model.
"""

import argparse
import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
matplotlib.rcParams["svg.hashsalt"] = "attribute_inference"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.figure import Figure
from numpy.typing import NDArray

FIGURES = Path(__file__).resolve().parent.parent / "figures"

Array = NDArray[np.float64]

# --- The smoking table ----------------------------------------------------------------------
# s: 0 = non-smoker, 1 = smoker. x: 0 = young, 1 = old. y: 0 = no disease, 1 = disease.

SMOKER_PRIOR = 0.3
P_OLD_GIVEN_S = (0.4, 0.6)
P_DISEASE = ((0.1, 0.3), (0.4, 0.7))  # P_DISEASE[s][x] = Pr[y = 1 | x, s]


def smoking_joint(
    prior: float = SMOKER_PRIOR,
    p_old: tuple[float, float] = P_OLD_GIVEN_S,
    p_disease: tuple[tuple[float, float], tuple[float, float]] = P_DISEASE,
) -> Array:
    """J[s, x, y] = π(s) p(x | s) p(y | x, s)."""
    joint = np.zeros((2, 2, 2))
    for s in (0, 1):
        ps = prior if s == 1 else 1 - prior
        for x in (0, 1):
            px = p_old[s] if x == 1 else 1 - p_old[s]
            for y in (0, 1):
                py = p_disease[s][x] if y == 1 else 1 - p_disease[s][x]
                joint[s, x, y] = ps * px * py
    return joint


def prior_vulnerability(joint: Array) -> float:
    """V(S) = max_s Pr[s]: the best blind guess."""
    return float(joint.sum(axis=(1, 2)).max())


def vulnerability_given_x(joint: Array) -> float:
    """V(S | X) = Σ_x max_s Pr[s, x]: the best guess from the public feature alone."""
    return float(joint.sum(axis=2).max(axis=0).sum())


def vulnerability_given_xy(joint: Array) -> float:
    """V(S | X, Y) = Σ_{x,y} max_s Pr[s, x, y]: the imputation baseline."""
    return float(joint.max(axis=0).sum())


def posterior_smoker(joint: Array) -> Array:
    """Pr[s = 1 | x, y] as a 2×2 array indexed [x, y]."""
    return joint[1] / joint.sum(axis=0)


def leakage_bits(v_after: float, v_before: float) -> float:
    """Min-entropy leakage log2(V_after / V_before)."""
    return math.log2(v_after / v_before)


# --- The toy population and model ------------------------------------------------------------

DIM = 10


@dataclass(frozen=True)
class Population:
    """s ~ Bern(prior); x ~ N(shift·s·e_1, I_d); y ~ Bern(σ(w_s·x + b_s))."""

    prior: float
    shift: float
    weights: Array  # shape (2, d): row s is w_s
    biases: Array  # shape (2,)

    @property
    def dim(self) -> int:
        return int(self.weights.shape[1])


def default_population(dim: int = DIM) -> Population:
    """Smokers (s = 1) have a higher base rate and depend on different features than non-smokers."""
    w0 = np.zeros(dim)
    w1 = np.zeros(dim)
    w0[:4] = 1.0
    w1[:2] = 1.0
    w1[4:6] = 1.0
    return Population(prior=0.3, shift=0.5, weights=np.stack([w0, w1]), biases=np.array([-1.0, 0.5]))


def sigmoid(z: Array) -> Array:
    return 1.0 / (1.0 + np.exp(-z))


@dataclass(frozen=True)
class Records:
    s: NDArray[np.int64]
    x: Array
    y: NDArray[np.int64]


def sample_records(rng: np.random.Generator, pop: Population, n: int) -> Records:
    s = (rng.random(n) < pop.prior).astype(np.int64)
    x = rng.standard_normal((n, pop.dim))
    x[:, 0] += pop.shift * s
    logits = np.einsum("ij,ij->i", x, pop.weights[s]) + pop.biases[s]
    y = (rng.random(n) < sigmoid(logits)).astype(np.int64)
    return Records(s=s, x=x, y=y)


def prior_given_x(pop: Population, x: Array) -> Array:
    """Pr[s = 1 | x] under the population: only x_1 carries information about s."""
    llr = pop.shift * x[:, 0] - pop.shift**2 / 2  # ln N(x_1; shift, 1) - ln N(x_1; 0, 1)
    log_odds = math.log(pop.prior / (1 - pop.prior)) + llr
    return sigmoid(log_odds)


def true_confidence(pop: Population, x: Array, s: int) -> Array:
    """Pr[y = 1 | x, s] under the population."""
    return sigmoid(x @ pop.weights[s] + pop.biases[s])


def features(x: Array, s: NDArray[np.int64] | int) -> Array:
    """φ(x, s) = [(1-s)·(x, 1), s·(x, 1)]: one logistic model per value of s."""
    n = x.shape[0]
    xb = np.hstack([x, np.ones((n, 1))])
    s_col = (np.zeros(n) + s)[:, None]
    return np.hstack([(1 - s_col) * xb, s_col * xb])


def train_logistic(phi: Array, y: NDArray[np.int64], ridge: float, steps: int = 30) -> Array:
    """Minimize Σ log-loss + (ridge/2)‖θ‖² by Newton's method."""
    theta = np.zeros(phi.shape[1])
    for _ in range(steps):
        p = sigmoid(phi @ theta)
        grad = phi.T @ (p - y) + ridge * theta
        hess = (phi * (p * (1 - p))[:, None]).T @ phi + ridge * np.eye(phi.shape[1])
        step = np.linalg.solve(hess, grad)
        theta -= step
        if float(np.max(np.abs(step))) < 1e-10:
            break
    return theta


RIDGE = 0.1


@dataclass(frozen=True)
class Model:
    theta: Array

    def confidence(self, x: Array, s: int) -> Array:
        """f(x, s) = model's Pr[y = 1 | x, s]."""
        return sigmoid(features(x, s) @ self.theta)


def train_model(train: Records, ridge: float = RIDGE) -> Model:
    return Model(train_logistic(features(train.x, train.s), train.y, ridge))


def mean_log_loss(model: Model, data: Records) -> float:
    p = np.empty(len(data.y))
    for s in (0, 1):
        mask = data.s == s
        p[mask] = model.confidence(data.x[mask], s)
    py = np.where(data.y == 1, p, 1 - p)
    return float(np.mean(-np.log(py)))


# --- Attacks ---------------------------------------------------------------------------------


def label_confidence(conf_y1: Array, y: NDArray[np.int64]) -> Array:
    """f_y: the confidence the model gives to the observed label y."""
    return np.where(y == 1, conf_y1, 1 - conf_y1)


def map_guess(prior1: Array, like0: Array, like1: Array) -> NDArray[np.int64]:
    """argmax_s Pr[s | x]·like_s, with ties broken toward the more likely value under the prior."""
    score1 = prior1 * like1
    score0 = (1 - prior1) * like0
    tie = score1 == score0
    return np.where(tie, prior1 > 0.5, score1 > score0).astype(np.int64)


def mask_confidence(conf: Array, mode: str) -> Array:
    """What the API releases: the full confidence, rounded to one decimal, or only the label."""
    if mode == "full":
        return conf
    if mode == "rounded":
        return np.round(conf, 1)
    if mode == "label":
        return (conf > 0.5).astype(np.float64)
    raise ValueError(mode)


def inversion_guess(model: Model, pop: Population, data: Records, mode: str = "full") -> NDArray[np.int64]:
    """MAP model inversion: argmax_s Pr[s | x]·f_y(x, s), querying the model at both values of s."""
    like = [label_confidence(mask_confidence(model.confidence(data.x, s), mode), data.y) for s in (0, 1)]
    return map_guess(prior_given_x(pop, data.x), like[0], like[1])


def imputation_guess(pop: Population, data: Records) -> NDArray[np.int64]:
    """Bayes imputation: argmax_s Pr[s | x, y] under the true population. Uses no model."""
    like = [label_confidence(true_confidence(pop, data.x, s), data.y) for s in (0, 1)]
    return map_guess(prior_given_x(pop, data.x), like[0], like[1])


def prior_guess(pop: Population, data: Records) -> NDArray[np.int64]:
    """Guess from the public features alone: argmax_s Pr[s | x]."""
    return (prior_given_x(pop, data.x) > 0.5).astype(np.int64)


def accuracy(guess: NDArray[np.int64], truth: NDArray[np.int64]) -> float:
    right: NDArray[np.bool_] = np.equal(guess, truth)
    return float(np.mean(right))


def inversion_evidence(model: Model, pop: Population, data: Records) -> Array:
    """ln of the MAP score of the true value over the other: positive means the attack is right."""
    prior1 = prior_given_x(pop, data.x)
    like = [label_confidence(model.confidence(data.x, s), data.y) for s in (0, 1)]
    log_odds1 = np.log(prior1) + np.log(like[1]) - np.log(1 - prior1) - np.log(like[0])
    return np.where(data.s == 1, log_odds1, -log_odds1)


# --- The attribute inference game ------------------------------------------------------------


@dataclass(frozen=True)
class GameResult:
    """Average success over many runs of the game, on members and on non-members."""

    n: int
    trials: int
    inversion_members: float
    inversion_nonmembers: float
    imputation_members: float
    imputation_nonmembers: float
    prior_members: float
    prior_nonmembers: float
    masked_members: dict[str, float]
    masked_nonmembers: dict[str, float]
    loss_members: float
    loss_nonmembers: float
    advantage_se: float  # standard error of the advantage, from its spread over the trained models

    @property
    def advantage(self) -> float:
        """Yeom et al.'s attribute advantage: success on members minus success on non-members."""
        return self.inversion_members - self.inversion_nonmembers


MASKS = ("full", "rounded", "label")


def play_game(rng: np.random.Generator, pop: Population, n: int, trials: int) -> GameResult:
    """Train `trials` models on n records each; attack the n members and n fresh non-members."""
    acc: dict[str, list[float]] = {}

    def add(key: str, value: float) -> None:
        acc.setdefault(key, []).append(value)

    for _ in range(trials):
        members = sample_records(rng, pop, n)
        others = sample_records(rng, pop, n)
        model = train_model(members)
        for tag, data in (("m", members), ("o", others)):
            add(f"imp_{tag}", accuracy(imputation_guess(pop, data), data.s))
            add(f"prior_{tag}", accuracy(prior_guess(pop, data), data.s))
            add(f"loss_{tag}", mean_log_loss(model, data))
            for mode in MASKS:
                add(f"{mode}_{tag}", accuracy(inversion_guess(model, pop, data, mode), data.s))
    mean = {k: float(np.mean(v)) for k, v in acc.items()}
    per_model = np.array(acc["full_m"]) - np.array(acc["full_o"])
    se = float(np.std(per_model, ddof=1) / math.sqrt(trials)) if trials > 1 else math.nan
    return GameResult(
        n=n,
        trials=trials,
        inversion_members=mean["full_m"],
        inversion_nonmembers=mean["full_o"],
        imputation_members=mean["imp_m"],
        imputation_nonmembers=mean["imp_o"],
        prior_members=mean["prior_m"],
        prior_nonmembers=mean["prior_o"],
        masked_members={mode: mean[f"{mode}_m"] for mode in MASKS},
        masked_nonmembers={mode: mean[f"{mode}_o"] for mode in MASKS},
        loss_members=mean["loss_m"],
        loss_nonmembers=mean["loss_o"],
        advantage_se=se,
    )


def population_vulnerability(rng: np.random.Generator, pop: Population, m: int) -> tuple[float, float]:
    """Monte Carlo V(S | X) and V(S | X, Y) on m fresh records (no model involved)."""
    data = sample_records(rng, pop, m)
    return accuracy(prior_guess(pop, data), data.s), accuracy(imputation_guess(pop, data), data.s)


# --- The chapter's numbers ------------------------------------------------------------------

N_TRAIN = 100
TRIALS = 200
SIZES = (25, 50, 100, 200, 400, 800, 1600)


def print_table(rng: np.random.Generator) -> None:
    joint = smoking_joint()
    post = posterior_smoker(joint)
    v0, vx, vxy = prior_vulnerability(joint), vulnerability_given_x(joint), vulnerability_given_xy(joint)
    print("[Smoking table: J[s, x, y]]")
    for s in (0, 1):
        cells = "  ".join(f"x={x},y={y}:{joint[s, x, y]:.3f}" for x in (0, 1) for y in (0, 1))
        print(f"  s={s}: {cells}")
    cells = "  ".join(f"x={x},y={y}:{post[x, y]:.4f}" for x in (0, 1) for y in (0, 1))
    print(f"  Pr[smoker | x, y]: {cells}")
    print(f"  V(S)={v0:.4f}  V(S|X)={vx:.4f}  V(S|X,Y)={vxy:.4f}  leakage={leakage_bits(vxy, v0):.4f} bits")

    pop = default_population()
    v_x, v_xy = population_vulnerability(rng, pop, 1_000_000)
    print(f"[Toy population, d={pop.dim}, prior={pop.prior:g}, shift={pop.shift:g}; 10^6 fresh records]")
    print(f"  prior guess Pr[s|x]: {v_x:.4f}  Bayes imputation Pr[s|x,y]: {v_xy:.4f}")

    g = play_game(rng, pop, N_TRAIN, TRIALS)
    print(f"[Game: n={g.n} training records, {g.trials} trained models, ridge={RIDGE:g}]")
    print(
        f"  mean log-loss: members={g.loss_members:.4f}  non-members={g.loss_nonmembers:.4f}  "
        f"gap={g.loss_nonmembers - g.loss_members:.4f}"
    )
    print(f"  prior guess:      members={g.prior_members:.4f}  non-members={g.prior_nonmembers:.4f}")
    print(
        f"  imputation:       members={g.imputation_members:.4f}  non-members={g.imputation_nonmembers:.4f}  "
        f"advantage={g.imputation_members - g.imputation_nonmembers:+.4f}"
    )
    print(
        f"  MAP inversion:    members={g.inversion_members:.4f}  non-members={g.inversion_nonmembers:.4f}  "
        f"advantage={g.advantage:+.4f} (standard error {g.advantage_se:.4f})"
    )
    print(
        f"  members above the Bayes imputation baseline: {g.inversion_members - v_xy:+.4f}  "
        f"non-members below it: {g.inversion_nonmembers - v_xy:+.4f}"
    )
    print(
        f"  as a membership test: TPR={g.inversion_members:.4f}  FPR={g.inversion_nonmembers:.4f}  "
        f"balanced accuracy={0.5 + 0.5 * g.advantage:.4f}"
    )
    for mode in MASKS:
        m, o = g.masked_members[mode], g.masked_nonmembers[mode]
        print(f"  masking={mode:<8} members={m:.4f}  non-members={o:.4f}  advantage={m - o:+.4f}")

    print("[Success against training-set size n (40 models each)]")
    for n in (25, 400, 1600):
        gn = play_game(rng, pop, n, 40)
        print(
            f"  n={n}: members={gn.inversion_members:.4f}  non-members={gn.inversion_nonmembers:.4f}  "
            f"advantage={gn.advantage:+.4f}"
        )


# --- Figures --------------------------------------------------------------------------------


def plot_smoking_posterior() -> Figure:
    joint = smoking_joint()
    post = posterior_smoker(joint)
    labels = ["young,\nno disease", "young,\ndisease", "old,\nno disease", "old,\ndisease"]
    values = [float(post[x, y]) for x in (0, 1) for y in (0, 1)]
    weights = [float(joint[:, x, y].sum()) for x in (0, 1) for y in (0, 1)]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    colors = ["C3" if v > 0.5 else "C0" for v in values]
    ax.bar(range(4), values, color=colors)
    for i, (v, w) in enumerate(zip(values, weights, strict=True)):
        ax.text(i, v + 0.02, f"{v:.2f}\n(Pr = {w:.2f})", ha="center", fontsize="small")
    ax.axhline(SMOKER_PRIOR, color="gray", linestyle=":", label=f"prior {SMOKER_PRIOR:g}")
    ax.axhline(0.5, color="black", linestyle="--", linewidth=1, label="guess 'smoker' above 0.5")
    ax.set_xticks(range(4), labels)
    ax.set(
        ylim=(0, 0.85),
        ylabel=r"$\Pr[\mathrm{smoker} \mid x, y]$",
        title="Imputation from the population: disease points to smoking",
    )
    ax.legend(loc="upper left", fontsize="small")
    return fig


def plot_inversion_evidence(rng: np.random.Generator, n: int = N_TRAIN, models: int = 50) -> Figure:
    pop = default_population()
    member_ev: list[Array] = []
    other_ev: list[Array] = []
    for _ in range(models):
        members = sample_records(rng, pop, n)
        others = sample_records(rng, pop, n)
        model = train_model(members)
        member_ev.append(inversion_evidence(model, pop, members))
        other_ev.append(inversion_evidence(model, pop, others))
    m, o = np.concatenate(member_ev), np.concatenate(other_ev)
    bins = [float(b) for b in np.linspace(-6, 6, 49)]
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.hist(o, bins=bins, density=True, alpha=0.55, color="C0", label=f"non-members: {np.mean(o > 0):.2f} right")
    ax.hist(m, bins=bins, density=True, alpha=0.55, color="C3", label=f"members: {np.mean(m > 0):.2f} right")
    ax.axvline(0, color="black", linewidth=1)
    ax.set(
        xlabel="log score of the true value over the other (attack is right if > 0)",
        ylabel="density",
        title=f"MAP inversion evidence, {models} models trained on n = {n}",
    )
    ax.legend(loc="upper left", fontsize="small")
    return fig


def plot_success_vs_n(rng: np.random.Generator, sizes: tuple[int, ...] = SIZES, trials: int = 40) -> Figure:
    pop = default_population()
    results = [play_game(rng, pop, n, trials) for n in sizes]
    v_x, v_xy = population_vulnerability(rng, pop, 400_000)
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.plot(sizes, [r.inversion_members for r in results], "o-", color="C3", label="inversion, members")
    ax.plot(sizes, [r.inversion_nonmembers for r in results], "s-", color="C0", label="inversion, non-members")
    ax.axhline(v_xy, color="black", linestyle="--", linewidth=1, label=f"Bayes imputation {v_xy:.3f}")
    ax.axhline(v_x, color="gray", linestyle=":", label=f"guess from x alone {v_x:.3f}")
    ax.set_xscale("log")
    ax.set_xticks(list(sizes), [str(n) for n in sizes])
    ax.set(
        xlabel="training-set size n",
        ylabel="fraction of attributes guessed right",
        title="Members leak more when the model can memorize",
    )
    ax.legend(loc="lower right", fontsize="small")
    return fig


def plot_masking(rng: np.random.Generator, n: int = N_TRAIN, trials: int = TRIALS) -> Figure:
    pop = default_population()
    g = play_game(rng, pop, n, trials)
    names = {"full": "full confidences", "rounded": "rounded to 0.1", "label": "label only"}
    xs = np.arange(len(MASKS))
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.bar(xs - 0.2, [g.masked_members[m] for m in MASKS], width=0.4, color="C3", label="members")
    ax.bar(xs + 0.2, [g.masked_nonmembers[m] for m in MASKS], width=0.4, color="C0", label="non-members")
    for i, mode in enumerate(MASKS):
        ax.text(i - 0.2, g.masked_members[mode] + 0.005, f"{g.masked_members[mode]:.3f}", ha="center", fontsize="small")
        ax.text(
            i + 0.2,
            g.masked_nonmembers[mode] + 0.005,
            f"{g.masked_nonmembers[mode]:.3f}",
            ha="center",
            fontsize="small",
        )
    ax.axhline(g.imputation_nonmembers, color="black", linestyle="--", linewidth=1, label="Bayes imputation")
    ax.axhline(g.prior_nonmembers, color="gray", linestyle=":", label="guess from x alone")
    ax.set_xticks(xs, [names[m] for m in MASKS])
    ax.set(ylim=(0.6, 0.9), ylabel="fraction of attributes guessed right", title=f"Confidence masking, n = {n}")
    ax.legend(loc="upper right", fontsize="small", ncols=2)
    return fig


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--out", type=Path, default=FIGURES, help="directory for the figures")
    args = parser.parse_args()
    seed: int = args.seed
    out: Path = args.out

    print_table(np.random.default_rng(seed))

    rng = np.random.default_rng(seed + 1)  # separate stream, so the table stays reproducible
    out.mkdir(parents=True, exist_ok=True)
    figures = {
        "attr_smoking_posterior.svg": plot_smoking_posterior(),
        "attr_inversion_evidence.svg": plot_inversion_evidence(rng),
        "attr_success_vs_n.svg": plot_success_vs_n(rng),
        "attr_masking.svg": plot_masking(rng),
    }
    for name, fig in figures.items():
        fig.tight_layout()
        fig.savefig(out / name, metadata={"Date": None})  # no timestamp, so output is reproducible
        plt.close(fig)
        print(f"saved {out / name}")


if __name__ == "__main__":
    main()
