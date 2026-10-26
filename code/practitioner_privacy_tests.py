"""What no maintained library provides: the LiRA score, the audit lower bounds and canary exposure.

The practitioner edition runs its attacks on maintained libraries where they exist: the Adversarial Robustness
Toolbox (ART) trains shadow models and runs the black-box membership and attribute inference attacks, scikit-learn
reads the ROC curve, and SciPy's `binomtest(...).proportion_ci(method="exact")` gives the Clopper-Pearson intervals.
The functions here are the rest: published formulas that no maintained library installable in this project
provides. Each docstring names the paper and the reference implementation it follows, so a reader can check it.

- `lira_scores`: online and offline LiRA (Carlini et al. 2022), as in TensorFlow Privacy's `privacy_tests`
  (`membership_inference_attack/advanced_mia.py`, `compute_score_lira`), which needs TensorFlow, a framework this
  project does not install.
- `exposure` and `extrapolated_exposure`: canary exposure (Carlini et al. 2019), as in TensorFlow Privacy's
  `privacy_tests/secret_sharer/exposures.py`.
- `clopper_pearson_bounds`, `mu_lower_bound`, `eps_lower_bound`: the lower bounds of a privacy audit from measured
  rates (Nasr et al. 2023; Jagielski, Ullman & Oprea 2020), on top of SciPy's exact binomial interval.
- `correct_guesses`, `one_run_pvalue`, `one_run_eps`: the one-run audit (Steinke, Nasr & Jagielski 2023,
  Corollary 5.4), which has no reference package.

Used by code/part02-attacks/practitioner_lira.py, code/part05-dp-ml/practitioner_audit.py and
code/part05-dp-ml/practitioner_lora.py.
"""

import math

import numpy as np
from numpy.typing import NDArray
from scipy import stats

Array = NDArray[np.float64]
Mask = NDArray[np.bool_]


# --- Membership inference: LiRA ------------------------------------------------------------------------------


def lira_scores(
    phi: Array, shadow_phi: Array, shadow_member: Mask, global_variance: bool = False
) -> tuple[Array, Array]:
    """Online and offline LiRA scores of every record on one target model (Carlini et al. 2022).

    `phi` holds the target's logit-scaled confidence φ on each record, shape (n,); `shadow_phi` and `shadow_member`
    the shadow models' φ and which shadow trained on which record, shape (K, n). Online LiRA is the log-likelihood
    ratio of φ under a normal fitted to the record's IN shadows against one fitted to its OUT shadows; offline LiRA
    is how many OUT standard deviations φ lies above the OUT mean. With `global_variance`, one spread is shared by
    all records, which is steadier with few shadows. Reference implementation: TensorFlow Privacy,
    `privacy_tests/membership_inference_attack/advanced_mia.py`, `compute_score_lira`.
    """
    phi_in = np.where(shadow_member, shadow_phi, np.nan)  # NaN where the shadow did not train on the record
    phi_out = np.where(~shadow_member, shadow_phi, np.nan)
    mu_in, sd_in = np.nanmean(phi_in, axis=0), np.nanstd(phi_in, axis=0, ddof=1)
    mu_out, sd_out = np.nanmean(phi_out, axis=0), np.nanstd(phi_out, axis=0, ddof=1)
    if global_variance:
        sd_in = np.full_like(sd_in, np.sqrt(np.nanmean(sd_in**2)))
        sd_out = np.full_like(sd_out, np.sqrt(np.nanmean(sd_out**2)))
    online = stats.norm.logpdf(phi, loc=mu_in, scale=sd_in) - stats.norm.logpdf(phi, loc=mu_out, scale=sd_out)
    offline = (phi - mu_out) / sd_out
    return online, offline


# --- Memorization: canary exposure ---------------------------------------------------------------------------


def exposure(canary_score: float, candidate_scores: Array) -> float:
    """Exposure in bits: log2 of 1 / (the share of candidates scored at least as high), the canary included.

    Carlini et al. (2019), estimated by ranking the canary among random candidates of the same format. Reference
    implementation: TensorFlow Privacy, `privacy_tests/secret_sharer/exposures.py`, `compute_exposure_interpolation`.
    """
    at_least = 1 + np.sum(candidate_scores >= canary_score)
    return math.log2((len(candidate_scores) + 1) / at_least)


def extrapolated_exposure(canary_score: float, candidate_scores: Array, secret_space: int) -> float:
    """Exposure from a skew-normal fitted to the candidates' scores, capped at log2 of the number of secrets.

    Carlini et al. (2019), Section 4.3, for canaries that beat every candidate. Reference implementation:
    TensorFlow Privacy, `privacy_tests/secret_sharer/exposures.py`, `compute_exposure_extrapolation`.
    """
    shape, loc, scale = stats.skewnorm.fit(candidate_scores)
    tail = stats.skewnorm.sf(canary_score, shape, loc=loc, scale=scale)
    return min(-math.log2(max(tail, 1e-300)), math.log2(secret_space))


# --- Auditing: lower bounds from measured rates ---------------------------------------------------------------


def clopper_pearson_bounds(k: int, n: int, level: float) -> tuple[float, float]:
    """One-sided Clopper-Pearson bounds at `level` each: the ends of SciPy's exact interval at 1 - 2 level."""
    ci = stats.binomtest(k, n).proportion_ci(confidence_level=1 - 2 * level, method="exact")
    return ci.low, ci.high


def mu_lower_bound(false_pos: int, n_out: int, true_pos: int, n_in: int, beta: float) -> float:
    """Lower bound on the Gaussian-DP μ of one step: Φ⁻¹(TPR_lo) − Φ⁻¹(FPR_hi), each rate bounded at β/2.

    The per-step audit with Dirac canaries of Nasr et al. (2023); it holds with probability at least 1 - β.
    """
    fpr_hi = clopper_pearson_bounds(false_pos, n_out, beta / 2)[1]
    tpr_lo = clopper_pearson_bounds(true_pos, n_in, beta / 2)[0]
    if tpr_lo <= 0.0:
        return 0.0
    return max(0.0, stats.norm.ppf(tpr_lo) - stats.norm.ppf(fpr_hi))


def eps_lower_bound(false_pos: int, true_pos: int, n: int, delta: float, beta: float = 0.05) -> float:
    """Certified ε lower bound from n independent trials per hypothesis, with probability at least 1 - β.

    Solves TPR ≤ e^ε FPR + δ and 1 − FPR ≤ e^ε (1 − TPR) + δ at the Clopper-Pearson ends, each at β/2
    (Jagielski, Ullman & Oprea 2020; Nasr et al. 2023).
    """
    fpr_hi = clopper_pearson_bounds(false_pos, n, beta / 2)[1]
    tpr_lo = clopper_pearson_bounds(true_pos, n, beta / 2)[0]
    bounds = [0.0]
    if tpr_lo > delta and fpr_hi > 0:
        bounds.append(math.log((tpr_lo - delta) / fpr_hi))
    if 1 - fpr_hi > delta and tpr_lo < 1:
        bounds.append(math.log((1 - fpr_hi - delta) / (1 - tpr_lo)))
    return max(bounds)


# --- Auditing: the one-run audit ------------------------------------------------------------------------------


def correct_guesses(scores: Array, included: Mask, r: int) -> int:
    """Guess "in" for the r/2 highest scores and "out" for the r/2 lowest; count the correct guesses."""
    order = np.argsort(-scores, kind="stable")
    guessed_in, guessed_out = order[: r // 2], order[scores.size - r // 2 :]
    return int(np.count_nonzero(included[guessed_in]) + np.count_nonzero(~included[guessed_out]))


def one_run_pvalue(m: int, r: int, v: int, eps: float, delta: float) -> float:
    """P[W ≥ v] under (ε, δ)-DP with m canaries: f(v) + 2mδ max_i (f(v − i) − f(v))/i, f the binomial tail.

    Steinke, Nasr & Jagielski (2023), Corollary 5.4.
    """
    p = 1 / (1 + math.exp(-eps))
    tail = stats.binom.sf(v - 1, r, p)  # f(v) = P[Bin(r, p) ≥ v]
    i = np.arange(1, v + 1, dtype=np.float64)
    gaps = (stats.binom.sf(v - i - 1, r, p) - tail) / i
    return min(1.0, tail + 2 * m * delta * float(gaps.max()))


def one_run_eps(m: int, r: int, v: int, delta: float, beta: float) -> float:
    """The largest ε rejected at level β by v correct guesses out of r: training is not (ε, δ)-DP for smaller ε.

    Steinke, Nasr & Jagielski (2023), Corollary 5.4, by bisection on ε.
    """
    if v <= 0 or one_run_pvalue(m, r, v, 0.0, delta) >= beta:
        return 0.0
    lo, hi = 0.0, 1.0
    while one_run_pvalue(m, r, v, hi, delta) < beta and hi < 64:
        lo, hi = hi, 2 * hi
    for _ in range(50):
        mid = (lo + hi) / 2
        lo, hi = (mid, hi) if one_run_pvalue(m, r, v, mid, delta) < beta else (lo, mid)
    return lo
