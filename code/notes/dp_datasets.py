"""The computed content of the study note notes/dp-datasets/: what the size and shape of a dataset mean for DP.

    uv run python code/notes/dp_datasets.py [--out DIR]

The note catalogues the public datasets that differential privacy (DP) papers use. Three of its points need numbers,
and this script computes them:

- The noise floor. Under add/remove-one-record DP, a count carries Laplace noise whose standard deviation, sqrt(2)/eps,
  does not depend on the dataset. What the size n of a dataset decides is how that fixed noise compares with the
  counts the dataset can answer: a count of a share s of the records is n*s, and the noise is a fraction
  sqrt(2)/(eps*n*s) of it. The script prints that relative error for the common benchmark datasets and draws it
  against n, with each dataset marked.
- User-level privacy. Most benchmark datasets have one record per person, so they can only test example-level DP.
  Online Retail (UCI dataset 352; Chen, 2015; licence CC BY 4.0), loaded by code/practitioner_datasets.py, has a
  customer id on 406,829 of its 541,909 invoice lines, so one customer contributes many records. The script prints
  the distribution of lines per customer and, for a histogram of lines per month, the error of the user-level DP
  release as a function of the contribution bound k: each customer keeps at most k lines, the histogram's
  sensitivity to one customer is then k, and the released cells carry Laplace noise of scale k/eps.
- Marginals. The DP synthetic data generators that win benchmarks fit low-order marginals, and the reason is in the
  domain sizes of a tabular dataset. For Adult (UCI dataset 2; Becker & Kohavi, 1996; licence CC BY 4.0) the script
  prints the number of categories of each attribute, the number of cells of the full contingency table, and the
  cells and average count of the one-way and two-way marginals, against the Laplace noise each would carry.

Output: two SVG figures (noise_floor.svg, contributions.svg) and three Markdown snippets (_sizes.md, _contributions.md,
_adult_domain.md) that the note includes, written to _build/figures/notes/dp-datasets/ (or --out), and every number
the note quotes, printed.
"""

import argparse
import math
import sys
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import numpy.typing as npt
import polars as pl

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import practitioner_datasets

matplotlib.rcParams["svg.hashsalt"] = "dp_datasets"  # stable element ids, so reruns don't churn the SVGs
matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files

NOTE = "dp-datasets"
FIGURES = Path(__file__).resolve().parents[2] / "_build" / "figures" / "notes" / NOTE

Array = npt.NDArray[np.float64]


# --- The noise floor -----------------------------------------------------------------------------------------------


@dataclass(frozen=True)
class Dataset:
    """A benchmark dataset as the note's figure marks it: its name, its number of records and what one record is."""

    name: str
    records: int
    record: str


# Record counts as the source pages state them (the note's catalogue cites each). A dataset with a train/test split
# is counted whole, since a DP experiment may use the split it likes.
DATASETS = [
    Dataset("Breast Cancer Wisconsin", 569, "patient"),
    Dataset("German Credit", 1_000, "applicant"),
    Dataset("Bank Marketing", 45_211, "client contact"),
    Dataset("Adult", 48_842, "person"),
    Dataset("CIFAR-10", 60_000, "image"),
    Dataset("MNIST", 70_000, "image"),
    Dataset("Diabetes 130-US hospitals", 101_766, "hospital stay"),
    Dataset("Credit Card Fraud", 284_807, "transaction"),
    Dataset("ImageNet (train)", 1_281_167, "image"),
    Dataset("ACSIncome 2018", 1_664_500, "person"),
]
SHARE = 0.01  # the count the figure judges: the records in a 1% slice of the dataset
# Where each dataset's label sits in the figure, in points from its marker, so that neighbours don't overlap.
LABEL_OFFSETS: dict[str, tuple[int, int]] = {
    "Breast Cancer Wisconsin": (5, 5),
    "German Credit": (5, -10),
    "Bank Marketing": (-78, 6),
    "Adult": (-32, -12),
    "CIFAR-10": (5, 8),
    "MNIST": (5, -12),
    "Diabetes 130-US hospitals": (8, 2),
    "Credit Card Fraud": (5, -12),
    "ImageNet (train)": (-80, 6),
    "ACSIncome 2018": (5, -12),
}
EPSILONS = [0.1, 1.0, 10.0]


def count_noise_sd(eps: float) -> float:
    """Standard deviation of the Laplace noise on a count under add/remove-one-record ε-DP: the scale is 1/ε."""
    return math.sqrt(2.0) / eps


def relative_error(count: float, eps: float) -> float:
    """The noise's standard deviation as a fraction of a count of this size."""
    return count_noise_sd(eps) / count


def records_for_relative_error(share: float, eps: float, target: float) -> float:
    """The smallest n at which a count of a share of the records has relative error at most the target."""
    return count_noise_sd(eps) / (share * target)


def figure_noise_floor(out: Path) -> None:
    """Relative error of a 1% count against the number of records, one line per ε, with the datasets marked."""
    fig, ax = plt.subplots(figsize=(7.0, 4.2))
    n = np.logspace(2.5, 6.5, 200)
    for eps, colour in zip(EPSILONS, ["#d62728", "#1f77b4", "#2ca02c"]):
        ax.plot(n, [relative_error(SHARE * v, eps) for v in n], color=colour, label=f"ε = {eps:g}")
    ax.axhline(0.1, color="grey", linestyle=":", linewidth=1)
    ax.text(10**5.9, 0.11, "10% of the count", color="grey", fontsize=8, va="bottom")
    for d in DATASETS:
        y = relative_error(SHARE * d.records, 1.0)
        ax.plot([d.records], [y], marker="o", color="#1f77b4", markersize=4)
        ax.annotate(d.name, (d.records, y), textcoords="offset points", xytext=LABEL_OFFSETS[d.name], fontsize=7)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("records in the dataset, n")
    ax.set_ylabel("noise sd / count, for a count of 1% of n")
    ax.set_title("Laplace noise on a count, relative to the count, by dataset size", fontsize=10)
    ax.legend(loc="upper right", fontsize=8)
    ax.grid(True, which="major", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "noise_floor.svg")
    plt.close(fig)


def sizes_table() -> str:
    """Each dataset's records and the relative error of a 1% count at ε = 0.1 and 1, as a Markdown table."""
    lines = [
        "| Dataset | Records | One record is | 1% of the records | Noise at $\\varepsilon = 0.1$ | Noise at $\\varepsilon = 1$ |",
        "|:---|---:|:---|---:|---:|---:|",
    ]
    for d in DATASETS:
        count = SHARE * d.records
        lines.append(
            f"| {d.name} | {d.records:,} | one {d.record} | {count:,.0f} | "
            f"{relative_error(count, 0.1):.1%} | {relative_error(count, 1.0):.1%} |"
        )
    lines.append("")
    lines.append(
        ": The Laplace noise on a count of 1% of each dataset, as a share of that count, under add/remove-one-record "
        "ε-DP. The noise's standard deviation is √2/ε whatever the dataset; what changes is the count it is compared "
        "with. {#tbl-sizes}"
    )
    return "\n".join(lines) + "\n"


# --- User-level privacy on Online Retail ---------------------------------------------------------------------------


def lines_per_customer(retail: pl.DataFrame) -> npt.NDArray[np.int64]:
    """How many invoice lines each identified customer has, one entry per customer."""
    with_id = retail.filter(pl.col("CustomerID").is_not_null())
    return with_id.group_by("CustomerID").agg(pl.len().alias("lines"))["lines"].to_numpy()


def contribution_quantiles(counts: npt.NDArray[np.int64]) -> dict[str, float]:
    """The median, 90th and 99th percentiles, the mean and the maximum of the records per user."""
    return {
        "median": float(np.median(counts)),
        "p90": float(np.percentile(counts, 90)),
        "p99": float(np.percentile(counts, 99)),
        "mean": float(counts.mean()),
        "max": float(counts.max()),
    }


def monthly_counts(retail: pl.DataFrame, bound: int | None) -> Array:
    """Invoice lines per month, December 2010 to December 2011, when each customer keeps at most `bound` lines.

    A customer's lines are kept in date order, so bounding drops a heavy customer's later purchases. `None` keeps
    every line, which gives the true histogram."""
    with_id = retail.filter(pl.col("CustomerID").is_not_null()).sort("InvoiceDate")
    if bound is not None:
        with_id = with_id.with_columns(pl.int_range(pl.len()).over("CustomerID").alias("rank"))
        with_id = with_id.filter(pl.col("rank") < bound)
    month = with_id["InvoiceDate"].dt.year() * 12 + with_id["InvoiceDate"].dt.month()
    first, last = int(month.min()), int(month.max())  # pyright: ignore[reportArgumentType]
    counts = np.zeros(last - first + 1)
    for m, c in month.value_counts().iter_rows():
        counts[int(m) - first] = float(c)
    return counts


def histogram_rmse(true: Array, bounded: Array, bound: int, eps: float) -> float:
    """Root mean squared error, over the cells, of the user-level DP histogram with contribution bound `bound`.

    The error of a cell is its bias, the lines that bounding dropped, plus Laplace noise of scale bound/ε, because
    adding or removing one customer moves at most `bound` lines. The two are independent, so the mean squared
    error is the squared bias plus the noise variance, 2 (bound/ε)²."""
    bias = true - bounded
    noise_variance = 2.0 * (bound / eps) ** 2
    return float(math.sqrt(float(np.mean(bias**2)) + noise_variance))


def best_bound(bounds: list[int], errors: list[float]) -> int:
    """The contribution bound with the smallest error."""
    return bounds[int(np.argmin(errors))]


def figure_contributions(counts: npt.NDArray[np.int64], retail: pl.DataFrame, out: Path) -> dict[float, int]:
    """Left: the share of customers with more than k lines. Right: the error of the monthly histogram against k.

    Returns the best bound at each ε, for the text."""
    fig, (left, right) = plt.subplots(1, 2, figsize=(9.0, 3.8))
    sorted_counts = np.sort(counts)
    survival = 1.0 - np.arange(len(sorted_counts)) / len(sorted_counts)
    left.step(sorted_counts, survival, where="post", color="#1f77b4")
    left.set_xscale("log")
    left.set_yscale("log")
    left.set_xlabel("invoice lines, k")
    left.set_ylabel("share of customers with more than k lines")
    left.set_title("Online Retail: lines per customer", fontsize=10)
    left.grid(True, which="major", alpha=0.3)

    true = monthly_counts(retail, None)
    bounds = [int(b) for b in np.unique(np.logspace(0, math.log10(counts.max()), 40).astype(int))]
    best: dict[float, int] = {}
    for eps, colour in zip([0.1, 1.0], ["#d62728", "#1f77b4"]):
        errors = [histogram_rmse(true, monthly_counts(retail, b), b, eps) for b in bounds]
        right.plot(bounds, errors, color=colour, label=f"ε = {eps:g}")
        best[eps] = best_bound(bounds, errors)
    right.set_xscale("log")
    right.set_yscale("log")
    right.set_xlabel("contribution bound, k lines per customer")
    right.set_ylabel("RMSE of the monthly histogram")
    right.set_title("Error of the user-level DP release", fontsize=10)
    right.legend(fontsize=8)
    right.grid(True, which="major", alpha=0.3)
    fig.tight_layout()
    fig.savefig(out / "contributions.svg")
    plt.close(fig)
    return best


def contributions_table(counts: npt.NDArray[np.int64], retail: pl.DataFrame) -> str:
    """For a few contribution bounds: the share of lines kept and the error of the histogram at ε = 1."""
    true = monthly_counts(retail, None)
    total = float(counts.sum())
    lines = [
        "| Bound $k$ | Customers above it | Lines kept | Lines dropped per month | Noise sd at $\\varepsilon = 1$ | RMSE at $\\varepsilon = 1$ |",
        "|---:|---:|---:|---:|---:|---:|",
    ]
    for k in (10, 42, 100, 212, 500, 1_000, int(counts.max())):
        bounded = monthly_counts(retail, k)
        above = float(np.mean(counts > k))
        kept = float(np.minimum(counts, k).sum()) / total
        dropped = float(np.mean(true - bounded))
        lines.append(
            f"| {k:,} | {above:.1%} | {kept:.1%} | {dropped:,.0f} | {count_noise_sd(1.0) * k:,.0f} | "
            f"{histogram_rmse(true, bounded, k, 1.0):,.0f} |"
        )
    lines.append("")
    lines.append(
        ": Bounding each customer of Online Retail to k invoice lines, and what it does to a user-level DP "
        "histogram of lines per month at ε = 1. The bias falls and the noise grows with k. {#tbl-contributions}"
    )
    return "\n".join(lines) + "\n"


# --- Marginals of Adult --------------------------------------------------------------------------------------------

NUMERIC_BINS = 10  # each numeric attribute of Adult is cut into this many equal-width bins


def domain_sizes(adult: pl.DataFrame) -> dict[str, int]:
    """The number of values of each attribute: its categories, or NUMERIC_BINS for a numeric column."""
    sizes: dict[str, int] = {}
    for name in adult.columns:
        sizes[name] = adult[name].n_unique() if adult[name].dtype == pl.String else NUMERIC_BINS
    return sizes


def joint_cells(sizes: dict[str, int]) -> int:
    """Cells of the full contingency table: the product of the domain sizes."""
    return math.prod(sizes.values())


def marginal_cells(sizes: dict[str, int], order: int) -> tuple[int, int]:
    """The number of marginals of this order (sets of `order` attributes) and their cells in total."""
    names = list(sizes)
    count, cells = 0, 0
    for subset in _subsets(names, order):
        count += 1
        cells += math.prod(sizes[a] for a in subset)
    return count, cells


def _subsets(names: list[str], order: int) -> list[tuple[str, ...]]:
    if order == 0:
        return [()]
    out: list[tuple[str, ...]] = []
    for i, a in enumerate(names):
        out += [(a, *rest) for rest in _subsets(names[i + 1 :], order - 1)]
    return out


def adult_domain_table(sizes: dict[str, int], n: int, eps: float) -> str:
    """The domain of each attribute, then the one-way, two-way, three-way marginals and the joint against n."""
    lines = ["| Attribute | Values |", "|:---|---:|"]
    for name, size in sizes.items():
        lines.append(f"| {name} | {size:,} |")
    lines.append("")
    lines.append(
        f": The domain of Adult's attributes: the categories of each categorical attribute, with the "
        f"missing-value code counted as one, and {NUMERIC_BINS} equal-width bins for each numeric one. "
        "{#tbl-adult-domain}"
    )
    lines.append("")
    lines.append(
        "| Marginals | How many | Cells in total | Average records per cell | Noise sd per cell at $\\varepsilon = 1$ |"
    )
    lines.append("|:---|---:|---:|---:|---:|")
    for order in (1, 2, 3):
        count, cells = marginal_cells(sizes, order)
        per_cell = n * count / cells  # every record falls in one cell of each marginal
        lines.append(f"| {order}-way | {count:,} | {cells:,} | {per_cell:,.1f} | {count_noise_sd(eps / count):,.1f} |")
    cells = joint_cells(sizes)
    lines.append(f"| the full table | 1 | {cells:,} | {n / cells:.2g} | {count_noise_sd(eps):,.1f} |")
    lines.append("")
    lines.append(
        f": Adult's marginals against its {n:,} records. The noise is what each cell carries when the ε = 1 budget "
        "is split evenly over all the marginals of that order by basic composition. {#tbl-adult-marginals}"
    )
    return "\n".join(lines) + "\n"


# --- Main ----------------------------------------------------------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--out", type=Path, default=FIGURES)
    args = parser.parse_args()
    out: Path = args.out
    out.mkdir(parents=True, exist_ok=True)

    print("noise floor: Laplace noise sd on a count, under add/remove, is sqrt(2)/eps")
    for eps in EPSILONS:
        print(f"  eps={eps:<4g} noise sd={count_noise_sd(eps):.3f}")
    print(
        f"  n for a 1% count with 10% relative error: eps=0.1 {records_for_relative_error(SHARE, 0.1, 0.1):,.0f}"
        f"  eps=1 {records_for_relative_error(SHARE, 1.0, 0.1):,.0f}"
    )
    for d in DATASETS:
        count = SHARE * d.records
        print(
            f"  {d.name:<26} n={d.records:>9,}  1%={count:>8,.0f}  rel err eps=0.1 {relative_error(count, 0.1):6.1%}"
            f"  eps=1 {relative_error(count, 1.0):6.1%}"
        )

    retail = practitioner_datasets.online_retail()
    counts = lines_per_customer(retail)
    q = contribution_quantiles(counts)
    with_id = int(retail["CustomerID"].is_not_null().sum())
    print(f"\nonline retail: lines {retail.height:,}  with a customer id {with_id:,}  customers {len(counts):,}")
    print("  lines per customer: " + "  ".join(f"{k}={v:,.1f}" for k, v in q.items()))
    true = monthly_counts(retail, None)
    print(
        f"  months {len(true)}  lines per month mean {true.mean():,.0f}  min {true.min():,.0f}  max {true.max():,.0f}"
    )
    best = figure_contributions(counts, retail, out)
    for eps, k in best.items():
        bounded = monthly_counts(retail, k)
        print(
            f"  best bound at eps={eps:g}: k={k}  rmse={histogram_rmse(true, bounded, k, eps):,.0f}"
            f"  unbounded (k={int(q['max'])}): rmse={histogram_rmse(true, true, int(q['max']), eps):,.0f}"
        )

    adult = practitioner_datasets.adult()
    sizes = domain_sizes(adult)
    n = adult.height
    print(
        f"\nadult: records {n:,}  attributes {len(sizes)}  joint cells {joint_cells(sizes):,}"
        f"  records per joint cell {n / joint_cells(sizes):.2g}"
    )
    for order in (1, 2, 3):
        count, cells = marginal_cells(sizes, order)
        print(
            f"  {order}-way marginals {count:,}  cells {cells:,}  records per cell {n * count / cells:,.1f}"
            f"  noise sd at eps=1 split evenly {count_noise_sd(1.0 / count):,.1f}"
        )

    figure_noise_floor(out)
    (out / "_sizes.md").write_text(sizes_table())
    (out / "_contributions.md").write_text(contributions_table(counts, retail))
    (out / "_adult_domain.md").write_text(adult_domain_table(sizes, n, 1.0))


if __name__ == "__main__":
    main()
