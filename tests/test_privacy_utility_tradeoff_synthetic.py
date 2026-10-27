"""code/notes/privacy_utility_tradeoff_synthetic.py: its accounting, its selection step, its metrics, and a short run."""

import importlib.util
import math
import subprocess
import sys
from pathlib import Path
from types import ModuleType

import numpy as np
import pytest

SCRIPT = Path(__file__).resolve().parent.parent / "code" / "notes" / "privacy_utility_tradeoff_synthetic.py"
ACCOUNTING = Path(__file__).resolve().parent.parent / "code" / "part04-privacy-accounting" / "concentrated_dp.py"

pytestmark = pytest.mark.libs  # the script imports mbi (jax), scikit-learn and, for the model route, Opacus


def load(path: Path = SCRIPT) -> ModuleType:
    spec = importlib.util.spec_from_file_location(path.stem, path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_belief_ceiling_and_its_inverse() -> None:
    m = load()
    assert m.belief_ceiling(1.0) == pytest.approx(math.e / (1 + math.e))
    assert m.epsilon_for_belief(0.75) == pytest.approx(math.log(3))  # the note's policy: a 75% ceiling
    assert m.tpr_bound(1.0, 0.01) == pytest.approx(math.e * 0.01 + 1e-5)


def test_zcdp_conversion_is_tight_and_consistent() -> None:
    """The tight conversion gives at least the budget of the simple one, and inverts itself."""
    m = load()
    simple = load(ACCOUNTING)
    for eps in (0.1, 1.0, 3.0):
        rho = m.rho_for_epsilon(eps, 1e-5)
        assert rho >= simple.rho_for_target(eps, 1e-5)
        assert m.zcdp_delta(rho, eps) == pytest.approx(1e-5, rel=1e-3)
        assert m.zcdp_delta(rho, eps * 1.01) < 1e-5
    # The value of cdp_rho(1.0, 1e-5) in the reference implementation (github.com/IBM/discrete-gaussian-differential-privacy).
    assert m.rho_for_epsilon(1.0, 1e-5) == pytest.approx(0.0305566, rel=1e-5)
    # The budgets the note's table quotes, at delta = 1e-5.
    assert m.rho_for_epsilon(0.1) == pytest.approx(0.00043, abs=0.000005)
    assert m.rho_for_epsilon(3.0) == pytest.approx(0.22, abs=0.005)


def test_exponential_mechanism_prefers_high_scores() -> None:
    m = load()
    rng = np.random.default_rng(0)
    scores = {("a",): 0.0, ("b",): 10.0}
    draws = [m.exponential_mechanism(scores, 1.0, rng) for _ in range(500)]
    share = draws.count(("b",)) / 500
    assert share == pytest.approx(math.exp(5) / (1 + math.exp(5)), abs=0.03)  # exp(eps * score / 2)


def test_encoding_follows_the_codebook() -> None:
    m = load()
    sys.path.insert(0, str(SCRIPT.parents[1]))
    import practitioner_datasets

    data = m.encode(practitioner_datasets.adult().head(1000))
    dom = data.domain
    assert list(dom) == m.ATTRIBUTES and len(dom) == 13
    assert dom.size() == 36_578_304_000
    for name in dom:
        column = np.asarray(data.df[name])
        assert column.min() >= 0 and column.max() < dom[name]
    assert len(m.pairs(dom)) == 78 and dom.size(m.SEGMENTS) == 90
    assert m.workload_error(data, data) == 0.0 and m.misplaced(data, data) == 0.0


def test_model_tables_agree_with_the_library() -> None:
    """The numpy message passing gives the model's one- and two-way tables, joined or disconnected, as mbi does."""
    m = load()
    sys.path.insert(0, str(SCRIPT.parents[1]))
    import practitioner_datasets

    data = m.encode(practitioner_datasets.adult().head(2000))
    rng = np.random.default_rng(0)
    cliques = [(a,) for a in data.domain] + [("age", "income"), ("income", "education"), ("sex", "relationship")]
    model = m.fit(data.domain, [m.measure(data, c, 5.0, rng) for c in cliques], iters=200)
    wanted = [(a,) for a in data.domain] + m.pairs(data.domain)
    tables = m.model_tables(model, wanted)
    for clique in wanted:
        assert tables[clique] == pytest.approx(np.asarray(model.project(clique).datavector()), abs=1e-6)


def test_frontier_and_reach() -> None:
    m = load()
    assert m.non_dominated([(0.1, 5.0), (0.3, 4.0), (1.0, 9.0), (3.0, 9.0)]) == [0, 2]
    assert m.reached(np.array([0.9, 0.1, 0.8, 0.2]), np.array([1, 1, 0, 0]), 2) == 1


def test_short_run_writes_the_note_inputs(tmp_path: Path) -> None:
    out = subprocess.run(
        [sys.executable, str(SCRIPT), "--seeds", "1", "--size-seeds", "1", "--epsilons", "1.0", "--out", str(tmp_path)],
        capture_output=True,
        text=True,
        check=True,
    ).stdout
    # The facts the note quotes as text.
    assert "rows: records 32,561  campaign 16,281  attributes 13" in out
    assert "two-way tables 78" in out
    assert "plan: 2 attributes, 90 segments, 7,841 eligible customers" in out
    assert "offers = eligible customers in the campaign: 3,846  random offers reach 908.5" in out
    assert "policy: belief ceiling 75% from even odds -> epsilon <= 1.0986" in out
    assert "eps 1 -> rho 0.03056" in out
    for name in (
        "technical.svg",
        "plan.svg",
        "business.svg",
        "sizes.svg",
        "_frontier.md",
        "_menu.md",
        "_sizes.md",
        "_decision.md",
    ):
        assert (tmp_path / name).stat().st_size > 0
