"""tools/prose.py flags the prose the author rejected and passes the prose the author approved.

The anchors are real passages. The rejected ones are quoted from the practitioner edition's
concentration chapter, as the author flagged them; the approved ones are their rewrites.
If a change to the checks or thresholds breaks one of these, the checks have drifted from
the author's judgement.
"""

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
TOOL = ROOT / "tools" / "prose.py"

REJECTED = [
    # Front-loaded list before a colon; a symbol used before it is explained; 35 words.
    (
        "A noisy answer, a measured error rate, the size of a sampled minibatch: each is random, and the practical "
        "question is **how wide an error bar must be to fail with probability at most $\\beta$**. Concentration "
        "inequalities answer it."
    ),
    # Symbols used before words explain them; mu never explained; the argument squeezed into one clause.
    (
        "A quantity $X \\ge 0$ with mean $\\mathbb E[X]$ cannot often be much larger than its mean, or those large "
        "values alone would pull the mean up. That is **Markov's inequality**, $\\Pr[X \\ge a] \\le \\mathbb E[X]/a$. "
        "Apply it to the squared deviation $(X - \\mu)^2$, whose mean is the variance $\\sigma^2$, and you get "
        "**Chebyshev's inequality**,"
    ),
    # Choppy: the rewrite of the first passage, which the author then rejected in turn. Every claim has its
    # own short sentence, but nothing says how they relate, so it reads in hiccups.
    (
        "Almost every number in a private system is random. The noise added to a count is random by design. An error "
        "rate measured on a test set depends on which examples happened to be in it. The number of examples in a "
        "randomly sampled minibatch changes from one step to the next. Before you can use any of these numbers, you "
        "need to know how far each one can stray from the value you care about."
    ),
    # A named result stated in inline math, at the end of a sentence that also introduces its symbols.
    (
        "For a quantity $X$ that cannot be negative, with average $\\mathbb E[X]$, and any positive threshold $a$, "
        "Markov's inequality says that $\\Pr[X \\ge a] \\le \\mathbb E[X]/a$."
    ),
]

APPROVED = [
    (
        "Almost every number in a private system is random. Some are random by design, like the noise a mechanism "
        "adds to a count. Others are random because of how they were measured. An error rate depends on which "
        "examples happened to land in the test set, and the size of a randomly sampled minibatch changes from one "
        "step to the next. Before you rely on any of these numbers, you need to know how far it can stray from the "
        "value you actually care about."
    ),
    (
        "Suppose customers wait 2 minutes at a counter on average. Then at most a quarter of them can wait 8 minutes "
        "or more. If more than a quarter did, those customers alone would push the average above 2 minutes, and no "
        "customer can have a negative wait that pulls it back down. In general, take a quantity $X$ that cannot be "
        "negative, and write $\\mathbb E[X]$ for its average. For any positive threshold $a$, **Markov's "
        "inequality** says that\n$$\\Pr[X \\ge a] \\le \\frac{\\mathbb E[X]}{a}.$$\nIn the example, an "
        "average of 2 minutes and a threshold of 8 give the bound of a quarter."
    ),
]

# Rejected passages that pass every mechanical check: only the cold-reader and editorial reviews can
# catch them. Each comes with the words the review prompts must keep using to name the failure.
REVIEW_ONLY = [
    (
        (
            "Start with the weakest thing you could know about a quantity: it is never negative, and you know its "
            "average."
        ),
        "unstated assumption",
    ),
    (
        "The first inequality is the triangle inequality, $|u|-|v|\\le|u-v|$ with $u=y-f(x')$ and $v=y-f(x)$.",
        "packed display",
    ),
    (
        "From a tail bound on the loss to $(\\varepsilon,\\delta)$-DP, and why the converse fails",
        "purposeless section",
    ),
    (
        (
            "The advanced composition theorem states that $k$ adaptive $(\\varepsilon,\\delta_0)$-DP mechanisms "
            "together satisfy [the theorem], with its proof two sections later."
        ),
        "out of order",
    ),
]
PROMPTS = ROOT / "tools" / "prose-review"


def flagged(text: str) -> list[str]:
    out = subprocess.run(
        [sys.executable, str(TOOL), "text", text, "--worst", "10"], capture_output=True, text=True, check=True
    ).stdout
    return [line.strip() for line in out.splitlines() if line.strip().startswith(("•", "¶"))]


def test_rejected_passages_are_flagged() -> None:
    for text in REJECTED:
        assert flagged(text), text[:60]
    first, second = flagged(REJECTED[0]), flagged(REJECTED[1])
    assert any("front-loaded list" in f for f in first)
    assert any("new symbol \\beta" in f for f in first)
    assert any("new symbol X" in f for f in second)
    assert any("new symbol \\mu" in f for f in second)  # named only as part of "the squared deviation"
    assert any("choppy" in f for f in flagged(REJECTED[2]))
    assert any("result in inline math" in f for f in flagged(REJECTED[3]))


def test_approved_passages_pass() -> None:
    for text in APPROVED:
        assert flagged(text) == [], text[:60]


def test_full_edition_reference_chapter_passes() -> None:
    """The reference chapter AGENTS.md points writers to must pass its own book's check."""
    out = subprocess.run(
        [
            sys.executable,
            str(TOOL),
            "check",
            "--book",
            "book-full",
            "--page",
            "part01-prerequisites/02-concentration.md",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert out.returncode == 0, out.stdout


def test_review_prompts_name_the_failures_the_analyzer_misses() -> None:
    prompts = (PROMPTS / "cold-reader.md").read_text() + (PROMPTS / "editor.md").read_text()
    for text, failure in REVIEW_ONLY:
        assert failure in prompts, text[:60]
