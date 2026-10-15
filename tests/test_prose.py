"""tools/prose.py flags the prose the author rejected and passes the prose he approved.

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
    "A noisy answer, a measured error rate, the size of a sampled minibatch: each is random, and the practical "
    "question is **how wide an error bar must be to fail with probability at most $\\beta$**. Concentration "
    "inequalities answer it.",
    # Symbols used before words explain them; mu never explained; the argument squeezed into one clause.
    "A quantity $X \\ge 0$ with mean $\\mathbb E[X]$ cannot often be much larger than its mean, or those large "
    "values alone would pull the mean up. That is **Markov's inequality**, $\\Pr[X \\ge a] \\le \\mathbb E[X]/a$. "
    "Apply it to the squared deviation $(X - \\mu)^2$, whose mean is the variance $\\sigma^2$, and you get "
    "**Chebyshev's inequality**,",
]

APPROVED = [
    "Almost every number in a private system is random. The noise added to a count is random by design. An error "
    "rate measured on a test set depends on which examples happened to be in it. The number of examples in a "
    "randomly sampled minibatch changes from one step to the next. Before you can use any of these numbers, you "
    "need to know how far it can stray from the value you care about. A concentration inequality answers that "
    "question. It tells you how wide to make an interval around your estimate so that the true value falls "
    "outside it only rarely, say once in a thousand runs.",
    "Start with the weakest thing you could know about a quantity: it is never negative, and you know its average. "
    "That alone limits how often it can be large. Suppose customers wait 2 minutes at a counter on average. Then "
    "at most a quarter of them can wait 8 minutes or more, because if more than a quarter did, those customers "
    "alone would push the average above 2 minutes. This is **Markov's inequality**. For a quantity $X$ that is "
    "never negative and a threshold $a$, it says that $\\Pr[X \\ge a] \\le \\mathbb E[X]/a$.",
]


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
    )
    assert out.returncode == 0, out.stdout
