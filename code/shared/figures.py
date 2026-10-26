"""The figure rules of AGENTS.md in one place: SVG output that is the same on every run.

Call ``configure(__name__)`` before importing ``matplotlib.pyplot``. It selects the Agg backend, salts
the SVG element ids with the module's name so reruns do not churn the files, and keeps text as text.
``save`` writes a figure without a timestamp, so the output is reproducible.
"""

from pathlib import Path

import matplotlib
from matplotlib.figure import Figure


def configure(module: str) -> None:
    """Set matplotlib up for deterministic SVG figures from the script named `module`."""
    matplotlib.use("Agg")
    matplotlib.rcParams["svg.hashsalt"] = module  # stable element ids, so reruns don't churn the SVGs
    matplotlib.rcParams["svg.fonttype"] = "none"  # keep text as text: smaller, diffable files


def save(fig: Figure, path: Path) -> None:
    """Write a figure as SVG with no timestamp, so the file is identical on every run."""
    fig.savefig(path, metadata={"Date": None})
