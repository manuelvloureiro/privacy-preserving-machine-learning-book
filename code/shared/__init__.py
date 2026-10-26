"""Helpers every script shares, so that a bound or a figure rule is written once.

A script imports them after putting ``code/`` on the path, as the practitioner scripts do for their
dataset loader::

    sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
    from shared.figures import configure
    from shared.stats import normal_cdf
"""
