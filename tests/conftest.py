"""Shared pytest / Hypothesis configuration for the project test suite.

Hypothesis is run in a deterministic profile so that property-based tests draw
the *same* examples on every machine. Without it, Hypothesis explores different
random inputs per run, so a failure can reproduce locally and not in CI, or the
other way round.
"""

from hypothesis import settings

settings.register_profile("cvxcla-deterministic", derandomize=True, database=None)
settings.load_profile("cvxcla-deterministic")
