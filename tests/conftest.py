"""Shared pytest / Hypothesis configuration for the project test suite.

Hypothesis is run in a deterministic profile so that property-based tests draw
the *same* examples on every machine. This keeps the mutmut mutation score
reproducible: without it, Hypothesis explores different random inputs per run,
so some mutants are killed only by chance and the score differs between local
and CI (see the mutation gate in ``.github/workflows/rhiza_mutation.yml``).
"""

from hypothesis import settings

settings.register_profile("cvxcla-deterministic", derandomize=True, database=None)
settings.load_profile("cvxcla-deterministic")
