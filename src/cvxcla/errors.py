"""Exceptions raised by cvxcla.

Input validation raises the built-in :class:`ValueError`; the exceptions here
signal numerical failures of a trace that are not the caller's input error.
"""

from __future__ import annotations

__all__ = ["ProjectionError"]


class ProjectionError(RuntimeError):
    """The feasibility projection of a turning point did not converge.

    The candidate lay outside its box by more than round-off, or the box and the
    active constraints barely intersect, so clipping and re-projecting onto
    ``{w : C w = d}`` did not reach a feasible point. This signals a numerical
    failure, not an infeasible problem; the remedy is usually a better-conditioned
    covariance (a small ridge, or a factor model).
    """
