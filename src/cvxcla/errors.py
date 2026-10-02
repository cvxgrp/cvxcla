"""Exceptions raised while tracing a frontier.

A trace can stop for three different reasons, and the remedy differs for each:

* :class:`InfeasibleProblemError` -- the constraints admit no portfolio at all
  (bounds that cross, a budget the box cannot reach, an inconsistent equality
  system, or a linear program that HiGHS proves infeasible). Fix the data.
* :class:`DegenerateProblemError` -- the problem is feasible but lies outside the
  domain the algorithm supports: a numerically singular free covariance block, or
  a maximum-return vertex whose free set cannot span its active constraints. A
  small ridge, a factor model, or a perturbation of the constraints usually helps.
* :class:`NumericalError` -- the trace broke down numerically: the iteration cap
  was hit, the feasibility projection did not converge (:class:`ProjectionError`),
  or a turning point failed its constraint validation (:class:`FeasibilityError`).

Every class derives from :class:`CLAError`, and also from the builtin exception the
package raised before these classes existed (``ValueError`` for the first two,
``RuntimeError`` for numerical failures, both for :class:`FeasibilityError`), so
code that catches the builtins keeps working. Malformed input (wrong shapes, a
non-symmetric covariance, a builder used out of order) still raises a plain
``ValueError``: those are programming errors, not properties of the problem.
"""

from __future__ import annotations

__all__ = [
    "CLAError",
    "DegenerateProblemError",
    "FeasibilityError",
    "InfeasibleProblemError",
    "NumericalError",
    "ProjectionError",
]


class CLAError(Exception):
    """Base class for every error the tracer raises about the problem or its numerics."""


class InfeasibleProblemError(CLAError, ValueError):
    """The constraints admit no portfolio."""


class DegenerateProblemError(CLAError, ValueError):
    """The problem is feasible but outside the domain the algorithm supports."""


class NumericalError(CLAError, RuntimeError):
    """The trace broke down numerically."""


class FeasibilityError(NumericalError, ValueError):
    """A turning point violated a constraint beyond tolerance after projection.

    Also a ``ValueError``, which is what this check raised before the hierarchy.
    """


class ProjectionError(NumericalError):
    """The feasibility projection of a turning point did not converge.

    The candidate lay outside its box by more than round-off, or the box and the
    active constraints barely intersect, so clipping and re-projecting onto
    ``{w : C w = d}`` did not reach a feasible point. This signals a numerical
    failure, not an infeasible problem; the remedy is usually a better-conditioned
    covariance (a small ridge, or a factor model).
    """
