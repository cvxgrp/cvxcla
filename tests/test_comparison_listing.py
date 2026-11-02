"""Run the 39-line KKT listing of the comparison paper as printed, and check its claims.

Appendix A of ``docs/paper/comparison.tex`` prints a complete CLA for box bounds and one
budget row (the listing labelled ``lst:kktcla``). These tests cut the code out of the
``lstlisting`` block in that figure, run it, and check what the appendix says of it:

1. On random 10-factor long-only problems with n = 20, ..., 640 it returns cvxcla's turning
   points: the same number, the number quoted in the appendix, and paths ``w(lambda)``
   within 1e-12.
2. It agrees with cvxcla on a few other problems: long-only and shorting factor models,
   upper bounds that bind exactly, and an ill-conditioned covariance.
3. It refuses, by raising, infeasible bounds, a tied largest expected return and a
   numerically singular free block, rather than returning a wrong frontier.

cvxcla is the reference because its own test suite certifies every segment with the KKT
conditions; these tests check the listing, not cvxcla.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from pathlib import Path

import numpy as np
import pytest

from cvxcla import CLA

PAPER = Path(__file__).resolve().parents[1] / "docs" / "paper" / "comparison.tex"
SIZES = [20, 40, 80, 160, 320, 640]
TOL = 1e-12


def _appendix() -> str:
    """Return the text of the appendix that holds the listing."""
    text = PAPER.read_text(encoding="utf-8")
    start = text.index(r"\label{app:kktcla}")
    return text[start : text.index(r"\end{document}", start)]


def _listing() -> str:
    """Return the code of the lstlisting block in the figure labelled lst:kktcla."""
    figures = re.findall(r"\\begin\{figure\}.*?\\end\{figure\}", PAPER.read_text(encoding="utf-8"), re.DOTALL)
    blocks = [f for f in figures if r"\label{lst:kktcla}" in f]
    assert len(blocks) == 1, "expected exactly one figure labelled lst:kktcla"
    match = re.search(r"\\begin\{lstlisting\}(?:\[[^\]]*\])?\n(.*?)\\end\{lstlisting\}", blocks[0], re.DOTALL)
    assert match is not None, "no lstlisting block in the figure labelled lst:kktcla"
    return match.group(1)


@pytest.fixture(scope="module")
def cla() -> Callable:
    """The ``cla`` function defined by the listing, executed as printed."""
    namespace: dict[str, object] = {}
    exec(compile(_listing(), f"{PAPER.name}:lst:kktcla", "exec"), namespace)  # noqa: S102  # nosec B102
    return namespace["cla"]


def _path_distance(a: tuple[np.ndarray, np.ndarray], b: tuple[np.ndarray, np.ndarray]) -> float:
    """Exact sup-norm distance between two piecewise-linear paths w(lambda).

    Each path is linear in lambda between turning points and constant above its first
    finite one, so the distance is attained at a turning point of one of the two.
    """

    def at(weights: np.ndarray, lambdas: np.ndarray, grid: np.ndarray) -> np.ndarray:
        """Interpolate the path (weights, lambdas) at each lambda in grid."""
        finite = np.isfinite(lambdas)
        order = np.argsort(lambdas[finite], kind="stable")
        lam, w = lambdas[finite][order], weights[finite][order]
        return np.stack([np.interp(grid, lam, w[:, j]) for j in range(w.shape[1])], axis=1)

    (wa, la), (wb, lb) = a, b
    grid = np.union1d(la[np.isfinite(la)], lb[np.isfinite(lb)])
    return max(float(np.abs(wa[0] - wb[0]).max()), float(np.abs(at(wa, la, grid) - at(wb, lb, grid)).max()))


def _both(cla: Callable, mu: np.ndarray, cov: np.ndarray, lo: np.ndarray, up: np.ndarray) -> tuple[tuple, tuple]:
    """Trace one problem with the listing and with cvxcla; return both paths."""
    pts = cla(mu, cov, lo, up)
    mine = (np.array([w for _, w in pts]), np.array([lam for lam, _ in pts]))
    n = len(mu)
    ref = CLA(mean=mu, covariance=cov, lower_bounds=lo, upper_bounds=up, a=np.ones((1, n)), b=np.ones(1))
    theirs = (np.array([tp.weights for tp in ref.turning_points]), np.array([tp.lamb for tp in ref.turning_points]))
    return mine, theirs


def _factor_cov(rng: np.random.Generator, n: int, k: int) -> np.ndarray:
    """A K-factor covariance: diagonal plus a rank-K term."""
    u = rng.standard_normal((n, k)) / np.sqrt(n)
    delta = rng.uniform(0.5, 2.0, k) * n
    d = rng.uniform(0.5, 2.0, n)
    return np.diag(d) + (u * delta) @ u.T


def _factor_problem(n: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """The seed-7 10-factor long-only problem of size n as (mu, cov, lo, up)."""
    rng = np.random.default_rng(7)
    cov = _factor_cov(rng, n, 10)
    return rng.uniform(0.0, 1.0, n), cov, np.zeros(n), np.ones(n)


def test_listing_has_39_lines() -> None:
    """The appendix's title and text say 39 lines."""
    assert len(_listing().splitlines()) == 39


def test_factor_problems_match_cvxcla_and_the_quoted_counts(cla: Callable) -> None:
    """The listing returns cvxcla's turning points, in the numbers the appendix quotes."""
    counts = []
    for n in SIZES:
        mine, theirs = _both(cla, *_factor_problem(n))
        assert len(mine[0]) == len(theirs[0]), f"n={n}: {len(mine[0])} vs cvxcla {len(theirs[0])} points"
        assert _path_distance(mine, theirs) <= TOL, f"n={n}"
        counts.append(len(theirs[0]))
    quoted = re.search(r"turning points\s+\(\$([0-9, ]+)\$\)", _appendix())
    assert quoted is not None, "no turning-point counts quoted in the appendix"
    assert [int(x) for x in quoted.group(1).split(",")] == counts


def _other_problems() -> list:
    """Shorting, binding-cap and ill-conditioned problems as pytest params (mu, cov, lo, up, tol)."""
    rng = np.random.default_rng(1)
    q, _ = np.linalg.qr(rng.standard_normal((20, 20)))
    ill = (q * np.geomspace(1.0, 1e-10, 20)) @ q.T
    problems = [
        ("long-only n=50", _factor_cov(rng, 50, 5), 0.0, 1.0, TOL),
        ("shorting n=50", _factor_cov(rng, 50, 5), -1.0, 1.0, TOL),
        ("caps 1/4 bind exactly, n=20", _factor_cov(rng, 20, 5), 0.0, 1 / 4, TOL),
        ("caps 1/5 bind, n=20", _factor_cov(rng, 20, 5), 0.0, 1 / 5, TOL),
        ("cond(S) = 1e10, n=20", 0.5 * (ill + ill.T), 0.0, 1.0, 1e-8),
    ]
    return [
        pytest.param(rng.uniform(0.0, 1.0, len(cov)), cov, lo, up, tol, id=label)
        for label, cov, lo, up, tol in problems
    ]


@pytest.mark.parametrize(("mu", "cov", "lo", "up", "tol"), _other_problems())
def test_other_problems_match_cvxcla(
    cla: Callable, mu: np.ndarray, cov: np.ndarray, lo: float, up: float, tol: float
) -> None:
    """Shorting, binding caps and ill-conditioning do not throw the listing off cvxcla's path."""
    n = len(mu)
    mine, theirs = _both(cla, mu, cov, np.full(n, lo), np.full(n, up))
    assert _path_distance(mine, theirs) <= tol


def _refusals() -> list:
    """Inputs the listing must refuse, as pytest params (args, expected exception)."""
    rng = np.random.default_rng(2)
    n = 20
    cov, mu = _factor_cov(rng, n, 5), rng.uniform(0.0, 1.0, n)
    tied = mu.copy()
    tied[np.argsort(mu)[-2]] = mu.max()
    singular = np.cov(rng.standard_normal((8, n)), rowvar=False)
    return [
        pytest.param((mu, cov, np.zeros(n), np.full(n, 0.9 / n)), ValueError, id="infeasible bounds"),
        pytest.param((tied, cov, np.zeros(n), np.ones(n)), ValueError, id="tied largest return"),
        pytest.param((mu, singular, np.zeros(n), np.ones(n)), np.linalg.LinAlgError, id="singular free block"),
    ]


@pytest.mark.parametrize(("args", "error"), _refusals())
def test_refuses_inputs_it_does_not_cover(cla: Callable, args: tuple, error: type[Exception]) -> None:
    """The listing raises rather than returning a wrong frontier."""
    with pytest.raises(error):
        cla(*args)
