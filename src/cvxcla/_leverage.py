"""Leverage (gross-exposure) constraints ``||w||_1 <= c`` via a signed lift.

The 1-norm is not linear in ``w``, but it is polyhedral: splitting each weight into
a long and a short leg, ``w_i = u_i - v_i`` with ``u_i, v_i >= 0``, turns
``||w||_1 <= c`` into the single linear row ``sum(u) + sum(v) <= c``. The lifted
problem is a CLA over the legs with one extra inequality row, and every other
constraint carries over by substituting ``w = P x``, where ``P`` maps each leg to
its asset with sign ``+1`` (long) or ``-1`` (short).

Only an asset whose box straddles zero (``lower < 0 < upper``) needs two legs. A
long-only asset (``lower >= 0``) keeps one ``+1`` leg and a short-only asset
(``upper <= 0``) one ``-1`` leg, so a long-only problem is not enlarged at all.

The lifted covariance ``P.T Sigma P`` is singular (the direction that raises both
legs of one asset is flat), but it only enters the solve through its free block,
and that block is a signed principal submatrix of ``Sigma`` as long as no asset has
both legs free. :class:`SignedLift` exposes it through the ``QuadraticForm``
interface without forming ``P.T Sigma P``, so structured backends keep their
advantage. Both legs are never free together on the traced path: while one leg is
off its lower bound, the other leg's multiplier equals twice the leverage row's
multiplier, which is non-negative, so the other leg can only become free at the
same ``lambda`` as the row releases. :func:`mask_leg_events` drops that
competing leg event so the row release wins the tie.

Under ``Sigma = X^T X`` and ``mu = X^T y`` the capped program is the constrained
LASSO read as a portfolio: its budget-indexed path is the LASSO path, and the tilt
sweep traced here is that path rescaled (Schmelzer and Hastie, arXiv:2609.25704,
Theorem 1 and Corollary 2). See :mod:`cvxcla.lasso`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray
from scipy.optimize import linprog  # type: ignore[import-untyped]

from .operators import QuadraticForm
from .types import TurningPoint


def _scale_rows(sign: NDArray[np.float64], v: NDArray[np.float64]) -> NDArray[np.float64]:
    """Multiply the rows of a vector or a column-stacked matrix by ``sign``."""
    result: NDArray[np.float64] = sign[:, None] * v if v.ndim == 2 else sign * v
    return result


class SignedLift(QuadraticForm):
    """The lifted quadratic form ``P.T @ base @ P`` of a signed leg-to-asset map.

    Leg ``k`` belongs to asset ``asset[k]`` with sign ``sign[k]``, so
    ``(P x)_i = sum_{k: asset[k] = i} sign[k] x_k``. Products are routed through
    the ``base`` operator on the distinct assets involved; a free-block solve is a
    signed solve on the base's principal block and is only defined while no asset
    has two legs in the free set.

    Attributes:
        base: The covariance over the assets.
        asset: The asset each leg belongs to (length = number of legs).
        sign: ``+1`` for a long leg, ``-1`` for a short leg.
    """

    def __init__(self, base: QuadraticForm, asset: NDArray[np.intp], sign: NDArray[np.float64]) -> None:
        """Wrap ``base`` with the leg map ``(asset, sign)``."""
        self.base = base
        self.asset = asset
        self.sign = sign

    @property
    def n(self) -> int:
        """Number of legs."""
        return int(self.asset.shape[0])

    def matvec(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return ``P.T @ base @ P @ x``."""
        legs = np.arange(self.n)
        return self.block_matvec(legs, legs, x)

    def block_matvec(self, rows: object, cols: object, v: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return the ``(rows, cols)`` block of the lifted form applied to ``v``.

        The column legs are first summed onto their distinct assets (both legs of
        one asset collapse to a single signed entry), the base block product is
        taken over distinct assets, and the result is spread back onto the row legs.
        """
        rows = np.asarray(rows, dtype=np.intp)
        cols = np.asarray(cols, dtype=np.intp)
        col_assets, col_inverse = np.unique(self.asset[cols], return_inverse=True)
        collapsed = np.zeros((col_assets.shape[0], *v.shape[1:]))
        np.add.at(collapsed, col_inverse, _scale_rows(self.sign[cols], np.asarray(v, dtype=np.float64)))
        row_assets, row_inverse = np.unique(self.asset[rows], return_inverse=True)
        product = np.asarray(self.base.block_matvec(row_assets, col_assets, collapsed))
        return _scale_rows(self.sign[rows], product[row_inverse])

    def solve_free(self, free: object, rhs: NDArray[np.float64]) -> NDArray[np.float64]:
        """Solve on the free block, a signed principal block of ``base``.

        Raises:
            numpy.linalg.LinAlgError: If an asset has both legs in ``free``, which
                makes the block singular.
        """
        free = np.asarray(free, dtype=np.intp)
        assets = self.asset[free]
        if np.unique(assets).size != assets.size:
            msg = "both legs of an asset are free, so the lifted free block is singular"
            raise np.linalg.LinAlgError(msg)
        sign = self.sign[free]
        return _scale_rows(sign, np.asarray(self.base.solve_free(assets, _scale_rows(sign, rhs))))

    def rcond_free(self, free: object) -> float:
        """Reciprocal condition number of the free block (``0`` if an asset has both legs free).

        Flipping signs leaves the spectrum unchanged, so this is the base's
        conditioning of the distinct free assets.
        """
        assets = self.asset[np.asarray(free, dtype=np.intp)]
        if np.unique(assets).size != assets.size:
            return 0.0
        return float(self.base.rcond_free(assets))


@dataclass(frozen=True)
class LeverageLift:
    """The signed leg structure of a leverage-constrained problem.

    Attributes:
        asset: The asset each leg belongs to.
        sign: ``+1`` for a long leg, ``-1`` for a short leg.
        partner: For a leg of a two-legged asset, the index of its other leg;
            ``-1`` for the single leg of a long-only or short-only asset.
        lower: Lower bounds of the legs.
        upper: Upper bounds of the legs.
    """

    asset: NDArray[np.intp]
    sign: NDArray[np.float64]
    partner: NDArray[np.intp]
    lower: NDArray[np.float64]
    upper: NDArray[np.float64]

    @classmethod
    def from_bounds(cls, lower: NDArray[np.float64], upper: NDArray[np.float64]) -> LeverageLift:
        """Build the legs from the asset box ``lower <= w <= upper``.

        An asset with ``lower >= 0`` gets one long leg on ``[lower, upper]``; one
        with ``upper <= 0`` gets one short leg on ``[-upper, -lower]``; one with
        ``lower < 0 < upper`` gets a long leg on ``[0, upper]`` and a short leg on
        ``[0, -lower]``, stored next to each other.
        """
        asset: list[int] = []
        sign: list[float] = []
        leg_lower: list[float] = []
        leg_upper: list[float] = []
        partner: list[int] = []
        for i, (lo, up) in enumerate(zip(lower.tolist(), upper.tolist(), strict=True)):
            if lo >= 0.0:
                asset.append(i)
                sign.append(1.0)
                leg_lower.append(lo)
                leg_upper.append(up)
                partner.append(-1)
            elif up <= 0.0:
                asset.append(i)
                sign.append(-1.0)
                leg_lower.append(-up)
                leg_upper.append(-lo)
                partner.append(-1)
            else:
                k = len(asset)
                asset += [i, i]
                sign += [1.0, -1.0]
                leg_lower += [0.0, 0.0]
                leg_upper += [up, -lo]
                partner += [k + 1, k]
        return cls(
            asset=np.array(asset, dtype=np.intp),
            sign=np.array(sign),
            partner=np.array(partner, dtype=np.intp),
            lower=np.array(leg_lower),
            upper=np.array(leg_upper),
        )

    def columns(self, matrix: NDArray[np.float64]) -> NDArray[np.float64]:
        """Return ``matrix @ P``: each asset column copied onto its legs with the leg's sign."""
        result: NDArray[np.float64] = matrix[:, self.asset] * self.sign
        return result

    def to_assets(self, x: NDArray[np.float64], n: int) -> NDArray[np.float64]:
        """Return the asset weights ``P @ x`` of the leg weights ``x``."""
        weights = np.zeros(n)
        np.add.at(weights, self.asset, self.sign * x)
        return weights

    def any_leg(self, mask: NDArray[np.bool_], n: int) -> NDArray[np.bool_]:
        """Return, per asset, whether any of its legs is set in ``mask``."""
        return np.bincount(self.asset, weights=mask.astype(np.float64), minlength=n) > 0

    def with_cap(
        self, g: NDArray[np.float64], h: NDArray[np.float64], cap: float | None
    ) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
        """Return the lifted ``G P``, ``h`` with the gross-exposure row ``sum(x) <= cap`` appended.

        ``cap=None`` maps the rows without appending the cap (a cap already
        resolved into tightened bounds).
        """
        lifted = self.columns(g)
        if cap is None:
            return lifted, h
        return np.vstack([lifted, np.ones((1, self.asset.shape[0]))]), np.append(h, cap)

    def to_turning_point(self, tp: TurningPoint, n: int, p: int) -> TurningPoint:
        """Map a lifted turning point back to the ``n`` asset weights and the first ``p`` rows."""
        return TurningPoint(
            lamb=tp.lamb,
            weights=self.to_assets(tp.weights, n),
            free=self.any_leg(tp.free, n),
            active_ineq=tp.active_ineq[:p],
        )

    def net(self, x: NDArray[np.float64]) -> NDArray[np.float64]:
        """Net out overlapping legs: subtract ``min(u_i, v_i)`` from both legs of each asset.

        The asset weights are unchanged and the gross exposure can only fall, so
        every constraint that held for ``x`` still holds.
        """
        paired = self.partner >= 0
        overlap = np.zeros_like(x)
        overlap[paired] = np.minimum(x[paired], x[self.partner[paired]])
        return x - overlap


def mask_leg_events(
    box: NDArray[np.float64], partner: NDArray[np.intp], at_lower: NDArray[np.bool_]
) -> NDArray[np.float64]:
    """Drop the leave-a-bound events of a leg whose partner is off its lower bound.

    While one leg of an asset is free (or at its upper bound), the other leg's
    multiplier is twice the leverage row's multiplier, so it reaches zero only
    exactly when the row releases. Freeing that leg would put both legs in the free
    set and make the lifted free block singular; the row release is the event that
    must fire, so the leg's leave events (columns 2 and 3) are removed.

    Args:
        box: The ``(legs, 4)`` box-event matrix.
        partner: The partner leg of every leg (``-1`` for none).
        at_lower: Mask of the legs held at their lower bound.

    Returns:
        A copy of ``box`` with the suppressed events set to ``-inf``.
    """
    paired = partner >= 0
    suppress = np.zeros(partner.shape[0], dtype=bool)
    suppress[paired] = ~at_lower[partner[paired]]
    box = box.copy()
    box[suppress, 2:] = -np.inf
    return box


def _gross_lp(
    lift: LeverageLift,
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    g: NDArray[np.float64],
    h: NDArray[np.float64],
    sense: float,
) -> tuple[float, NDArray[np.float64]] | None:
    """Optimise the gross exposure ``sum(x)`` over the lifted feasible set.

    Minimises ``sense * sum(x)`` subject to ``A P x = b``, ``G P x <= h`` and the leg
    box, via HiGHS.

    Returns:
        ``(sum(x), reduced costs of the leg lower bounds)`` at the optimum, or
        ``None`` if the linear program has no solution (the caller's own first
        vertex then reports the infeasibility).
    """
    n_legs = lift.asset.shape[0]
    has_ineq = g.shape[0] > 0
    result = linprog(
        c=np.full(n_legs, sense),
        A_eq=lift.columns(a),
        b_eq=b,
        A_ub=lift.columns(g) if has_ineq else None,
        b_ub=h if has_ineq else None,
        bounds=list(zip(lift.lower, lift.upper, strict=True)),
        method="highs",
    )
    if not result.success:
        return None
    return float(np.sum(result.x)), np.asarray(result.lower.marginals, dtype=np.float64)


def tighten_at_minimum_gross(
    lower: NDArray[np.float64],
    upper: NDArray[np.float64],
    a: NDArray[np.float64],
    b: NDArray[np.float64],
    g: NDArray[np.float64],
    h: NDArray[np.float64],
    leverage: float,
    tol: float,
) -> tuple[NDArray[np.float64], NDArray[np.float64], bool]:
    """Resolve a cap sitting at the smallest feasible gross exposure.

    When ``leverage`` equals ``c_min = min ||w||_1`` over the feasible set, the cap
    row is tight at every feasible point and implies a set of zero legs through the
    other rows. With a fully-invested budget, ``leverage = 1`` forces every short leg
    to zero. Carrying the cap row alongside those rows makes the maximum-return vertex
    degenerate: the free set cannot span them all. This is also the one cap at which
    Slater's condition fails.

    The feasible set is then the optimal face of ``min sum(x)``. By complementary
    slackness, a leg whose lower bound carries a positive reduced cost is zero on
    that whole face. Pinning it there is therefore exact: the short leg of a split
    asset pins ``lower = 0``, and its long leg pins ``upper = 0``. If the cap is
    then implied by the tightened box, because ``max sum(x)`` over it is at most
    ``leverage``, the row is dropped.

    Args:
        lower: Asset lower bounds.
        upper: Asset upper bounds.
        a: Equality-constraint matrix over the assets.
        b: Equality-constraint right-hand side.
        g: Inequality-constraint matrix over the assets.
        h: Inequality-constraint right-hand side.
        leverage: The gross-exposure cap ``c``.
        tol: Tolerance for comparing the cap with ``c_min`` and for a positive
            reduced cost.

    Returns:
        ``(lower, upper, keep_cap)``: the (possibly tightened) asset bounds and
        whether the cap row is still needed. Unchanged bounds and ``True`` when the
        cap exceeds ``c_min``.
    """
    lift = LeverageLift.from_bounds(lower, upper)
    minimum = _gross_lp(lift, a, b, g, h, sense=1.0)
    if minimum is None or leverage > minimum[0] + tol * max(1.0, minimum[0]):
        return lower, upper, True

    pinned = minimum[1] > tol
    lower, upper = lower.copy(), upper.copy()
    split = lift.partner >= 0
    short = split & (lift.sign < 0) & pinned
    long = split & (lift.sign > 0) & pinned
    lower[lift.asset[short]] = 0.0
    upper[lift.asset[long]] = 0.0

    maximum = _gross_lp(LeverageLift.from_bounds(lower, upper), a, b, g, h, sense=-1.0)
    keep_cap = maximum is None or maximum[0] > leverage + tol * max(1.0, leverage)
    return lower, upper, keep_cap
