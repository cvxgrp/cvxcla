#    Copyright 2023 Stanford University Convex Optimization Group
#
#    Licensed under the Apache License, Version 2.0 (the "License");
#    you may not use this file except in compliance with the License.
#    You may obtain a copy of the License at
#
#        http://www.apache.org/licenses/LICENSE-2.0
#
#    Unless required by applicable law or agreed to in writing, software
#    distributed under the License is distributed on an "AS IS" BASIS,
#    WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#    See the License for the specific language governing permissions and
#    limitations under the License.
"""Markowitz implementation of the Critical Line Algorithm.

This module provides the CLA class, which implements the Critical Line Algorithm
as described by Harry Markowitz and colleagues. The algorithm computes the entire
efficient frontier by finding all turning points, which are the points where the
set of assets at their bounds changes.
"""

import logging
from dataclasses import dataclass, field
from functools import cached_property

import numpy as np
from numpy.typing import NDArray

from .first import init_algo
from .operators import CovarianceOperator, DenseCovariance
from .types import Frontier, FrontierPoint, TurningPoint


@dataclass(frozen=True)
class CLA:
    """Critical Line Algorithm implementation based on Markowitz's approach.

    This class implements the Critical Line Algorithm as described by Harry Markowitz
    and colleagues. It computes the entire efficient frontier by finding all turning
    points, which are the points where the set of assets at their bounds changes.

    The algorithm starts with the first turning point (the portfolio with the highest
    expected return) and then iteratively computes the next turning point with a lower
    expected return until it reaches the minimum variance portfolio.

    Attributes:
        mean: Vector of expected returns for each asset.
        covariance: Covariance matrix of asset returns, either as a plain
            ``numpy`` array or as a ``CovarianceOperator`` backend
            (see ``cvxcla.operators``).
        lower_bounds: Vector of lower bounds for asset weights.
        upper_bounds: Vector of upper bounds for asset weights.
        a: Matrix for linear equality constraints (Ax = b).
        b: Vector for linear equality constraints (Ax = b).
        turning_points: List of turning points on the efficient frontier.
        tol: Tolerance for numerical calculations.
        logger: Logger instance for logging information and errors.

    """

    mean: NDArray[np.float64]
    covariance: NDArray[np.float64] | CovarianceOperator
    lower_bounds: NDArray[np.float64]
    upper_bounds: NDArray[np.float64]
    a: NDArray[np.float64]
    b: NDArray[np.float64]
    turning_points: list[TurningPoint] = field(default_factory=list)
    tol: float = 1e-5  # pragma: no mutate
    logger: logging.Logger = field(default_factory=lambda: logging.getLogger(__name__))

    @cached_property
    def covariance_operator(self) -> CovarianceOperator:
        """Return the covariance as a ``CovarianceOperator`` backend.

        A plain ``numpy`` covariance matrix is wrapped in ``DenseCovariance``;
        an object already implementing the protocol is passed through. This is
        the single point where the input form is normalised.
        """
        if isinstance(self.covariance, CovarianceOperator):
            return self.covariance
        return DenseCovariance(self.covariance)

    def __post_init__(self) -> None:
        """Initialize the CLA object and compute the efficient frontier.

        This method is automatically called after initialization. It computes
        the entire efficient frontier by finding all turning points, starting
        from the first turning point (highest expected return) and iteratively
        computing the next turning point with a lower expected return until
        it reaches the minimum variance portfolio.

        The reduced KKT system at each turning point is solved by block
        elimination: two multi-RHS solves against the free covariance block
        (via the covariance backend) and a small m x m Schur complement
        ``A_F @ Sigma_FF^{-1} @ A_F.T``, where m is the number of equality
        constraints. The covariance only enters through the
        ``CovarianceOperator`` interface, so structured backends (e.g.
        ``FactorCovariance``) never materialise an n x n matrix.

        Raises:
            RuntimeError: If all variables are blocked, which would make the
                          system of equations singular.

        """
        m = self.a.shape[0]
        ns = len(self.mean)
        cov = self.covariance_operator
        tol = self.tol

        # Compute and store the first turning point
        self._append(self._first_turning_point())

        lam = np.inf

        # Safety bound: each iteration fixes the activity of at least one asset,
        # so a correct trace runs O(ns) times. Far exceeding this means the event
        # loop failed to terminate (e.g. cycling); fail loudly instead of hanging.
        # The bound is far above any valid frontier length.
        max_iterations = 100 * (ns + 1)  # pragma: no mutate
        iterations = 0  # pragma: no mutate

        while lam > 0:  # pragma: no mutate
            iterations += 1  # pragma: no mutate
            if iterations > max_iterations:  # pragma: no mutate
                msg = "CLA failed to converge: too many iterations"  # pragma: no mutate
                raise RuntimeError(msg)  # pragma: no mutate
            last = self.turning_points[-1]

            # --- Identify active set ---
            blocked = ~last.free
            if np.all(blocked):
                msg = "All variables cannot be blocked"
                raise RuntimeError(msg)

            at_upper = blocked & (np.abs(last.weights - self.upper_bounds) <= tol)  # pragma: no mutate
            at_lower = blocked & (np.abs(last.weights - self.lower_bounds) <= tol)  # pragma: no mutate

            _out = at_upper | at_lower
            _in = ~_out

            fixed_weights = np.zeros(ns)
            fixed_weights[at_upper] = self.upper_bounds[at_upper]
            fixed_weights[at_lower] = self.lower_bounds[at_lower]

            # --- Solve the reduced KKT system by block elimination ---
            # [Sigma_FF  A_F.T] [x ]   [r1]      with r1, r2 the RHS for the
            # [A_F       0    ] [nu] = [r2]      alpha (weights) and beta system
            a_free = self.a[:, _in]

            # Free-block solves: Sigma_FF^{-1} [A_F.T | r1_alpha | r1_beta]
            rhs_free = np.column_stack(
                [
                    a_free.T,
                    -cov.cross(_in, fixed_weights),
                    self.mean[_in],
                ]
            )
            solved = cov.solve_free(_in, rhs_free)
            y = solved[:, :m]  # Sigma_FF^{-1} A_F.T
            z_alpha = solved[:, m]
            z_beta = solved[:, m + 1]

            # Schur complement A_F Sigma_FF^{-1} A_F.T and multipliers
            schur = a_free @ y
            r2_alpha = self.b - self.a[:, _out] @ fixed_weights[_out]
            nu = np.linalg.solve(schur, np.column_stack([a_free @ z_alpha - r2_alpha, a_free @ z_beta]))
            nu_alpha, nu_beta = nu[:, 0], nu[:, 1]

            # Back-substitute the free weights
            r_alpha = fixed_weights.copy()
            r_alpha[_in] = z_alpha - y @ nu_alpha
            r_beta = np.zeros(ns)
            r_beta[_in] = z_beta - y @ nu_beta

            # --- Compute Lagrange multipliers and directional derivatives ---
            gamma = cov.matvec(r_alpha) + self.a.T @ nu_alpha
            delta = cov.matvec(r_beta) + self.a.T @ nu_beta - self.mean

            # --- Compute event ratios ---
            # A free weight moves along w(lam) = r_alpha + lam * r_beta, so even
            # a tiny slope crosses a bound given a long enough lam range.
            # Filtering slopes at self.tol misses such crossings and lets
            # weights drift out of bounds; only slopes at floating-point noise
            # level are excluded: below sqrt(machine epsilon) a slope is
            # indistinguishable from solve noise, and the huge ratios it would
            # produce only amplify rounding errors. Spurious ratios above the
            # current lam are removed by the lam window below.
            eps = np.sqrt(np.finfo(np.float64).eps)
            # 4 columns = the 4 event types; extra unused columns are harmless.
            l_mat = np.full((ns, 4), -np.inf)  # pragma: no mutate

            # Precompute each event mask exactly once. The <,> vs <=,>= choice at
            # the eps boundary is numerically irrelevant — a slope/derivative
            # landing exactly on +/-sqrt(machine-eps) never occurs with real
            # data — so those boundary comparisons are marked no-mutate.
            beta_down = _in & (r_beta < -eps)  # pragma: no mutate
            beta_up = _in & (r_beta > +eps)  # pragma: no mutate
            delta_down = at_upper & (delta < -eps)  # pragma: no mutate
            delta_up = at_lower & (delta > +eps)  # pragma: no mutate

            # Columns 0,1 are "moves to a bound" (free->blocked) and 2,3 are
            # "leaves a bound" (blocked->free); the next-free update below only
            # tests dirchg >= 2, so swapping a column *within* a group (0<->1 or
            # 2<->3) is behaviourally identical and marked no-mutate. Crossing
            # the 1<->2 group boundary IS exercised by the frontier tests.
            l_mat[beta_down, 0] = (  # pragma: no mutate
                self.upper_bounds[beta_down] - r_alpha[beta_down]
            ) / r_beta[beta_down]
            l_mat[beta_up, 1] = (self.lower_bounds[beta_up] - r_alpha[beta_up]) / r_beta[beta_up]
            l_mat[delta_down, 2] = -gamma[delta_down] / delta[delta_down]  # pragma: no mutate
            l_mat[delta_up, 3] = -gamma[delta_up] / delta[delta_up]

            # --- Determine next event ---
            # The current segment w(lam) = r_alpha + lam * r_beta is only valid
            # for lam at or below the current value: the frontier is traced with
            # non-increasing lam, so ratios above it are spurious crossings
            # outside the segment and must not be selected. Ties at the current
            # lam are kept; degenerate problems resolve them one per iteration.
            l_mat[l_mat > lam + tol] = -np.inf  # pragma: no mutate

            lam_max = np.max(l_mat)
            if lam_max < 0:  # pragma: no mutate
                break

            # Bland-style anti-cycling rule: on degenerate problems (tied means,
            # duplicated assets) several events coincide at the same ratio. Among
            # all events within tol of the best ratio we pick the lowest asset
            # index (and, within an asset, the lowest event type), so the choice
            # is deterministic and cannot cycle.
            tied = np.argwhere(l_mat >= lam_max - tol)  # pragma: no mutate
            secchg, dirchg = tied[0]
            lam = l_mat[secchg, dirchg]

            # --- Update free set ---
            free = last.free.copy()
            free[secchg] = dirchg >= 2  # boundary → IN if dirchg in {2, 3}

            # --- Compute new turning point ---
            new_weights = r_alpha + lam * r_beta
            self._emit(lam, new_weights, free)

        # Final point at lambda = 0
        self._emit(0.0, r_alpha, last.free)

    def __len__(self) -> int:
        """Get the number of turning points in the efficient frontier.

        Returns:
            The number of turning points currently stored in the object.

        """
        return len(self.turning_points)

    def _first_turning_point(self) -> TurningPoint:
        """Calculate the first turning point on the efficient frontier.

        This method uses the init_algo function to find the first turning point
        based on the mean returns and the bounds on asset weights.

        Returns:
            A TurningPoint object representing the first point on the efficient frontier.

        """
        first = init_algo(
            mean=self.mean,
            lower_bounds=self.lower_bounds,
            upper_bounds=self.upper_bounds,
        )
        return first

    def _append(self, tp: TurningPoint, tol: float | None = None) -> None:
        """Append a turning point to the list of turning points.

        This method validates that the turning point satisfies the constraints
        before adding it to the list.

        Args:
            tp: The turning point to append.
            tol: Tolerance for constraint validation. If None, uses the class's
                tol attribute. Pass 0 for exact validation.

        Raises:
            ValueError: If the turning point violates any constraints.

        """
        tol = self.tol if tol is None else tol

        if not np.all(tp.weights >= (self.lower_bounds - tol)):  # pragma: no mutate
            msg = "Weights below lower bounds"
            raise ValueError(msg)
        if not np.all(tp.weights <= (self.upper_bounds + tol)):  # pragma: no mutate
            msg = "Weights above upper bounds"
            raise ValueError(msg)
        if not np.allclose(np.sum(tp.weights), 1.0):
            msg = "Weights do not sum to 1"
            raise ValueError(msg)

        self.turning_points.append(tp)

    def _emit(self, lamb: float, weights: NDArray[np.float64], free: NDArray[np.bool_]) -> None:
        """Build and store a turning point, projecting away sub-tolerance round-off.

        On tie-heavy or near-degenerate problems (a short, near-rank-deficient
        sample covariance, duplicated assets, or many coincident events) the walk
        can reach a degenerate vertex at which a free weight sits essentially on
        one of its bounds. Accumulated floating-point round-off over the many
        turning points of a large trace then places that weight a hair outside its
        box. The covariance there has near-flat directions (its small eigenvalues),
        and the round-off lies in exactly those directions, so the candidate is
        optimal to solver precision but not exactly feasible.

        We distinguish two regimes by the conditioning of the free-asset block.
        While that block stays numerically full rank its solve is reliable and any
        box violation is round-off: we project the candidate onto the feasible box
        and, for the canonical budget constraint, rescale to restore the budget
        exactly. The projected point is then exactly feasible while remaining
        optimal (its objective matches a reference QP solve to roughly ``1e-8``;
        the weight difference is the problem's own non-uniqueness along the flat
        directions, not suboptimality). This is a no-op for well-posed turning
        points, which are already strictly feasible. Once the free set grows past
        the covariance rank the block is numerically singular and its solve is
        unreliable; whatever weights it produces (feasible or not) cannot be
        trusted, so we refuse and raise an actionable diagnosis instead of
        silently returning a possibly-suboptimal frontier.

        The discriminator is the free block's reciprocal condition number, read
        from its symmetric eigenvalues. Unlike the magnitude of the box violation,
        which is the residual of a singular solve and therefore varies with the
        BLAS/LAPACK build, the conditioning is deterministic and portable, so the
        completed-vs-declined boundary is the same on every platform.

        Raises:
            ValueError: With a degeneracy-specific message when the free-asset
                block is numerically singular (an unreliable solve); otherwise
                propagates nothing.
        """
        # A genuinely rank-deficient free block has a reciprocal condition number
        # at round-off level (~1e-16); a well-posed or merely near-degenerate
        # block sits many orders above it (>= ~1e-4 across the degeneracy sweep in
        # experiments/degeneracy_boundary.py). The 1e-12 cut sits in the wide gap
        # between the two and is the conventional numerical-singularity scale.
        rcond_floor = 1e-12  # pragma: no mutate
        rcond = self.covariance_operator.rcond_free(free)
        if rcond < rcond_floor:
            n_free = int(np.count_nonzero(free))
            msg = (
                f"Critical Line Algorithm hit a degeneracy at lambda={lamb:.4g} "
                f"(free-set size {n_free}): the free-asset covariance block is "
                f"numerically singular (reciprocal condition number {rcond:.2g}), "
                "so its solve is unreliable and the turning point cannot be "
                "trusted. The trace was stopped rather than risk silently "
                "returning a suboptimal frontier. This happens when the free set "
                "grows past the covariance rank (for example a sample covariance "
                "from far fewer days than assets). Use a well-conditioned, "
                "full-rank estimate (ample history), or a FactorCovariance backend "
                "(diagonal-plus-low-rank), which is positive definite by construction."
            )
            raise ValueError(msg)
        # Full-rank regime: project onto the feasible box to clear round-off, then
        # restore the budget for the canonical single all-ones equality constraint
        # (the rescale factor is 1 +/- O(round-off), so it perturbs nothing material).
        weights = np.clip(weights, self.lower_bounds, self.upper_bounds)
        if self.a.shape[0] == 1 and np.allclose(self.a, 1.0):
            total = float(weights.sum())
            if total > 0.0:
                weights = weights * (self.b[0] / total)
        self._append(TurningPoint(lamb=lamb, weights=weights, free=free))

    @property
    def frontier(self) -> Frontier:
        """Get the efficient frontier constructed from the turning points.

        This property creates a Frontier object from the list of turning points,
        which can be used to analyze the risk-return characteristics of the
        efficient portfolios.

        Returns:
            A Frontier object representing the efficient frontier.

        """
        return Frontier(
            covariance=self.covariance,
            mean=self.mean,
            frontier=[FrontierPoint(point.weights) for point in self.turning_points],
        )
