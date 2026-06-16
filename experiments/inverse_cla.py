"""An incremental-inverse CLA baseline that isolates solve-vs-inverse (issue #671).

``cvxcla``'s dense backend re-solves a fresh block-eliminated KKT system at every
turning point (``np.linalg.solve(Sigma_FF, .)``), discarding the previous
factorisation. A classical alternative is to keep an explicit ``Sigma_FF^{-1}``
and update it incrementally as a single asset enters or leaves the free set.
``InverseCLA`` below does exactly that: it shares ``cvxcla``'s *vectorised* event
logic (the whole event search is one matrix operation, no Python loop over
candidate assets), and differs from the dense backend in one thing only --- it
maintains ``Sigma_FF^{-1}`` through rank-1 bordered (asset enters) and deletion
(asset leaves) updates, O(n_F^2) per step instead of the O(n_F^3) of a fresh
factorisation. Timing it against ``cvxcla``'s dense backend therefore isolates the
cost of the linear-algebra *strategy* (fresh solve vs maintained inverse) with the
event logic held fixed. It is restricted to the budget-constrained box problem (a
single equality row ``1^T w = 1``), the case the scaling experiment uses.

This is NOT a reimplementation of PyPortfolioOpt. PyPortfolioOpt's CLA (after
Bailey & Lopez de Prado) does the opposite of an incremental update: it calls
``np.linalg.inv`` on the free block *from scratch* at every step, and in the
branch that decides which blocked asset re-enters it recomputes that full inverse
once for *every* candidate blocked asset (a Python loop) --- O(n) dense inverses
per turning point. That structure, not a lack of numpy, is why it scales steeply.

Usage:
    uv run python experiments/inverse_cla.py        # verify vs cvxcla
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from cvxcla import CLA
from cvxcla.first import init_algo


class InverseCLA:
    """Vectorised CLA that maintains an explicit free-block inverse.

    Restricted to the budget constraint ``1^T w = 1`` (m = 1) with box bounds.
    Traces the same turning points as ``cvxcla.CLA`` (verified in ``__main__``)
    but with PyPortfolioOpt's incremental-inverse linear algebra, so a timing
    comparison isolates the cost of the linear-algebra strategy from
    vectorisation.
    """

    def __init__(
        self,
        mean: NDArray[np.float64],
        covariance: NDArray[np.float64],
        lower_bounds: NDArray[np.float64],
        upper_bounds: NDArray[np.float64],
        tol: float = 1e-5,
    ) -> None:
        """Trace the frontier on construction (mirrors ``cvxcla.CLA``)."""
        self.mean = mean
        self.cov = covariance
        self.lower_bounds = lower_bounds
        self.upper_bounds = upper_bounds
        self.tol = tol
        self.weights: list[NDArray[np.float64]] = []
        self._solve()

    def __len__(self) -> int:
        """Return the number of turning points traced."""
        return len(self.weights)

    def _solve(self) -> None:
        mean, cov = self.mean, self.cov
        lb, ub, tol = self.lower_bounds, self.upper_bounds, self.tol
        ns = len(mean)
        eps = np.sqrt(np.finfo(np.float64).eps)

        first = init_algo(mean=mean, lower_bounds=lb, upper_bounds=ub)
        free = first.free.copy()
        self.weights.append(first.weights)

        # Explicit inverse of the free covariance block, aligned with the
        # ordered index list ``order`` (not necessarily ascending: inserts
        # append at the end, deletes preserve relative order).
        order = list(np.where(free)[0])
        sinv = np.linalg.inv(cov[np.ix_(order, order)])

        lam = np.inf
        max_iterations = 100 * (ns + 1)
        iterations = 0
        r_alpha = first.weights

        while lam > 0:
            iterations += 1
            if iterations > max_iterations:
                msg = "InverseCLA failed to converge: too many iterations"
                raise RuntimeError(msg)

            blocked = ~free
            if np.all(blocked):
                msg = "All variables cannot be blocked"
                raise RuntimeError(msg)

            at_upper = blocked & (np.abs(self.weights[-1] - ub) <= tol)
            at_lower = blocked & (np.abs(self.weights[-1] - lb) <= tol)
            fixed = np.zeros(ns)
            fixed[at_upper] = ub[at_upper]
            fixed[at_lower] = lb[at_lower]

            idx = np.array(order, dtype=int)
            # Budget constraint: a_free is a row of ones, so A_F Sinv A_F.T is
            # the total sum of Sinv and the multipliers are scalars.
            y = sinv.sum(axis=1)  # Sinv @ 1
            cross = cov[np.ix_(idx, np.where(blocked)[0])] @ fixed[blocked]
            z_alpha = sinv @ (-cross)
            z_beta = sinv @ mean[idx]
            schur = y.sum()
            r2_alpha = 1.0 - fixed[blocked].sum()
            nu_alpha = (z_alpha.sum() - r2_alpha) / schur
            nu_beta = z_beta.sum() / schur

            r_alpha = fixed.copy()
            r_alpha[idx] = z_alpha - y * nu_alpha
            r_beta = np.zeros(ns)
            r_beta[idx] = z_beta - y * nu_beta

            gamma = cov @ r_alpha + nu_alpha
            delta = cov @ r_beta + nu_beta - mean

            free_in = free
            l_mat = np.full((ns, 4), -np.inf)
            beta_down = free_in & (r_beta < -eps)
            beta_up = free_in & (r_beta > +eps)
            delta_down = at_upper & (delta < -eps)
            delta_up = at_lower & (delta > +eps)
            l_mat[beta_down, 0] = (ub[beta_down] - r_alpha[beta_down]) / r_beta[beta_down]
            l_mat[beta_up, 1] = (lb[beta_up] - r_alpha[beta_up]) / r_beta[beta_up]
            l_mat[delta_down, 2] = -gamma[delta_down] / delta[delta_down]
            l_mat[delta_up, 3] = -gamma[delta_up] / delta[delta_up]
            l_mat[l_mat > lam + tol] = -np.inf

            lam_max = np.max(l_mat)
            if lam_max < 0:
                break

            tied = np.argwhere(l_mat >= lam_max - tol)
            secchg, dirchg = tied[0]
            lam = l_mat[secchg, dirchg]
            enters = dirchg >= 2

            self.weights.append(r_alpha + lam * r_beta)

            # --- Incremental update of Sigma_FF^{-1} for the one-asset flip ---
            if enters:
                # Bordered inverse: append asset ``secchg`` at the end.
                c = cov[idx, secchg]
                s = cov[secchg, secchg]
                v = sinv @ c
                schur_i = s - c @ v
                top = sinv + np.outer(v, v) / schur_i
                col = (-v / schur_i).reshape(-1, 1)
                sinv = np.block([[top, col], [col.T, np.array([[1.0 / schur_i]])]])
                order.append(int(secchg))
            else:
                # Deletion: drop the row/col of ``secchg`` from the inverse.
                p = order.index(int(secchg))
                mask = np.ones(len(order), dtype=bool)
                mask[p] = False
                b1p = sinv[mask, p]
                bpp = sinv[p, p]
                sinv = sinv[np.ix_(mask, mask)] - np.outer(b1p, b1p) / bpp
                order.pop(p)

            free = free.copy()
            free[secchg] = enters

        self.weights.append(r_alpha)


def _make_problem(rng: np.random.Generator, n: int, k: int) -> tuple[np.ndarray, dict]:
    """Same ground-truth K-factor dense problem as ``runtime_scaling.py``."""
    u = rng.standard_normal((n, k)) / np.sqrt(n)
    delta = rng.uniform(0.5, 2.0, k) * n
    d = rng.uniform(0.5, 2.0, n)
    dense = np.diag(d) + (u * delta) @ u.T
    problem = {
        "mean": rng.uniform(0.0, 1.0, n),
        "lower_bounds": np.zeros(n),
        "upper_bounds": np.ones(n),
        "a": np.ones((1, n)),
        "b": np.ones(1),
    }
    return dense, problem


def main() -> None:
    """Verify InverseCLA matches cvxcla, then report a small timing comparison."""
    for n in (20, 40, 80, 160):
        rng = np.random.default_rng(7)
        dense, problem = _make_problem(rng, n, 10)
        ref = CLA(covariance=dense, **problem)
        inv = InverseCLA(
            mean=problem["mean"],
            covariance=dense,
            lower_bounds=problem["lower_bounds"],
            upper_bounds=problem["upper_bounds"],
        )
        ref_w = np.array([tp.weights for tp in ref.turning_points])
        inv_w = np.array(inv.weights)
        same_count = len(ref) == len(inv)
        max_diff = float(np.max(np.abs(ref_w - inv_w))) if same_count else float("nan")
        status = "OK" if same_count and max_diff < 1e-8 else "MISMATCH"
        print(f"n={n:4d}  cvxcla pts={len(ref):4d}  inverse pts={len(inv):4d}  max|dw|={max_diff:.2e}  [{status}]")


if __name__ == "__main__":
    main()
