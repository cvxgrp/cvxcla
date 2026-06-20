"""Timing: scikit-learn versus cvxcla for the LASSO path, with and without constraints.

The LASSO analogue of the CLA scaling sweep (``runtime_scaling.py``). For ``n``
features in ``{20, 40, 80, 160, 320, 640}`` (the CLA dimensions) and ``m = 2n``
observations, we trace the entire regularisation path and time the wall-clock as the
median of REPEATS repetitions.

The apples-to-apples comparison is the *exact* path both ways:

* **Unconstrained** -- ``cvxcla.Lasso`` against scikit-learn's ``lars_path`` (the
  canonical exact LARS/LASSO path). Both return the same breakpoints; we verify the
  coefficients agree before reporting the times.
* **Non-negative** (``beta >= 0``) -- ``cvxcla.Lasso(nonneg=True)`` against
  scikit-learn's ``LassoLars(positive=True)``.
* **Inequality** (``G beta <= h``) -- ``cvxcla`` only: scikit-learn has no
  general inequality-constrained LASSO path, so this column is reported for context
  with no scikit-learn baseline. (Per-lambda QP sweeps are the only general-solver
  alternative; cf. the Clarabel baseline in ``clarabel_baseline.py``.)

scikit-learn parametrises the penalty as ``(1/2m)||y - X beta||^2 + alpha||beta||_1``;
cvxcla uses ``(1/2)||y - X beta||^2 + lambda||beta||_1``, so ``lambda = m * alpha``.

Prints a table and, when matplotlib is available, writes docs/paper/lasso_timing.pdf.

Usage:
    uv run --with scikit-learn --with matplotlib python experiments/lasso_timing.py
"""

from __future__ import annotations

import time

import numpy as np

from cvxcla import Lasso

SIZES = [20, 40, 80, 160, 320, 640]
SEED = 7
REPEATS = 3


def make_problem(rng: np.random.Generator, n: int) -> tuple[np.ndarray, np.ndarray]:
    """A standardised ``2n x n`` regression with a sparse ground truth."""
    m = 2 * n
    x = rng.standard_normal((m, n))
    x = x - x.mean(0)
    beta = np.zeros(n)
    support = rng.choice(n, max(1, n // 5), replace=False)
    beta[support] = rng.standard_normal(support.size)
    y = x @ beta + 0.1 * rng.standard_normal(m)
    return x, y - y.mean()


def group_caps(n: int, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-group exposure caps ``G beta <= h`` (groups of 3, h > 0, tuned to bind)."""
    groups = np.arange(n) // 3
    p = int(groups.max()) + 1
    g = np.array([(groups == j).astype(float) for j in range(p)])
    h = np.maximum(np.abs(g @ np.linalg.lstsq(x, y, rcond=None)[0]) * 0.5, 0.05)
    return g, h


def median_time(fn: object) -> float:
    """Median wall-clock seconds of calling ``fn`` REPEATS times."""
    times = []
    for _ in range(REPEATS):
        start = time.perf_counter()
        fn()  # type: ignore[operator]
        times.append(time.perf_counter() - start)
    return float(np.median(times))


def main() -> None:
    """Run the sweep, verify cvxcla matches scikit-learn, print a table, write the figure."""
    from sklearn.linear_model import LassoLars, lars_path

    ns, sk, cvx, sk_nn, cvx_nn, cvx_ineq, points = [], [], [], [], [], [], []
    worst_match = 0.0
    for n in SIZES:
        rng = np.random.default_rng(SEED)
        x, y = make_problem(rng, n)
        m = x.shape[0]
        g, h = group_caps(n, x, y)

        # exact-path agreement: cvxcla vs scikit-learn lars_path (lambda = m * alpha)
        path = Lasso(x=x, y=y)
        alphas, _active, coefs = lars_path(x, y, method="lasso")
        match = max(float(np.max(np.abs(path.solution(m * a) - coefs[:, i]))) for i, a in enumerate(alphas) if a > 0)
        worst_match = max(worst_match, match)

        t_sk = median_time(lambda x=x, y=y: lars_path(x, y, method="lasso"))
        t_cvx = median_time(lambda x=x, y=y: Lasso(x=x, y=y))
        t_sk_nn = median_time(lambda x=x, y=y: LassoLars(alpha=0.0, positive=True).fit(x, y))
        t_cvx_nn = median_time(lambda x=x, y=y: Lasso(x=x, y=y, nonneg=True))
        t_cvx_ineq = median_time(lambda x=x, y=y, g=g, h=h: Lasso(x=x, y=y, g=g, h=h))

        ns.append(n)
        points.append(len(path.path))
        sk.append(t_sk)
        cvx.append(t_cvx)
        sk_nn.append(t_sk_nn)
        cvx_nn.append(t_cvx_nn)
        cvx_ineq.append(t_cvx_ineq)
        print(
            f"n={n:4d}  bps={len(path.path):4d}  match={match:.1e}   "
            f"sklearn={t_sk * 1e3:8.1f} ms  cvxcla={t_cvx * 1e3:8.1f} ms   "
            f"[nonneg] sklearn={t_sk_nn * 1e3:8.1f} ms  cvxcla={t_cvx_nn * 1e3:8.1f} ms   "
            f"[G<=h] cvxcla={t_cvx_ineq * 1e3:8.1f} ms"
        )

    print(f"\nmax |beta_cvxcla - beta_sklearn| over all sizes/breakpoints: {worst_match:.1e}")

    def slope(t: list[float]) -> float:
        return float(np.polyfit(np.log(ns[-4:]), np.log(t[-4:]), 1)[0])

    print(f"empirical exponent p (time ~ n^p): sklearn {slope(sk):.2f}, cvxcla {slope(cvx):.2f}")

    try:
        import matplotlib as mpl

        mpl.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.ticker import NullFormatter, ScalarFormatter
    except ImportError:
        print("matplotlib not available - skipping docs/paper/lasso_timing.pdf")
        return

    fig, ax = plt.subplots(figsize=(5.0, 3.4))
    ax.loglog(ns, sk, "-^", ms=4, color="#7f7f7f", label="scikit-learn LARS (lars\\_path)")
    ax.loglog(ns, cvx, "-o", ms=4, color="#c00000", label="cvxcla (unconstrained)")
    ax.loglog(ns, cvx_nn, "-s", ms=4, color="#1f4e79", label="cvxcla ($\\beta \\geq 0$)")
    ax.loglog(ns, cvx_ineq, "-D", ms=4, color="#2ca02c", label="cvxcla ($G\\beta \\leq h$)")
    ax.set_xlabel("Number of features $n$")
    ax.set_ylabel("Full-path trace time [s]")
    ax.set_title("LASSO path: scikit-learn vs cvxcla", fontsize=9)
    ax.set_xticks(ns)
    ax.xaxis.set_major_formatter(ScalarFormatter())
    ax.xaxis.set_minor_formatter(NullFormatter())
    ax.set_xlim(ns[0] * 0.85, ns[-1] * 1.18)
    ax.tick_params(axis="x", labelsize=8)
    ax.grid(True, which="both", alpha=0.3)
    ax.legend(fontsize=7.5)
    fig.tight_layout()
    out = "docs/paper/lasso_timing.pdf"
    fig.savefig(out)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
