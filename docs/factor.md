# Factor covariance backend

RMT-cleaned covariances (constant-residual-eigenvalue / eigenvalue clipping)
and commercial factor risk models share the structure

$$\Sigma = D + U \Delta U^\top, \qquad D \succ 0 \text{ diagonal},\; U \in \mathbb{R}^{n \times k},\; k \ll n,$$

a diagonal matrix plus a low-rank term. The `FactorCovariance` backend exploits
this structure through the Woodbury identity: the critical line algorithm runs
in $O(n(k+m))$ memory and each turning point costs $O(n_F k^2 + k^3)$ instead
of $O((n_F+m)^3)$ — no $n \times n$ matrix is ever formed.

## Usage

```python
import numpy as np
from cvxcla import CLA, FactorCovariance

n, k = 10_000, 50
rng = np.random.default_rng(0)

covariance = FactorCovariance(
    d=rng.uniform(0.1, 0.5, n),            # idiosyncratic variances
    u=rng.standard_normal((n, k)),         # factor loadings
    delta=rng.uniform(0.5, 2.0, k),        # factor (co)variances, (k,) or (k, k)
)

frontier = CLA(
    mean=rng.uniform(0.0, 0.1, n),
    covariance=covariance,                 # any ndarray still works as before
    lower_bounds=np.zeros(n),
    upper_bounds=np.ones(n),
    a=np.ones((1, n)),
    b=np.ones(1),
).frontier
```

`CLA` accepts either a plain `numpy` covariance matrix (wrapped automatically
in `DenseCovariance`) or any object implementing the `CovarianceOperator`
protocol:

```python
class CovarianceOperator(Protocol):
    n: int
    def matvec(self, x): ...               # Sigma @ x
    def solve_free(self, free, rhs): ...   # Sigma[free][:, free]^{-1} @ rhs
    def cross(self, free, x): ...          # Sigma[free][:, ~free] @ x[~free]
```

## RMT-cleaned covariances

Eigenvalue clipping at the Marchenko–Pastur edge produces exactly this
structure: keeping the $k$ eigenpairs $(\lambda_i, v_i)$ above the edge and
replacing the rest by their average $\bar{d}$ gives

$$\Sigma_{\text{clean}} = \bar{d}\, I + V_k (\Lambda_k - \bar{d} I) V_k^\top,$$

i.e. `FactorCovariance(d=np.full(n, d_bar), u=v_k, delta=lambda_k - d_bar)`.
See the [factor notebook](notebooks.md) for an end-to-end example: simulated
returns → Marchenko–Pastur threshold → factors → exact frontier.

## Benchmark

Full frontier (all turning points) of random long-only factor-model problems,
$k = n/20$, budget constraint, run with `experiments/factor_benchmark.py`
(Apple Silicon, single seed; the dense column wraps the same problem in a
plain `numpy` matrix):

| n | k | turning points | dense [s] | factor [s] | speedup |
|---:|---:|---:|---:|---:|---:|
| 500 | 25 | 505 | 0.42 | 0.08 | 5x |
| 2000 | 100 | 2051 | 31.90 | 1.34 | 24x |
| 5000 | 250 | 5166 | 726.72 | 20.08 | 36x |

At $n = 20{,}000$, $k = 100$ the dense path would need 3.2 GB for the
covariance alone; the factor backend traces a frontier at that size in a
fraction of a second (see `tests/test_operators.py::TestFactorLargeScale`).

The dense path spends $O((n_F+m)^3)$ per turning point on the free-block
solve (plus the $O(n^2)$ memory for the matrix itself), while the factor
path solves the same system through the $k \times k$ Woodbury correction
matrix.

## References

- Perold, *Large-Scale Portfolio Optimization*, Management Science 30(10), 1984.
- Niedermayer & Niedermayer, *Applying Markowitz's Critical Line Algorithm*, 2010.
- Schmelzer, Stoll, Wolf, *From Marchenko–Pastur to Woodbury: Direct Solvers
  for Long-Only Mean-Variance Portfolios*, 2026.
