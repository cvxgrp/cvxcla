# Writing a covariance backend

The critical line algorithm never sees the covariance matrix. It reaches it
through five operations, collected in the protocol `cvxcla.QuadraticForm` (the
`cvx.linalg.SymmetricOperator` contract). Any object implementing them can be
passed as `covariance=` to `CLA`, and the algorithm runs unchanged: the bundled
`DenseCovariance`, `FactorCovariance` and `GramCovariance` are three such
backends, and you can add your own.

## The contract

A backend subclasses `QuadraticForm` and implements:

| Member | Returns | Used for |
|---|---|---|
| `n` | the number of assets | sizes |
| `matvec(x)` | `Sigma @ x` (`x` a vector or a matrix of columns) | gradients, the free-to-blocked cross product |
| `block_matvec(rows, cols, v)` | `Sigma[rows, cols] @ v` | the gross-exposure (leverage) lift |
| `solve_free(free, rhs)` | `Sigma[free, free]^{-1} @ rhs` | the reduced KKT solve, once per turning point |
| `rcond_free(free)` | the reciprocal condition number of `Sigma[free, free]`, in `[0, 1]` | the singularity guard |

The algorithm assumes `Sigma` is symmetric and that every free block it solves
is positive definite; `rcond_free` lets it detect a block that is not and stop
with an error that names the problem instead of trusting the solve.

## Conventions

These are the conventions `CLA` relies on. `tests/test_backend_protocol.py`
checks each of them for every backend it tests, and is a template for testing
your own.

**Arrays.** Every array passed in is `float64` (vectors and right-hand sides)
or `numpy.intp` (index sets). Return `float64` arrays.

**Index sets.** `rows`, `cols` and `free` are 1-d integer arrays into
`range(n)` without duplicates, but *not necessarily sorted*: the gross-exposure
lift passes the assets of the free legs in leg order. Results are aligned with
the order given, so `solve_free(free, rhs)[i]` belongs to asset `free[i]`.
`CLA` never calls `solve_free` with an empty free set; `rcond_free` of an empty
set should return `1.0`.

**Shapes.** `matvec` takes `x` of shape `(n,)` or `(n, k)` and returns the
same shape. `block_matvec(rows, cols, v)` takes `v` of shape `(len(cols),)` or
`(len(cols), k)` and returns `(len(rows),)` or `(len(rows), k)`. `solve_free`
takes `rhs` of shape `(len(free),)` or `(len(free), k)` and returns the same
shape. `CLA` passes a matrix: the reduced KKT solve stacks the constraint
columns and its two right-hand sides into one multi-column solve, so `k` is the
number of active constraint rows plus two. NumPy broadcasting usually gives both
shapes for free.

**No mutation, no aliasing.** Do not modify any argument in place, and do not
return an array that aliases an argument or a buffer a later call overwrites:
the caller keeps the results (the turning points hold them). A backend may keep
internal state between calls, as `IncrementalDenseCovariance` does, provided the
results do not depend on the call history beyond round-off.

**Accuracy.** The frontier is exact to the accuracy of `solve_free`. A backend
that solves its free blocks only approximately (an iterative solver stopped
early, say) traces the frontier of the matrix it effectively applies, with errors
of the order of its relative residual; the KKT residuals of the result are a
direct check. `matvec` and `block_matvec` must agree with the matrix the solve
inverts, or the event search and the solve describe different problems.

**Conditioning.** `rcond_free(free)` returns the reciprocal 2-norm condition
number of `Sigma[free, free]`, `lambda_min / lambda_max`, in `[0, 1]`, or a
*lower bound* on it (the factor backend returns a bound from Weyl's
inequalities, so it never forms the block). `CLA` declines a free block below
`cvxcla.operators.RCOND_FLOOR` (`1e-12`), so a lower bound errs on the safe side
and an overestimate can let a singular solve through. `CLA` first calls
`rcond_free(range(n))` once: if the whole covariance clears the floor, no free
block can fall below it (eigenvalue interlacing) and the per-step check is
skipped, so that call must not overstate the conditioning either.

**Errors.** Validate the data when the backend is built and raise `ValueError`
for malformed or inadmissible input, as the bundled builders do. During the
trace, a `solve_free` that meets a singular block may raise
`numpy.linalg.LinAlgError`. `CLA` does not translate it: with a correct
`rcond_free` the guard declines first, with a `DegenerateProblemError`.

## Example: a block-diagonal covariance

Assets in groups with no cross-group covariance give a block-diagonal `Sigma`.
Its free-block solve splits into one small Cholesky solve per block:

```python
import numpy as np
from scipy.linalg import block_diag, cho_factor, cho_solve

from cvxcla import CLA, QuadraticForm


class BlockDiagonalCovariance(QuadraticForm):
    """Sigma = blockdiag(S_1, ..., S_m) with symmetric positive-definite blocks."""

    def __init__(self, blocks):
        self._blocks = [np.asarray(b, dtype=float) for b in blocks]
        sizes = [len(b) for b in self._blocks]
        self._starts = np.cumsum([0, *sizes])
        self._block_of = np.repeat(np.arange(len(sizes)), sizes)

    @property
    def n(self):
        return int(self._starts[-1])

    def matvec(self, x):
        return np.concatenate([b @ x[s:e] for b, s, e in zip(self._blocks, self._starts[:-1], self._starts[1:])])

    def block_matvec(self, rows, cols, v):
        x = np.zeros((self.n, *np.shape(v)[1:]))
        x[np.asarray(cols)] = v
        return self.matvec(x)[np.asarray(rows)]

    def _free_blocks(self, free):
        free = np.asarray(free)
        for k in np.unique(self._block_of[free]):
            pos = np.flatnonzero(self._block_of[free] == k)
            local = free[pos] - self._starts[k]
            yield pos, self._blocks[k][np.ix_(local, local)]

    def solve_free(self, free, rhs):
        out = np.empty_like(np.asarray(rhs, dtype=float))
        for pos, sub in self._free_blocks(free):
            out[pos] = cho_solve(cho_factor(sub), rhs[pos])
        return out

    def rcond_free(self, free):
        eig = np.concatenate([np.linalg.eigvalsh(sub) for _, sub in self._free_blocks(free)])
        return float(eig.min() / eig.max())


rng = np.random.default_rng(4)
blocks = [m @ m.T / 5 + 0.2 * np.eye(5) for m in rng.standard_normal((4, 5, 5))]
mean = rng.uniform(0.0, 1.0, 20)
problem = dict(lower_bounds=np.zeros(20), upper_bounds=np.ones(20), a=np.ones((1, 20)), b=np.ones(1))

custom = CLA(mean=mean, covariance=BlockDiagonalCovariance(blocks), **problem)
dense = CLA(mean=mean, covariance=block_diag(*blocks), **problem)
assert np.allclose([tp.weights for tp in custom.turning_points], [tp.weights for tp in dense.turning_points])
```

The same backend works with general constraints and with a gross-exposure cap
(`leverage=`); `tests/test_custom_backend.py` checks all three against the dense
trace. The structure need not be block-diagonal: a sparse matrix with a sparse
Cholesky, a Kronecker product, or a matrix-free operator with an iterative
`solve_free` fit the same five methods. `tests/test_backend_protocol.py` takes
four independently written backends (block-diagonal, a hand-written
diagonal-plus-low-rank Woodbury solve, a Kronecker product and a permuted dense
matrix) through the conventions above and through randomized problems with and
without general constraints, each against the dense trace of the same matrix.
