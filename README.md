<div align="center" markdown="1">

# 📈 [cvxcla](https://www.cvxgrp.org/cvxcla) - Critical Line Algorithm for Portfolio Optimization

[![PyPI version](https://img.shields.io/pypi/v/cvxcla)](https://pypi.org/project/cvxcla/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Downloads](https://static.pepy.tech/personalized-badge/cvxcla?period=month&units=international_system&left_color=black&right_color=orange&left_text=PyPI%20downloads%20per%20month)](https://pepy.tech/project/cvxcla)
[![Coverage](https://www.cvxgrp.org/cvxcla/coverage-badge.svg)](https://www.cvxgrp.org/cvxcla/reports/html-coverage/index.html)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22209209.svg)](https://doi.org/10.5281/zenodo.22209209)

---

**Quick Links:**
[📖 Documentation](https://www.cvxgrp.org/cvxcla) •
[🐛 Report Bug](https://github.com/cvxgrp/cvxcla/issues) •
[💡 Request Feature](https://github.com/cvxgrp/cvxcla/issues)

---

</div>

## 📋 Overview

`cvxcla` is a Python package that implements the Critical Line Algorithm (CLA)
for portfolio optimization.
The CLA efficiently computes the entire efficient frontier for portfolio optimization
problems with linear constraints and bounds on the weights.

The same parametric active-set engine also traces the **LASSO** regularisation path
(the `Lasso` class) — including general inequality constraints `Gβ ≤ h` and the
non-negative LASSO `β ≥ 0` — so the Critical Line Algorithm and the LARS/LASSO
homotopy come out as two instances of one path-following engine.

The Critical Line Algorithm was introduced by Harry Markowitz
in [The Optimization of Quadratic Functions Subject to Linear Constraints](https://www.rand.org/pubs/research_memoranda/RM1438.html)
and further described in his book [Portfolio Selection](https://www.wiley.com/en-us/Portfolio+Selection%3A+Efficient+Diversification+of+Investments%2C+2nd+Edition-p-9781557861085).

The algorithm is based on the observation that the optimal *weights* are a
piecewise linear function of the return tilt λ. The CLA computes the turning
points (corners) where that path bends, and the whole frontier is recovered from
them exactly. In the (variance, return) plane the frontier is a chain of parabolic
arcs that meet with continuous slope, which is why its corners are invisible on a
frontier plot but show up as kinks on a LASSO coefficient path.

I gave the plenary talk at [EQD's Singapore conference](https://tschm.github.io/eqd_markowitz/PresentationEQDweb.pdf).

## 🧮 Why the Algorithm Works

The Markowitz problem is a quadratic program parametrized by a return tilt λ
(a weight on expected return, not a return target or a risk aversion):

```text
min  ½ wᵀΣw - λ · μᵀw
s.t. Aw = b,  Gw ≤ h,  lb ≤ w ≤ ub
```

where `Aw = b` are linear equalities (the budget, sector/factor neutrality, …) and
`Gw ≤ h` are linear inequalities (group or sector exposure caps; a `≥` floor is the
negated row). As λ sweeps from ∞ (maximize return) down to 0 (minimize variance), the solution
traces the entire efficient frontier. The key insight is that **between consecutive
events, the optimal weights are a linear function of λ**:

```text
w(λ) = α + λ · β
```

This holds because the KKT optimality conditions are linear in λ whenever the active
set — which assets sit at their bounds — is fixed. The algorithm exploits this in
three steps:

1. **Start** at λ = ∞, where the portfolio concentrates on the highest-return asset
   within bounds
   ([`init_algo`](https://github.com/cvxgrp/cvxcla/blob/main/src/cvxcla/first.py)).

2. **Solve** the KKT system for the current active set to find α and β by
   block elimination, then decrease λ until one of two events occurs
   (main loop in [`cla.py`](https://github.com/cvxgrp/cvxcla/blob/main/src/cvxcla/cla.py)):
   - a **free** asset hits its bound (leaves the free set), or
   - a **blocked** asset's KKT multiplier changes sign (enters the free set).

   General inequality rows `Gw ≤ h` add the row analogue of these two events: an
   inactive row becomes binding when its slack reaches zero, and an active row is
   released when its multiplier reaches zero. An active row is carried through the
   same block-eliminated solve as an extra equality row, so the structure is
   preserved.

3. **Update** the active set (exactly one asset or constraint row changes status) and
   repeat until λ ≤ 0.

Because only one coordinate changes per step and each step requires only a single
linear solve, the algorithm traces the full frontier cheaply and exactly — no
approximation needed. Ties are broken by a Bland-style lowest-index rule, which
keeps the walk deterministic and finite.

The four events are the same ones the LASSO homotopy uses, under different names
(Schmelzer and Hastie, 2026, Table 2):

| Event | Critical Line Algorithm | LASSO / LARS |
|-------|-------------------------|--------------|
| P₁ | an asset reaches a bound and leaves the free set | a coefficient crosses zero (the "leave" move) |
| P₂ | a group or exposure row `Gw ≤ h` becomes tight | a linear inequality on β becomes tight |
| D₁ | a blocked asset's reduced cost changes sign and it re-enters | an inactive correlation reaches λ (the "enter" move) |
| D₂ | an active row's multiplier hits zero and it releases | an active inequality's multiplier hits zero |

The only difference at the level of the algorithm is D₁: for a box the threshold is
fixed at zero, while for the ℓ₁ penalty it moves with λ.

## ✨ Features

- Efficient computation of the entire efficient frontier
- Fluent builder API — `CLA.problem(mean, cov).long_only().budget().trace()` — as a
  readable alternative to the explicit constructor
- Box bounds on every weight, plus general linear **equality** constraints
  `Aw = b` (budget, dollar-neutral, sector/factor neutrality) and general linear
  **inequality** constraints `Gw ≤ h` (group or sector exposure caps)
- Leverage (gross-exposure) caps `‖w‖₁ ≤ c`, traced exactly (130/30 books,
  capped dollar-neutral books)
- Factor covariance backend: exact frontiers for diagonal-plus-low-rank
  covariances (factor models, RMT-cleaned matrices) in O(nk) memory via the
  Woodbury identity
- Visualization of the efficient frontier using Plotly
- Computation of the maximum Sharpe ratio portfolio
- **LASSO regularisation path** through the same engine (`Lasso`), with the same
  fluent builder, general inequality constraints `Gβ ≤ h`, and the non-negative
  LASSO `β ≥ 0`
- Fully tested and documented codebase

## 🚀 Installation

### Using pip

```bash
pip install cvxcla
```

To include plotting support (Plotly and Kaleido):

```bash
pip install cvxcla[plot]
```

### Development Setup

To set up a development environment:

1. Clone the repository:

    ```bash
    git clone https://github.com/cvxgrp/cvxcla.git
    cd cvxcla
    ```

2. Create a virtual environment and install dependencies:

    ```bash
    make install
    ```

This will:

- Install the uv package manager
- Create a Python 3.12 virtual environment
- Install all dependencies from pyproject.toml

## 🔧 Usage

Here's a simple example of how to use `cvxcla` to compute the efficient frontier:

```python
import numpy as np

# Set a seed for reproducibility
np.random.seed(42)
from cvxcla import CLA

# Define your portfolio problem
n = 10  # Number of assets
mean = np.random.randn(n)  # Expected returns
cov = np.random.randn(n, n)
covariance = cov @ cov.T  # Covariance matrix
lower_bounds = np.zeros(n)  # No short selling
upper_bounds = np.ones(n)  # No leverage

# Create a CLA instance
cla = CLA(
    mean=mean,
    covariance=covariance,
    lower_bounds=lower_bounds,
    upper_bounds=upper_bounds,
    a=np.ones((1, n)),  # Fully invested constraint
    b=np.ones(1),
)

# Access the efficient frontier
frontier = cla.frontier

# Get the maximum Sharpe ratio portfolio
max_sharpe_ratio, max_sharpe_weights = frontier.max_sharpe
print(f"Maximum Sharpe ratio: {max_sharpe_ratio:.6f}")
# Print first few weights to avoid long output
print(f"First 3 weights: {max_sharpe_weights[:3]}")
```

```result
Maximum Sharpe ratio: 2.946979
First 3 weights: [0.         0.         0.08509841]
```

The same problem reads more declaratively through the fluent builder, reached via
`CLA.problem(...)`. Each step maps onto one constructor argument, and the terminal
`.trace()` returns the identical `CLA`:

```python
# Same fully-invested, long-only problem as above, assembled fluently.
built = CLA.problem(mean, covariance).long_only().budget().trace()

# Pure sugar over the constructor: it returns the identical frontier.
assert built.frontier.max_sharpe[0] == cla.frontier.max_sharpe[0]
```

`.bounds`, `.equality`, and `.inequality` add general bounds and constraints, as in
the examples below.

For visualization, you can plot the efficient frontier:

```python
# Plot the efficient frontier
fig = frontier.plot(volatility=True)
fig.show()
```

### Group and sector constraints

Pass `g` and `h` to add general inequality rows `Gw ≤ h` on top of the box bounds and
the budget. For example, to cap the combined weight of the first three assets (a
sector) at 40%:

```python
import numpy as np
from cvxcla import CLA

n = 10
rng = np.random.default_rng(42)
mean = rng.standard_normal(n)
factor = rng.standard_normal((n, n))
covariance = factor @ factor.T  # symmetric positive definite

# sum of the first three weights ≤ 0.40
g = np.zeros((1, n))
g[0, :3] = 1.0
h = np.array([0.40])

cla = CLA(
    mean=mean,
    covariance=covariance,
    lower_bounds=np.zeros(n),
    upper_bounds=np.full(n, 0.35),
    a=np.ones((1, n)),  # fully invested
    b=np.ones(1),
    g=g,  # inequality matrix  G  (p x n)
    h=h,  # inequality vector  h  (p,)
)

# every turning point now satisfies the sector cap
assert all(tp.weights[:3].sum() <= 0.40 + cla.tol for tp in cla.turning_points)
```

`g`/`h` default to `None`, so omitting them recovers the equality-only problem
unchanged. Stack multiple rows in `g` for several caps at once, and express a `≥`
floor by negating a row (`-Gw ≤ -h`).

### Leverage (gross-exposure) caps

Pass `leverage=c` (or chain `.leverage(c)` on the builder) to cap the gross
exposure `‖w‖₁ = Σ|wᵢ| ≤ c`. With a fully-invested budget, `c = 1.3` is a 130/30
book:

```python
import numpy as np
from cvxcla import CLA

n = 10
rng = np.random.default_rng(42)
mean = rng.standard_normal(n)
factor = rng.standard_normal((n, n))
covariance = factor @ factor.T  # symmetric positive definite

cla = (
    CLA.problem(mean, covariance)
    .bounds(-0.2, 0.45)  # shorts allowed down to -20% per asset
    .budget()  # fully invested
    .leverage(1.3)  # at most 130% long + short
    .trace()
)

assert all(np.abs(tp.weights).sum() <= 1.3 + cla.tol for tp in cla.turning_points)
```

The 1-norm is polyhedral, so the frontier is still traced exactly. Internally each
asset whose box straddles zero is split into a long and a short leg, `wᵢ = uᵢ − vᵢ`,
and the cap becomes one extra inequality row `Σuᵢ + Σvᵢ ≤ c`. The turning points
are reported in the original weights. Assets that can only be long (or only short)
are not split, so the cap is redundant for a long-only, fully-invested book.

### Factor models at scale

For diagonal-plus-low-rank covariances (factor risk models, eigenvalue-clipped
covariances) pass a `FactorCovariance` instead of a dense matrix; the algorithm
then never forms an n-by-n matrix:

```python
import numpy as np
from cvxcla import CLA, FactorCovariance

rng = np.random.default_rng(42)
n, k = 10_000, 50
covariance = FactorCovariance(
    d=rng.uniform(0.1, 0.5, n),  # idiosyncratic variances
    u=rng.standard_normal((n, k)),  # factor loadings
    delta=rng.uniform(0.5, 2.0, k),  # factor variances, (k,) or (k, k)
)
```

See the [factor backend documentation](https://www.cvxgrp.org/cvxcla/factor/)
for the protocol, the math, and benchmarks against the dense path.

### The LASSO through the same engine

The same path-tracer drives the LASSO regularisation path. `Lasso` plays the Gram
matrix `XᵀX` and the vector `Xᵀy` the way the CLA plays the covariance and the mean,
and traces the whole path from `λ_max` (where `β = 0`) down to the least-squares fit:

```python
import numpy as np
from cvxcla import Lasso

rng = np.random.default_rng(0)
X = rng.standard_normal((60, 12))
y = X @ rng.standard_normal(12) + 0.1 * rng.standard_normal(60)

# plain LASSO path; .solution(lam) evaluates beta at any penalty
lasso = Lasso(x=X, y=y)
beta = lasso.solution(0.5 * lasso.lam_max)

# the same fluent builder as the CLA
lasso = Lasso.problem(X, y).trace()
```

It carries the same constraints the CLA does. Add general inequality rows `Gβ ≤ h`
(with `h > 0`, e.g. per-group exposure caps), or restrict to the **non-negative
LASSO** `β ≥ 0` — where the ℓ₁ penalty collapses to the linear term `λ·1ᵀβ`, making
it exactly the CLA's box-bounded parametric QP:

```python
# group-exposure caps G beta <= h (cap features 0–3 and 4–7)
G = np.zeros((2, 12))
G[0, :4] = 1.0
G[1, 4:8] = 1.0
h = np.array([1.0, 1.0])
lasso = Lasso.problem(X, y).inequality(G, h).trace()

# non-negative LASSO (beta >= 0)
lasso = Lasso.problem(X, y).non_negative().trace()
```

Both return the exact path, validated breakpoint-by-breakpoint against a per-λ QP
solver. (Equality constraints `Aβ = b` need a feasibility seed and are not yet
supported; the canonical sum-to-zero case cannot be traced one coordinate at a time.)

### One curve, two literatures

The link between the two classes is more than a shared engine. Under `Σ = XᵀX` and
`μ = Xᵀy`, the gross-exposure-capped Markowitz program and the constrained LASSO
trace **the same piecewise-linear curve**, under arbitrary linear equality and
inequality constraints
([Schmelzer and Hastie, 2026](https://arxiv.org/abs/2609.25704)):

```text
M_c:  min ½ wᵀΣw − μᵀw        s.t. ‖w‖₁ ≤ c,  Aw = b,  Gw ≤ h
L_λ:  min ½‖y − Xβ‖² + λ‖β‖₁  s.t.          Aβ = b,  Gβ ≤ h
```

- **Theorem 1.** Given general position and a constraint qualification, `β(λ)` solves
  `M_c` at `c(λ) = ‖β(λ)‖₁`. Wherever `c` is strictly decreasing, the two paths
  share the same breakpoints, visited in the same order.
- **Budget versus tilt (Corollary 2).** With homogeneous constraints (`b = 0`,
  `h = 0`), sweeping the tilt λ at a fixed leverage cap `c` (which is what
  `CLA(leverage=c)` does) gives the budget-indexed path rescaled:
  `w_c(λ) = λ · w^M(c/λ)`.
- **The frontier is a LASSO path (Proposition 3).** The long-only, fully invested
  frontier is the non-negative LASSO path, rescaled radially by `t = 1/λ`.
- **Degrees of freedom (Proposition 4).** A turning point's fit has
  `df = E[|F| − rank M_F]`, where `F` is the free set and `M_F` stacks the active
  constraint rows on it. For a long-only, fully invested frontier this is the
  expected number of holdings away from a bound, less one.

In cvxcla terms, a `CLA` with `mean = Xᵀy`, `covariance = XᵀX` and
`leverage = c` reproduces the rescaled LASSO path, and so does a `Lasso` fitted on
`(X, y)`:

```python
import numpy as np
from cvxcla import CLA, Lasso

rng = np.random.default_rng(0)
X = rng.standard_normal((40, 8))
y = X @ rng.standard_normal(8) + 0.3 * rng.standard_normal(40)

lasso = Lasso(x=X, y=y)
c = 0.8 * np.abs(lasso.path[-1].beta).sum()  # a binding leverage cap
cla = CLA(
    mean=X.T @ y,
    covariance=X.T @ X,
    lower_bounds=np.full(8, -50.0),
    upper_bounds=np.full(8, 50.0),
    a=np.zeros((0, 8)),
    b=np.zeros(0),
    leverage=c,
)
# every turning point w at tilt lam is lam * beta, where ||beta||_1 = c / lam
```

The identity carries statements about the curve, such as the breakpoints, the
order in which coordinates enter, and the path length. It does not carry
statements averaged over the response at a fixed parameter, such as
post-selection intervals, because a fixed λ and a fixed `c` are different
experiments.

## 🧪 Testing

Run the test suite with:

```bash
make test
```

or directly via pytest:

```bash
uv run pytest
```

## 🧹 Code Quality

Format and lint the code with:

```bash
make fmt
```

## 📖 Documentation

- [Online Documentation](https://www.cvxgrp.org/cvxcla/)
- [Factor backend](https://www.cvxgrp.org/cvxcla/factor/)
- [Notebooks](https://www.cvxgrp.org/cvxcla/notebooks/cla.html)
- [Test report](https://www.cvxgrp.org/cvxcla/reports/html-report/report.html)
- [Coverage report](https://www.cvxgrp.org/cvxcla/reports/html-coverage/index.html)
- [The Critical Line Algorithm and the Constrained LASSO: One Curve, Two Literatures](https://arxiv.org/abs/2609.25704)
  (Schmelzer and Hastie, 2026)

## 📚 Citing

If you use `cvxcla`, please cite the software through its
[Zenodo DOI](https://doi.org/10.5281/zenodo.22209208) (see [`CITATION.cff`](CITATION.cff)).
If you use the CLA–LASSO correspondence, please also cite:

```bibtex
@misc{schmelzer2026onecurve,
  title         = {The Critical Line Algorithm and the Constrained {LASSO}:
                   One Curve, Two Literatures},
  author        = {Schmelzer, Thomas and Hastie, Trevor},
  year          = {2026},
  eprint        = {2609.25704},
  archivePrefix = {arXiv},
  primaryClass  = {stat.ME},
  url           = {https://arxiv.org/abs/2609.25704}
}
```

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/amazing-feature`)
3. Run the tests to make sure everything works (`make test`)
4. Format your code (`make fmt`)
5. Commit your changes (`git commit -m 'Add some amazing feature'`)
6. Push to the branch (`git push origin feature/amazing-feature`)
7. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE)
file for details.

## 🔍 Related Projects

- [PyCLA](https://github.com/phschiele/PyCLA) by Philipp Schiele - A
previous implementation of the Critical Line Algorithm in Python.

- [CLA](https://github.com/mdengler/cla) by Martin Dengler - The
original implementation by David Bailey and Marcos Lopez de Prado.
