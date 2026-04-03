<div align="center">

# 📈 [cvxcla](https://www.cvxgrp.org/cvxcla) - Critical Line Algorithm for Portfolio Optimization

[![PyPI version](https://badge.fury.io/py/cvxcla.svg)](https://badge.fury.io/py/cvxcla)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Downloads](https://static.pepy.tech/personalized-badge/cvxcla?period=month&units=international_system&left_color=black&right_color=orange&left_text=PyPI%20downloads%20per%20month)](https://pepy.tech/project/cvxcla)
[![Coverage](https://raw.githubusercontent.com/cvxgrp/cvxcla/gh-pages/coverage-badge.svg)](https://www.cvxgrp.org/cvxcla/tests/html-coverage/index.html)

---

**Quick Links:**
[📖 Documentation](https://www.cvxgrp.org/cvxcla) •
[🐛 Report Bug](https://github.com/cvxgrp/cvxcla/issues) •
[💡 Request Feature](https://github.com/cvxgrp/cvxcla/issues) •
[💬 Discussions](https://github.com/cvxgrp/cvxcla/discussions)

---

</div>

## 📋 Overview

`cvxcla` is a Python package that implements the Critical Line Algorithm (CLA)
for portfolio optimization.
The CLA efficiently computes the entire efficient frontier for portfolio optimization
problems with linear constraints and bounds on the weights.

The Critical Line Algorithm was introduced by Harry Markowitz
in [The Optimization of Quadratic Functions Subject to Linear Constraints](https://www.rand.org/pubs/research_memoranda/RM1438.html)
and further described in his book [Portfolio Selection](https://www.wiley.com/en-us/Portfolio+Selection%3A+Efficient+Diversification+of+Investments%2C+2nd+Edition-p-9781557861085).

The algorithm is based on the observation that the efficient frontier
is a piecewise linear function when expected return is plotted against
expected variance. The CLA computes the turning points (corners)
of the efficient frontier, allowing for efficient representation of the entire frontier.

I gave the plenary talk at [EQD's Singapore conference](https://tschm.github.io/eqd_markowitz/PresentationEQDweb.pdf).

## 🧮 Why the Algorithm Works

The Markowitz problem is a quadratic program parametrized by a return target λ:

```
min  wᵀΣw - λ · μᵀw
s.t. Aw = b,  lb ≤ w ≤ ub
```

As λ sweeps from ∞ (maximize return) down to 0 (minimize variance), the solution
traces the entire efficient frontier. The key insight is that **between consecutive
events, the optimal weights are a linear function of λ**:

```
w(λ) = α + λ · β
```

This holds because the KKT optimality conditions are linear in λ whenever the active
set — which assets sit at their bounds — is fixed. The algorithm exploits this in
three steps:

1. **Start** at λ = ∞, where the portfolio concentrates on the highest-return asset
   within bounds
   ([`init_algo`](https://github.com/cvxgrp/cvxcla/blob/main/src/cvxcla/first.py),
   called from [`_first_turning_point`](https://github.com/cvxgrp/cvxcla/blob/main/src/cvxcla/cla.py#L223)).

2. **Solve** the KKT system for the current active set to find α and β
   ([`_solve`](https://github.com/cvxgrp/cvxcla/blob/main/src/cvxcla/cla.py#L189)),
   then decrease λ until one of two events occurs
   ([main loop](https://github.com/cvxgrp/cvxcla/blob/main/src/cvxcla/cla.py#L122)):
   - a **free** asset hits its bound (leaves the free set), or
   - a **blocked** asset's KKT multiplier changes sign (enters the free set).

3. **Update** the active set (exactly one asset changes status) and repeat until λ ≤ 0.

Because only one asset changes per step and each step requires only a single linear
solve, the algorithm traces the full frontier cheaply and exactly — no approximation
needed.

## ✨ Features

- Efficient computation of the entire efficient frontier
- Support for linear constraints and bounds on portfolio weights
- Multiple implementations based on different approaches from the literature
- Visualization of the efficient frontier using Plotly
- Computation of the maximum Sharpe ratio portfolio
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
    mean = mean,
    covariance = covariance,
    lower_bounds = lower_bounds,
    upper_bounds = upper_bounds,
    a = np.ones((1, n)),  # Fully invested constraint
    b = np.ones(1)
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

For visualization, you can plot the efficient frontier:

```python
# Plot the efficient frontier
fig = frontier.plot(volatility=True)
fig.show()
```


## 📚 Literature and Implementations

The package includes implementations based on several key papers:

### 📝 Niedermayer and Niedermayer

They suggested a method to avoid the expensive inversion
of the covariance matrix in [Applying Markowitz's critical line algorithm](https://www.researchgate.net/publication/226987510_Applying_Markowitz%27s_Critical_Line_Algorithm).
Our testing shows that in Python, this approach is not significantly
faster than explicit matrix inversion using LAPACK via `numpy.linalg.inv`.

### 📝 Bailey and Lopez de Prado

We initially started with their code published
in [An Open-Source Implementation of the Critical-Line Algorithm for Portfolio Optimization](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2197616).
We've made several improvements:

- Using boolean numpy arrays to indicate whether a weight is free or blocked
- Rewriting the computation of the first turning point
- Isolating the computation of λ and weight updates to make them testable
- Using modern and immutable dataclasses throughout

Our updated implementation is included in the tests but not part of cvxcla package.
We use it to verify our results and include it for educational purposes.

### 📝 Markowitz et al

In
[Avoiding the Downside: A Practical Review of the Critical Line Algorithm for Mean-Semivariance Portfolio Optimization](https://www.hudsonbaycapital.com/documents/FG/hudsonbay/research/599440_paper.pdf),
Markowitz and researchers from Hudson Bay Capital Management and Constantia Capital
present a step-by-step tutorial.

We address a problem they overlooked: after finding the first starting point,
all variables might be blocked. We enforce that one variable
labeled as free (even if it sits on a boundary) to avoid a singular matrix.

Rather than using their sparse matrix construction, we bisect the
weights into free and blocked parts and use a linear solver for the free part only.

## 🧪 Testing

Run the test suite with:

```bash
make test
```

## 🧹 Code Quality

Format and lint the code with:

```bash
make fmt
```

## 📖 Documentation

- [Online Documentation](https://www.cvxgrp.org/cvxcla/book)
- [API Reference](https://www.cvxgrp.org/cvxcla/pdoc/)

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

This project is licensed under the Apache License 2.0 - see the [LICENSE](LICENSE)
file for details.

## 🔍 Related Projects

- [PyCLA](https://github.com/phschiele/PyCLA) by Philipp Schiele - A
previous implementation of the Critical Line Algorithm in Python.

- [CLA](https://github.com/mdengler/cla) by Martin Dengler - The
original implementation by David Bailey and Marcos Lopez de Prado.
