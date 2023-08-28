# [cvxcla](https://www.cvxgrp.org/cvxcla/book)

[![PyPI version](https://badge.fury.io/py/cvxcla.svg)](https://badge.fury.io/py/cvxcla)
[![Apache 2.0 License](https://img.shields.io/badge/License-APACHEv2-brightgreen.svg)](https://github.com/cvxgrp/cvxcla/blob/master/LICENSE)
[![PyPI download month](https://img.shields.io/pypi/dm/cvxcla.svg)](https://pypi.python.org/pypi/cvxcla/)
[![Coverage Status](https://coveralls.io/repos/github/cvxgrp/cvxcla/badge.png?branch=main)](https://coveralls.io/github/cvxgrp/cvxcla?branch=main)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/cvxgrp/cvxcla)

## Critical line algorithm

The critical line algorithm is a method to compute the efficient frontier of a
portfolio optimization problem.

The algorithm has been introduced by Markowitz in
[The Optimization of Quadratic Functions Subject to Linear Constraints](https://www.rand.org/pubs/research_memoranda/RM1438.html)
and subsequently described in his book [Portfolio Selection](https://www.wiley.com/en-us/Portfolio+Selection%3A+Efficient+Diversification+of+Investments%2C+2nd+Edition-p-9781557861085).
Bailey and Lopez de Prado revisited (a special case of) this algorithm in their paper
[An Open-Source Implementation of the Critical-Line Algorithm for Portfolio Optimization](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2197616).
There the authors publish a Python implementation of the algorithm.

The algorithm is based on the observation that the efficient frontier is a piecewise
linear function if expected return is plotted over expected variance.
The critical line algorithm computes the turning points, e.g. the corners
of the efficient frontier.

## About the code

The code in this repository is an adoption of the paper by Bailey and Lopez de Prado.
We have updated their original code and covered it in tests. We have made a few
noteworthy changes:

* Use a boolean numpy array to indicate whether a weight is free or blocked.
* Rewrote the computation of the first turning point.
* Isolated the computation of $\lambda$ and the update of weights to make them testable.
* Use modern and immutable dataclasses throughout.
* Use GitHub Actions to run tests, create documentation and deploy to PyPI.

Note that for this project we have not addressed the more fundamental bottlenecks
of the original implementation.
We use this code as a baseline to compute frontiers for our tests in a
forthcoming more radical implementation of the algorithm.

## Poetry

We assume you share already the love for [Poetry](https://python-poetry.org).
Once you have installed poetry you can perform

```bash
make install
```

to replicate the virtual environment we have defined in [pyproject.toml](pyproject.toml)
and locked in [poetry.lock](poetry.lock).

## Jupyter

We install [JupyterLab](https://jupyter.org) on fly within the aforementioned
virtual environment. Executing

```bash
make jupyter
```

will install and start the jupyter lab.
