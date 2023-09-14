# [cvxcla](https://www.cvxgrp.org/cvxcla/book)

[![PyPI version](https://badge.fury.io/py/cvxcla.svg)](https://badge.fury.io/py/cvxcla)
[![Apache 2.0 License](https://img.shields.io/badge/License-APACHEv2-brightgreen.svg)](https://github.com/cvxgrp/cvxcla/blob/master/LICENSE)
[![Downloads](https://static.pepy.tech/personalized-badge/cvxcla?period=month&units=international_system&left_color=black&right_color=orange&left_text=PyPI%20downloads%20per%20month)](https://pepy.tech/project/cvxcla)
[![Coverage Status](https://coveralls.io/repos/github/cvxgrp/cvxcla/badge.png?branch=main)](https://coveralls.io/github/cvxgrp/cvxcla?branch=main)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/cvxgrp/cvxcla)

## Critical line algorithm

The critical line algorithm is a method to compute the efficient frontier of a
portfolio optimization problem.

The algorithm has been introduced by Markowitz in
[The Optimization of Quadratic Functions Subject to Linear Constraints](https://www.rand.org/pubs/research_memoranda/RM1438.html)
and subsequently described in his book [Portfolio Selection](https://www.wiley.com/en-us/Portfolio+Selection%3A+Efficient+Diversification+of+Investments%2C+2nd+Edition-p-9781557861085).

The algorithm is based on the observation that the efficient frontier is a piecewise
linear function if expected return is plotted over expected variance.
The critical line algorithm computes the turning points, e.g. the corners
of the efficient frontier.

## Literature

We are using the following sources

### Niedermayer and Niedermayer

They suggested a method to avoid the expensive inversion of the covariance matrix.
[Applying Markowitz's critical line algorithm](https://www.researchgate.net/publication/226987510_Applying_Markowitz%27s_Critical_Line_Algorithm)
It turns out that implementing their method in Python is not significantly faster
than the explicit inversion of the covariance matrix relying on LAPACK via `numpy.linalg.inv`.

### Bailey and Lopez de Prado

We have initially started with their code published in
[An Open-Source Implementation of the Critical-Line Algorithm for Portfolio Optimization](https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2197616).
We have updated their original code and covered it in tests. We have made a few
noteworthy changes:

* Use a boolean numpy array to indicate whether a weight is free or blocked.
* Rewrote the computation of the first turning point.
* Isolated the computation of $\lambda$ and the update of weights to make them testable.
* Use modern and immutable dataclasses throughout.

The code is not part of the published package though.
It is only used for testing purposes. We recommend it for educational purposes only.

### Markowitz et al

In [Avoiding the Downside: A Practical Review of the Critical
Line Algorithm for Mean-Semivariance Portfolio Optimizatio](https://www.hudsonbaycapital.com/documents/FG/hudsonbay/research/599440_paper.pdf)
Markowitz and a team of researchers from Hudson Bay Capital Management and Constantia
Capital provide a step-by-step tutorial on how to implement the critical line algorithm.

We address a problem they oversaw: After finding the first starting point
all variables might be blocked. We need to enforce that one variable is labeled
as free even it sits on a boundary otherwise the matrix needed is singular.

Rather than using their involved construction of the sparse matrix
to estimate the weights we bisect the weights into a free and a blocked part.
We then use a linear solver to compute the weights only for the free part.

We alter some of their Python code. Our experiments to combine it with Niedermayer's
ideas to accelerate the computation of the matrix inverses did not yet justify
the additional complexity.

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
