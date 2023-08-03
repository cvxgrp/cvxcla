# [cvxcla](https://www.cvxgrp.org/cvxcla/)

[![PyPI version](https://badge.fury.io/py/cvxcla.svg)](https://badge.fury.io/py/cvxcla)
[![Apache 2.0 License](https://img.shields.io/badge/License-APACHEv2-brightgreen.svg)](https://github.com/cvxgrp/cvxcla/blob/master/LICENSE)
[![PyPI download month](https://img.shields.io/pypi/dm/cvxcla.svg)](https://pypi.python.org/pypi/cvxcla/)
[![Coverage Status](https://coveralls.io/repos/github/cvxgrp/cvxcla/badge.png?branch=main)](https://coveralls.io/github/cvxgrp/cvxcla?branch=main)

[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/cvxgrp/cvxcla)

## Critical Line Algorithm

The code in this repository is an adoption of the paper by Bailey and Lopez de Prado.
We have updated their original code and covered it in tests. We have made a few
noteworthy changes:

* Use a boolean numpy array to indicate whether a weight is free or blocked.
* Rewrote the computation of the first turning point.
* Isolated the computation of lambdas and the update of weights to make them testable.
* Use modern and immutable dataclass throughout

Note that for this project we have not addressed the more fundamental bottlenecks
of the original implementation.
We use this code as a baseline to compute to frontiers for our testing.
Some loops could be avoided by we may have to sacrifice readability.

## Poetry

We assume you share already the love for [Poetry](https://python-poetry.org).
Once you have installed poetry you can perform

```bash
make install
```

to replicate the virtual environment we have defined in [pyproject.toml](pyproject.toml)
and locked in [poetry.lock](poetry.lock).

## Kernel

We install [JupyterLab](https://jupyter.org) within your new virtual environment.
Executing

```bash
make kernel
```

constructs a dedicated [Kernel](https://docs.jupyter.org/en/latest/projects/kernels.html)
for the project.
