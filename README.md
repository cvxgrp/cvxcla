# [cvxcla](https://www.cvxgrp.org/cvxcla/)

[![PyPI version](https://badge.fury.io/py/cvxcla.svg)](https://badge.fury.io/py/cvxcla)
[![Apache 2.0 License](https://img.shields.io/badge/License-APACHEv2-brightgreen.svg)](https://github.com/cvxgrp/cvxcla/blob/master/LICENSE)
[![PyPI download month](https://img.shields.io/pypi/dm/cvxcla.svg)](https://pypi.python.org/pypi/cvxcla/)
[![Coverage Status](https://coveralls.io/repos/github/cvxgrp/cvxcla/badge.png?branch=main)](https://coveralls.io/github/cvxgrp/cvxcla?branch=main)


## Poetry

We assume you share already the love for [Poetry](https://python-poetry.org). Once you have installed poetry you can perform

```bash
poetry install
```

to replicate the virtual environment we have defined in [pyproject.toml](pyproject.toml) and locked in [poetry.lock](poetry.lock).

## Kernel

We install [JupyterLab](https://jupyter.org) within your new virtual environment. Executing

```bash
./create_kernel.sh
```

constructs a dedicated [Kernel](https://docs.jupyter.org/en/latest/projects/kernels.html) for the project.
