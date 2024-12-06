---
title: Cla
marimo-version: 0.9.31
---

# Critical Line Algorithm
<!---->
We compute an efficient frontier using the critical line algorithm (cla)

```{.python.marimo}
import numpy as np

from cvx.cla.markowitz.cla import CLA as MARKOWITZ
```

```{.python.marimo}
n = 10
mean = np.random.randn(n)
lower_bounds = np.zeros(n)
upper_bounds = np.ones(n)

factor = np.random.randn(n, n)
covariance = factor @ factor.T

f1 = MARKOWITZ(
    mean=mean,
    covariance=covariance,
    lower_bounds=lower_bounds,
    upper_bounds=upper_bounds,
    A=np.ones((1, len(mean))),
    b=np.ones(1),
).frontier
```

```{.python.marimo}
f1.interpolate(10).plot(volatility=True, markers=False)
```

```{.python.marimo}
f1.plot()
```

```{.python.marimo}
import marimo as mo
```