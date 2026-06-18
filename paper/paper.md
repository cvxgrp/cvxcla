---
title: 'cvxcla: A matrix-free Critical Line Algorithm for the mean--variance efficient frontier'
tags:
  - Python
  - portfolio optimization
  - critical line algorithm
  - quadratic programming
  - efficient frontier
authors:
  - name: Thomas Schmelzer
    orcid: 0000-0000-0000-0000  # TODO: insert ORCID before submission
    affiliation: 1
  - name: Philipp Schiele
    orcid: 0000-0000-0000-0000  # TODO: insert ORCID before submission
    affiliation: 2
affiliations:
  - name: Jebel Quant Research, Abu Dhabi, United Arab Emirates
    index: 1
  - name: Stanford University, Stanford, CA, United States
    index: 2
date: 18 June 2026
bibliography: paper.bib
---

# Summary

`cvxcla` is a Python implementation of the Critical Line Algorithm (CLA) of
Markowitz [@markowitz1952], which computes the *entire* mean--variance efficient
frontier exactly rather than sampling it point by point. The frontier is piecewise
linear in portfolio weights: between consecutive *turning points* the optimal
weights are an affine function $w(\lambda) = \alpha + \lambda\beta$ of the
risk-aversion parameter $\lambda$. The algorithm walks from the maximum-return
portfolio ($\lambda = \infty$) to the minimum-variance portfolio ($\lambda = 0$),
changing the status of one asset at each turning point via a single solve of a
reduced Karush--Kuhn--Tucker system handled by block elimination through a Schur
complement.

Its distinguishing feature is that this loop touches the covariance matrix through
only three operations: a matrix--vector product, a solve against the free-asset
block, and a free-to-blocked cross product. `cvxcla` captures this contract in a
small `QuadraticForm` protocol, decoupling the algorithm from how the covariance is
represented. Four backends implement it: a dense reference; an incremental-inverse
variant that maintains the free-block inverse across turning points; a
diagonal-plus-low-rank `FactorCovariance` that solves through the Woodbury identity
without forming an $n \times n$ matrix; and a `GramCovariance` that works directly
from the (centered) return matrix $X$, since a sample covariance is the Gram matrix
$X^\top X/(T-1)$. The event logic is hardened against degeneracy with a Bland-style
anti-cycling rule, and a reciprocal-condition-number test refuses unreliable turning
points on rank-deficient inputs. The library is small, fully type-checked, and
covered by an automated test suite.

# Statement of need

Practitioners and researchers who need the complete efficient frontier, not a
single optimal portfolio, are well served by the CLA, but existing open
implementations operate on an explicit dense covariance matrix
[@bailey2013; @niedermayer2010]. The widely used `PyPortfolioOpt` package
[@pyportfolioopt] ships the Bailey--López de Prado reference CLA [@bailey2013],
which is dense and rebuilds linear algebra per candidate asset. As the asset
universe grows into the hundreds, the dense $O(n^3)$-per-step cost dominates, and on
the short-sample covariances common in finance ($T < n$) a dense estimate is
rank-deficient and the solve is unreliable.

`cvxcla` addresses both problems through the operator abstraction. Structured risk
models, the diagonal-plus-low-rank factor models and shrinkage estimators that real
portfolio construction uses [@ledoit2004], drop in as backends and trace the *same
exact frontier* without ever materialising the dense matrix: memory and per-solve
cost scale with the model's rank rather than $n^2$. On a factor model with a few
tens of factors over several hundred assets, the factor backend traces the full
frontier orders of magnitude faster than the dense reference while returning a
numerically identical result (the minimum-variance portfolio agrees with
`PyPortfolioOpt` to machine precision). The data-matrix backend brings the same
benefit to the short-sample regime, where an optional ridge restores positive
definiteness and the solve is performed in the low-dimensional observation space.

Real mandates also impose linear equality constraints beyond the budget: dollar-neutral
long/short books, sector- or factor-neutral portfolios, or a target gross total.
`cvxcla` traces the exact frontier under any such $A w = b$. The turning-point engine is
already general in $A$; only the maximum-return starting vertex is constraint-specific,
found by a closed-form greedy fill for the all-ones budget and, for a general $A$, by a
small linear program solved with HiGHS [@huangfu2018] via SciPy [@virtanen2020]. The
covariance is never formed in either case, so the matrix-free advantage carries over to
constrained mandates unchanged.

Beyond the implementation, `cvxcla` makes explicit that the CLA is one instance of a
broader family of parametric active-set path-tracers. It is the one-parameter
specialisation of multi-parametric quadratic programming and explicit model
predictive control [@bemporad2002; @tondel2003], and it shares its mechanics with
the least angle regression and LASSO homotopy of statistical learning
[@tibshirani1996; @efron2004]. The same `QuadraticForm` protocol and path-tracing
engine that drive the portfolio frontier also trace a LASSO regularisation path,
with the covariance $\Sigma$ replaced by the Gram matrix $X^\top X$. This framing,
together with a reusable and benchmarked reference implementation, is intended to
serve both quantitative-finance users who need fast, exact frontiers at scale and
researchers studying parametric active-set methods.

# Acknowledgements

`cvxcla` originated in the Stanford University Convex Optimization Group. We thank
its contributors and the maintainers of the `rhiza` project infrastructure.

# References
