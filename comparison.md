# `cvxcla` vs. Bailey–López de Prado (2013)

A comparison of the `cvxcla` paper (Schmelzer, *The Critical Line Algorithm as
implemented in cvxcla*) with **Bailey, D. H. and López de Prado, M. (2013), "An
Open-Source Implementation of the Critical-Line Algorithm for Portfolio
Optimization", *Algorithms* 6(1):169–196** (SSRN abstract 2197616).

This is not an arbitrary comparison: Bailey–López de Prado (BLdP) is the
`bailey2013` reference that `cvxcla` benchmarks against, and the PyPortfolioOpt
`CLA` used in the experiments is an implementation of it. The comparison
therefore doubles as a fact-check of `cvxcla`'s §8 ("Relation to the
Bailey–López de Prado algorithm").

## Shared lineage

Both papers compute the **entire** efficient frontier exactly as a finite list
of Markowitz turning points, in Python. They share several concrete design
choices:

- **First turning point.** The same greedy construction: start every weight at
  its lower bound, fill assets in descending expected-return order until the
  budget binds; the last (partially filled) asset is the sole free asset
  (BLdP `initAlgo`, Snippet 2; `cvxcla` `init_algo`, §3).
- **Minimum-variance endpoint.** Both append an explicit turning point at
  λ = 0, and both explicitly note this **departs from Niedermayer et al.**, who
  keep searching for turning points at negative λ.
- **Maximum-Sharpe portfolio.** Both locate it by a one-dimensional search over
  the convex combination of the two turning points that bracket the discrete
  Sharpe maximum (BLdP: golden-section search, Snippets 13–14; `cvxcla`:
  one-dimensional search, §6).
- **Frontier interpolation.** Both reconstruct intermediate frontier portfolios
  as convex combinations of adjacent turning points.

## Head-to-head differences (verified against the BLdP source)

| Axis | Bailey–López de Prado (2013) | `cvxcla` |
|---|---|---|
| Covariance representation | Explicit dense matrix, sliced via `getMatrices` / `reduceMatrix` (Snippet 5) | Operator protocol (`matvec`, `solve_free`, `cross`) + structured backends |
| Free-block linear algebra | Explicit `np.linalg.inv(covarF)` at each step (Snippets 3, 7–9) | Single block-eliminated KKT solve via an `m×m` Schur complement; no explicit inverse |
| Asset-entry search | Full inverse recomputed **inside** the candidate loop `for i in b` (Snippet 3) → O(n) inverses per turning point | Vectorised event search; one solve per step |
| Structured risk models | Not supported | Diagonal-plus-low-rank factor backend via the Woodbury identity (no `n×n` matrix ever formed) |
| Constraints | Box bounds + single (budget) equality only (inputs: `mean, covar, lB, uB`) | Arbitrary `m` linear equality constraints `A w = b` |
| Degeneracy handling | No general anti-cycling rule; a single hack (`mean[-1]+=1e-5`) for all-equal means | Bland-style lowest-index tie-break + √ε slope floor (§4.3–4.4) |

### The central claim checks out

`cvxcla`'s most load-bearing §8/§9.3 claim — that PyPortfolioOpt "recomputes
that full inverse once for every candidate" in the asset-entry search — is
literally BLdP Snippet 3: `covarF_inv = np.linalg.inv(covarF)` sits **inside**
the `for i in b` candidate loop. This single structural fact explains the
≈340× runtime gap and is faithfully represented in `cvxcla`.

One nuance worth noting: BLdP's *removal* branch (Snippet 7) computes the
inverse once *outside* its loop, so the per-candidate rebuild is specific to the
*addition* branch — which is exactly how `cvxcla` phrases it. The
characterization is accurate and fair.

## Where each paper is stronger

### Bailey–López de Prado, on its own terms

- **Pedagogy.** It is a tutorial: 17 complete code snippets, step by step. It
  established *the* open-source CLA reference (it became PyPortfolioOpt and is
  widely cited).
- **Novelty-at-the-time.** In 2013 it filled a genuine gap — the first
  documented open-source CLA in a scientific language (prior art was the
  Markowitz–Todd VBA-Excel implementation). That is higher-impact novelty than
  an interface refinement.
- **Ground-truth validation.** It reproduces the canonical Markowitz–Todd
  10-asset example to 15 significant figures.

### `cvxcla`

- **Scale demonstrated.** BLdP only ever runs its n = 10 toy example. `cvxcla`
  shows n = 640 benchmarks, an n = 20,000 factor stress test, and a real
  n = 494 S&P 500 frontier.
- **Robustness.** BLdP never discusses conditioning, cycling, or tolerances and
  has no general anti-cycling rule. `cvxcla` hardens the event logic *and*
  honestly documents where it still fails (§9.6).
- **Representation-agnosticism.** The covariance-operator abstraction (and the
  resulting sub-quadratic factor backend) is the one element not present in
  BLdP — or, per `cvxcla`, in any standard CLA implementation.
- **Independent exactness check.** `cvxcla` validates against a general convex
  QP solver (CVXPY/Clarabel) at solver precision, rather than against a single
  known reference answer.

## Verdict

BLdP is a **pedagogical reference implementation** that established the
open-source baseline; `cvxcla` is an **engineering / performance paper** that
re-architects that baseline — operator abstraction, block elimination,
structured backends, hardened degeneracy handling — and quantifies the payoff.
`cvxcla`'s characterization of BLdP is accurate and fair; if anything it
*under-credits* BLdP's pedagogical completeness, which a single sentence in §8
could acknowledge.
