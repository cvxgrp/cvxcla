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
the `for i in b` candidate loop. The per-candidate loop is what drives the
runtime gap, but the measurements below show that the dense *inverse* inside it
accounts for only a small part of that gap.

One nuance worth noting: BLdP's *removal* branch (Snippet 7) computes the
inverse once *outside* its loop, so the per-candidate rebuild is specific to the
*addition* branch — which is exactly how `cvxcla` phrases it.

### Measured: where the gap comes from (#869, track 1)

[`experiments/bldp_gap.py`](experiments/bldp_gap.py) re-measures the gap and
breaks it down. It uses PyPortfolioOpt 1.6.0, numpy 2.4.6 (Accelerate) and
cvxcla 2.6.0 on an Apple M4 Pro, with the factor-model problems of the paper's
scaling figure (K = 10, long-only, budget), 3 seeds × 3 repeats per size, and
each measurement in a fresh process. Ratios are medians over seeds; the spread
over seeds is within ±10%.

- **Code reading confirmed.** In PyPortfolioOpt's `_solve` (verified by parsing
  the shipped source, not the paper's snippet numbering), the only
  `np.linalg.inv` inside a loop is the addition branch's `for i in b`. At n = 640
  a trace performs 321 inverses per turning point, about n/2.
- **The gap grows with n, with no plateau.** BLdP/cvxcla is 2.8× at n = 20,
  16× at 80, 146× at 320 and **418×** at 640, against the ≈340× reported. Over
  the last doubling, BLdP scales like n^3.7 and cvxcla like n^2.1.
- **The gap breaks into three factors**, each from changing one thing at a time:

  | n | inverse → bordered update, loop kept | Python loop → vectorised search | minimal vectorised CLA → cvxcla | total |
  |---:|---:|---:|---:|---:|
  | 20 | 0.9× | 12× | 0.3× | 2.8× |
  | 80 | 1.0× | 37× | 0.4× | 16× |
  | 320 | 1.3× | 86× | 1.3× | 146× |
  | 640 | 1.6× | 141× | 1.9× | 418× |

  Replacing each candidate's fresh O(n_F³) inverse with an O(n_F²) bordered
  update, while keeping the loop, buys at most 1.6×. Most of the gap comes from
  evaluating O(n) candidates one at a time in Python. Each evaluation re-slices
  `covarF`/`covarFB` and re-runs the λ formula, so it costs O(n·n_F) with or
  without an inverse. A phase profile of the shipped code agrees: the
  addition-branch inverses take 8% of the trace at n = 20 and 41% at n = 640. The
  slicing takes 27–40%, and the λ formula plus the removal branch most of the
  rest. The last factor is cvxcla's own cost relative to a ~100-line vectorised
  CLA: cvxcla is 3× *slower* at n = 20 (validation and guards), and 1.9× faster at
  n = 640 (a block-eliminated solve instead of an explicit inverse). A vectorised
  CLA with an incrementally updated inverse is a further 3.9× faster than the
  fresh-inverse one, and about 2× faster than cvxcla's dense backend at n = 640.
- **BLdP can cycle.** Its re-entry test `lam < self.ls[-1]` has no tolerance and
  its loop has no iteration cap. With bordered updates, which change λ by about
  1e-15, an asset alternately leaves and re-enters at the same λ forever: 5 of
  the 18 runs of that variant never terminate. The shipped code finished on all
  18 problems, but only by the luck of its rounding. (While checking the
  vectorised variants against cvxcla, the experiment also found that
  `experiments/inverse_cla.py` used an absolute 1e-5 tie window on λ that
  stepped past turning points from n = 160 on, with weights down to −5e-6 under
  a zero bound. It now uses 1e-9.)
- **Not measured:** other CPUs or BLAS builds (`--threads` is available but was
  not swept), n > 640, other problem families, memory, or the original BLdP
  code as published rather than PyPortfolioOpt's descendant of it.

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
