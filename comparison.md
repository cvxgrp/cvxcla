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
  n = 640 (a block-eliminated solve instead of an explicit inverse).
- **Maintaining the inverse pays off at scale, in both codes.** Updating the
  free-block inverse by rank-one steps instead of rebuilding it makes the
  minimal CLA 3.9× faster at n = 640, and makes cvxcla 2.5× faster with
  `IncrementalDenseCovariance` instead of the default dense backend (no gain
  below n = 80). With matching linear algebra, cvxcla's incremental backend is
  3× slower than the minimal CLA at n = 20 but 1.25× *faster* at n = 640
  (204 ms against 247 ms), and it scales better (n^1.7 against n^2.3 over the
  last doubling). These numbers come from a separate run of the four fast
  variants; the BLdP columns above were not re-timed with it.
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

### Measured: do they compute the same frontier? (#869, track 2)

[`experiments/bldp_agree.py`](experiments/bldp_agree.py) feeds the same `mean`,
`covariance`, `lB` and `uB` to PyPortfolioOpt 1.6.0's `CLA` and to cvxcla 2.7.0. It
checks each trace on its own against CVXPY 1.9.3 / Clarabel 0.11.1 at tolerance
1e-12, at every turning point and every segment midpoint, and compares the two
paths `w(λ)` exactly. Both use the same λ convention. There are 90 problems in
eight families over 3 seeds: generic, shorting, all-equal means, tied means,
duplicated assets, rank-deficient sample covariances, a conditioning sweep from
1e2 to 1e16, and bounds that bind exactly. "Exact" means feasible to 1e-9, λ
running from ∞ to 0, and a relative objective gap below 1e-8 everywhere.

- **On generic problems the shipped code is often wrong.** cvxcla is exact on all
  57 feasible non-degenerate problems except two at κ ≥ 1e14 (below). The shipped
  BLdP code is exact on only 21 of them. The rest lose real turning points, with
  objective gaps up to 3e-4 and paths up to 0.35 apart in max-norm, and no error
  is raised. Even plain long-only problems fail on 4 of 12 (n = 10 to 100). Two bugs explain all
  of it:
  1. **Stale blocked weights.** When a free asset hits a bound, `_solve` sets the
     working vector `w[i] = bound`. `_get_matrices` then reads the blocked weights
     from `self.w[-1]`, the *previous* turning point. The new point is solved with
     asset `i` at its old free value, so its budget is off by exactly
     `bound − w_prev[i]` (e.g. −4.1e-2). `_purge_num_err` deletes it as infeasible,
     and a true vertex goes with it. Sometimes the trace recovers instead by
     removing, re-adding and removing the asset at the same λ. That leaves three
     copies of one point, the same leave-and-re-enter pattern that track 1 saw
     cycle forever under bordered updates. Whether these stale weights cause that
     cycle was not checked. Reading the
     blocked weights from the working vector (`bldp-fix-w`, a two-method subclass)
     makes 52 of the 57 exact.
  2. **A zero-tolerance purge.** `_purge_excess` deletes a turning point whenever a
     later one has a strictly higher `μ'w`. Where `μ'w` is flat, rounding makes it
     rise by about 1e-14, and a true vertex is deleted. With a 1e-12 relative
     tolerance as well (`bldp-fix-wp`), 56 of 57 are exact with gaps ≤ 7e-13. The
     remaining case cycles at κ = 1e16.

  With both fixes, BLdP and cvxcla trace the same path to 1e-8 on every one of
  these problems except the two below. The defects are in the implementation,
  not in the algorithm.
- **Degenerate inputs break BLdP; cvxcla handles every one.** cvxcla is exact on all 30
  problems in the all-equal, tied, duplicated and rank-deficient families. BLdP,
  with or without the two fixes:
  - *All-equal means:* raises `TypeError` on all 6. PyPortfolioOpt ships the
    paper's `mean[-1] += 1e-5` hack commented out. With it switched back on, BLdP
    still raises for n = 20, and for n = 5 traces the frontier of a perturbed
    problem whose λ is scaled by 1/1e-5 (gap ≈ 4e-5). The true frontier is the
    single minimum-variance point.
  - *Ties in expected returns:* raises `TypeError` on 7 of 9. When the
    event-search denominator `c` is exactly 0, `_compute_lambda` returns `None`,
    and the comparison `lam < self.ls[-1]` fails on it. Once, rounding kept `c`
    off zero and the trace ran silently down the wrong path: `_init_algo` starts at
    one vertex of the tied face and never frees the other tied assets, giving a
    gap of 0.25. cvxcla starts at the face minimiser.
  - *Duplicated assets / rank-deficient Σ:* `LinAlgError` on 8 of 9 duplicated
    problems and `TypeError` on the 3 problems with T < n/2. Worse, when an exact
    duplicate of the top asset enters later, `np.linalg.inv` of the singular free
    block (cond 1.5e16) does not raise. Every later point violates the budget,
    `_purge_num_err` deletes them all, and the frontier silently stops at
    λ = 11.4 instead of 0. With T = 30 < n = 40 the fixed code is exact on 2 of
    3 problems and goes wrong near λ = 0 on the third (gap 2e-6).
  - *Infeasible bounds* (`sum(uB) < 1`): `IndexError` from `_init_algo` walking
    off the asset list. cvxcla raises `InfeasibleProblemError`.
- **Which degrades first with conditioning?** With both bugs fixed, BLdP's
  explicit inverse holds up surprisingly well: gaps ≤ 4e-13 up to κ = 1e14, and
  one cycle at 1e16. cvxcla stays exact up to 1e12. At κ = 1e14 and 1e16 it has
  one problem each with a gap of 5e-8 and 8e-8. That is on a segment at
  λ ≈ 1e-10, where its weights are 2e-3 off the QP's along a near-null direction
  of Σ, while BLdP-fixed and the QP agree to 1e-14. As shipped, BLdP is wrong
  from κ = 1e2 because of the two bugs.
- **Not measured:** the 2013 paper's own code as published rather than
  PyPortfolioOpt's descendant (whether bugs 1 and 2 are in the original is not
  checked here), the Markowitz–Todd 10-asset example, n > 100, and constraint
  classes beyond a budget and box bounds (BLdP supports nothing else).

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
