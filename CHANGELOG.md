# Changelog

All notable changes to this project are documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com),
and entries are generated from [Conventional Commits](https://www.conventionalcommits.org).

## [1.8.0] - 2026-06-19

### New Features
- *(operators)* IncrementalDenseCovariance — opt-in incremental free-block inverse (#677)
- *(cla)* Support general equality constraints A w = b (#711)
- *(cla)* Support general inequality constraints G w ≤ h (#715)
- *(cla)* Add a fluent ProblemBuilder convenience layer (#734)

### Bug Fixes
- *(cla)* Diagnose covariance degeneracy with an actionable error (#669)
- *(cla)* Correct the degeneracy diagnosis; characterize and visualize it (#692)
- *(cla)* Project sub-tolerance round-off so near-degenerate traces complete (#693)
- *(cla)* Project onto the capped simplex so tie-heavy traces complete (#708) (#710)

### Documentation
- Compile notebooks and reports (follow jquantstats standard) (#667)
- *(paper)* Sharpen novelty framing, termination rigor, and speedup attribution (#675)
- *(paper)* Vectorised explicit-inverse baseline + equal benchmark protocol (#671, #670) (#676)
- Link compiled paper PDF in book nav (#680)
- *(paper)* Note the 2.7 max Sharpe is measured in-sample (#682)
- Fix coverage badge link to published report path (#683)
- *(paper)* Dispersion bands, hardware note, contributions, BLdP credit (#691)
- *(paper)* Introduction history, pinned source links, and framing fixes for cla.tex (#709)
- *(paper)* Explain in sec. 11 why slack variables do not extend the CLA (#712)
- *(paper)* Sharpen the Bailey-Lopez de Prado comparison (sec. 6) (#714)
- *(readme)* Document general inequality constraints G w <= h (#716)
- *(paper)* Cite Markowitz et al. (2021) mean-semivariance CLA review (#717)
- *(readme)* Remove the Literature and Implementations section (#718)
- *(paper)* Make algorithm sections consistent with A w = b and G w <= h (#720) (#724)
- Make the S&P 500 experiment reproduce from a frozen committed snapshot (#721) (#725)
- Add a usage section with runnable code and a paper replication script (#719) (#726)
- *(paper)* Broaden and reframe the software comparison (#722) (#727)
- *(paper)* Add a "Regularization through the operator" subsection (sec. 5) (#728)
- *(paper)* Call Bailey-Lopez de Prado an implementation, not an algorithm (#729)
- *(paper)* Narrow Table 1 by dropping units and the cvxcla prefix from headers (#732)
- *(paper)* Unify the step-length wording as "smallest positive step to the next breakpoint" (#733)
- *(paper)* Demonstrate the LARS/LASSO equivalence and address minor referee points (#741)
- *(paper)* Add a general-solver baseline -- Clarabel swept over lambda (#742)
- *(paper)* Surface the degeneracy and general-A limitations up front (#743)

### Performance
- *(cla)* Skip the per-turning-point degeneracy guard when the covariance is well conditioned (#731)

### Maintenance
- Add Rhiza Claude commands (/rhiza_quality, /rhiza_update) (#656)
- Sync rhiza template v0.18.8 → v0.19.0 (#655)
- Chore(deps)(deps): bump python-multipart from 0.0.29 to 0.0.31 (#661)
- Chore(deps)(deps): bump starlette from 1.2.0 to 1.3.1 (#662)
- Cover the last two defensive branches (100% coverage) (#663)
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 3 updates (#666)
- *(operators)* Cover degenerate-pivot fallbacks for 100% coverage (#681)
- *(cla)* Decompose the turning-point trace into focused helpers (#695)
- Defend 100% coverage in the gate (#696)
- Move CONTRIBUTING and CODE_OF_CONDUCT to the repository root (#704)
- *(make)* Add paper timing-benchmark targets with pinned BLAS threads (#730)

### Other Changes
- Quality roadmap: degeneracy fixes, property tests, doc and DX cleanups (#653)
- Sync Rhiza template v0.19.0 → v0.19.1 + add github-paper bundle (#657)
- Numerical experiments for the CLA paper (200x2000 frontier + runtime scaling) (#658)
- Pin rhiza_scorecard to the permissions-fixed reusable workflow (temporary) (#660)
- Paper polish — 20x50 frontier, scaling axis, authorship, acknowledgements, references (#659)
- Delete .claude/commands/rhiza.md (#664)
- Strengthen the paper — contribution, Bailey comparison, rank + real-data experiments, exactness (#668)
- Sync Rhiza template v0.19.1 → v0.19.3 (#678)
- Move paper/ → docs/paper/ to match Rhiza PAPER_DIR (#679)
- Extract a generic parametric active-set path tracer (CLA + LASSO) (#697)
- Remove PyPI and conda publishing from the release workflow (#705)

## [1.7.0] - 2026-06-11

### New Features
- Covariance backend abstraction (Phase 1 of #646) (#647)

### Maintenance
- Update rhiza template to v0.18.8 (#643)
- Chore(deps)(deps): bump the github-actions group with 9 updates (#644)
- Chore(deps-dev)(deps-dev): bump marimo in the python-dependencies group (#645)

### Other Changes
- Remove test_github_targets
- Bump version 1.6.3 → 1.7.0

## [1.6.3] - 2026-06-04

### New Features
- Add profiles: github-project to template.yml

### Bug Fixes
- Replace broken badge.fury.io PyPI badge with shields.io
- *(deps)* Update dependency kaleido to v1.3.0
- Resolve sync conflicts (accept template version)
- Update workflow references from rhiza v0.14.0 to v0.15.2
- Pin rhiza workflows to fix commit (setup-uv v7.6.0)
- Strip whitespace from PYTHON_VERSION and update test assertion
- Add classifiers and doctest to resolve skipped rhiza tests
- Add lower bound for pytest to fix lowest-direct resolution
- Skip plot tests when plotly is not installed

### Documentation
- Add docstrings to nested functions to reach 100% docs coverage

### Dependencies
- *(deps)* Update github/codeql-action action to v4.35.2
- *(deps)* Update dependency astral-sh/uv to v0.11.7
- *(deps)* Update astral-sh/setup-uv action to v8.1.0
- *(deps)* Update dependency marimo to v0.23.3
- *(deps)* Update dependency astral-sh/uv to v0.11.8
- *(deps)* Update dependency marimo to v0.23.4
- *(deps)* Update github/codeql-action action to v4.35.3
- *(deps)* Update dependency marimo to v0.23.5
- *(deps)* Update dependency astral-sh/uv to v0.11.9
- *(deps)* Update dependency astral-sh/uv to v0.11.10
- *(deps)* Update dependency astral-sh/uv to v0.11.11
- *(deps)* Update github/codeql-action action to v4.35.4
- *(deps)* Update dependency astral-sh/uv to v0.11.12

### Maintenance
- Sync rhiza template v0.9.5 → v0.10.3
- Chore(deps-dev)(deps-dev): bump marimo in the python-dependencies group
- Merge main into rhiza/update-0.10.3, resolve conflicts
- Update template.lock sync timestamp
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 2 updates
- Chore(deps)(deps): bump github/codeql-action in the github-actions group
- Chore(deps)(deps): bump the python-dependencies group with 2 updates
- Chore(deps)(deps): bump idna from 3.11 to 3.15
- Chore(deps)(deps): bump pymdown-extensions from 10.21.2 to 10.21.3
- Sync with rhiza v0.11.0
- Restore workflow files removed by previous sync
- Sync .github workflows from cvxrisk
- Sync .rhiza/tests from cvxrisk
- Sync with rhiza template v0.15.2
- Apply rhiza sync v0.15.3
- Apply rhiza sync v0.17.0
- Apply rhiza sync v0.18.4
- Add pip dependabot entry for .rhiza/requirements
- Chore(deps-dev)(deps-dev): bump marimo in the python-dependencies group
- Chore(deps)(deps): bump the github-actions group with 8 updates
- Update uv.lock with latest dependency versions
- Chore(deps)(deps): bump the python-dependencies group with 2 updates
- Chore(deps)(deps): bump the github-actions group with 9 updates (#641)

### Other Changes
- Merge pull request #604 from cvxgrp/renovate/github-codeql-action-4.x
- Merge pull request #605 from cvxgrp/renovate/astral-sh-uv-0.x
- Merge pull request #606 from cvxgrp/renovate/astral-sh-setup-uv-8.x
- Merge pull request #610 from cvxgrp/dependabot/uv/python-dependencies-763d8570f9
- Merge pull request #611 from cvxgrp/dependabot/pip/dot-rhiza/requirements/python-dotenv-1.2.2
- Merge pull request #612 from cvxgrp/rhiza/update-0.10.3
- Merge pull request #609 from cvxgrp/renovate/marimo-0.x
- Merge pull request #614 from cvxgrp/renovate/astral-sh-uv-0.x
- Merge pull request #615 from cvxgrp/renovate/marimo-0.x
- Merge pull request #616 from cvxgrp/renovate/github-codeql-action-4.x
- Initial plan
- Fix coverage badge URL in README to point to GitHub Pages
- Merge pull request #618 from cvxgrp/copilot/fix-coverage-badge-readme
- Merge pull request #620 from cvxgrp/renovate/kaleido-1.x
- Merge pull request #619 from cvxgrp/renovate/marimo-0.x
- Merge pull request #621 from cvxgrp/renovate/astral-sh-uv-0.x
- Merge pull request #622 from cvxgrp/renovate/astral-sh-uv-0.x
- Merge pull request #623 from cvxgrp/renovate/astral-sh-uv-0.x
- Merge pull request #624 from cvxgrp/renovate/github-codeql-action-4.x
- Merge pull request #625 from cvxgrp/renovate/astral-sh-uv-0.x
- Delete renovate.json
- Merge pull request #626 from cvxgrp/tschm-patch-1
- Merge pull request #627 from cvxgrp/dependabot/uv/python-dependencies-f78d620da7
- Merge pull request #628 from cvxgrp/dependabot/github_actions/github-actions-bcb0c4251a
- Merge pull request #629 from cvxgrp/dependabot/uv/python-dependencies-1da79af2d9
- Merge pull request #630 from cvxgrp/dependabot/uv/idna-3.15
- Merge pull request #631 from cvxgrp/dependabot/uv/pymdown-extensions-10.21.3
- Remove unused templates from template.yml
- Merge pull request #632 from cvxgrp/rhiza11
- Delete .github/workflows/rhiza_devcontainer.yml
- Merge pull request #634 from cvxgrp/rhiza_v0.15.1
- Update repository reference version to v0.15.2
- Merge pull request #635 from cvxgrp/tschm-patch-1
- Merge pull request #636 from cvxgrp/rhiza_v0.15.3
- Merge pull request #637 from cvxgrp/rhiza_v0.17.0
- Merge pull request #638 from cvxgrp/rhiza_v0.18.4
- Merge pull request #640 from cvxgrp/dependabot/uv/python-dependencies-46a3b006cf
- Merge pull request #639 from cvxgrp/dependabot/github_actions/github-actions-f379237d3f
- Merge pull request #642 from cvxgrp/dependabot/uv/python-dependencies-d3db5dae90
- Ignore ocaml folder
- Bump version 1.6.2 → 1.6.3

## [1.6.2] - 2026-04-14

### Bug Fixes
- Add mkdocs.yml to set correct site name and repo for cvxcla
- Add markdown="1" to README div for proper rendering

### Dependencies
- *(deps)* Update dependency marimo to v0.22.4
- *(deps)* Update dependency astral-sh/uv to v0.11.5
- *(deps)* Update peter-evans/create-pull-request action to v8.1.1
- *(deps)* Update dependency marimo to v0.23.1
- *(deps)* Update dependency astral-sh/uv to v0.11.6
- *(deps)* Update actions/upload-artifact action to v7.0.1
- *(deps)* Update actions/upload-pages-artifact action to v5
- *(deps)* Update dependency mosek to v11.1.11

### Maintenance
- Chore(deps-dev)(deps-dev): bump marimo from 0.22.4 to 0.23.0
- Chore(deps)(deps): bump pytest from 9.0.2 to 9.0.3
- Chore(deps)(deps): bump plotly in the python-dependencies group
- Update rhiza template to v0.9.5
- Sync with rhiza template v0.9.5
- Sync rhiza template files
- Move marimo notebooks into book/marimo/notebooks/

### Other Changes
- Merge pull request #588 from cvxgrp/renovate/marimo-0.x
- Merge pull request #593 from cvxgrp/dependabot/uv/marimo-0.23.0
- Merge pull request #591 from cvxgrp/renovate/astral-sh-uv-0.x
- Merge pull request #595 from cvxgrp/renovate/peter-evans-create-pull-request-8.x
- Merge pull request #590 from cvxgrp/renovate/marimo-0.x
- Merge pull request #594 from cvxgrp/renovate/astral-sh-uv-0.x
- Merge pull request #596 from cvxgrp/renovate/actions-upload-artifact-7.x
- Merge pull request #597 from cvxgrp/renovate/actions-upload-pages-artifact-5.x
- Merge pull request #599 from cvxgrp/renovate/mosek-11.x
- Merge pull request #600 from cvxgrp/dependabot/uv/pytest-9.0.3
- Merge pull request #602 from cvxgrp/dependabot/uv/python-dependencies-f9e657d83c
- Update cla.py
- Update .env
- Update cla.py
- Merge pull request #601 from cvxgrp/rhiza/update-0.9.5
- Delete docs/DEVCONTAINER.md
- Merge pull request #603 from cvxgrp/rhiza/update-0.9.5
- Bump version 1.6.1 → 1.6.2

## [1.6.1] - 2026-04-03

### Bug Fixes
- Update coverage badge to point to gh-pages SVG

### Dependencies
- *(deps)* Update github/codeql-action action to v4.35.1
- *(deps)* Update dependency astral-sh/uv to v0.11.3
- *(deps)* Update astral-sh/setup-uv action to v8

### Maintenance
- Remove devcontainer and fix SECURITY.md for cvxcla

### Other Changes
- Merge pull request #584 from cvxgrp/renovate/github-codeql-action-4.x
- Merge pull request #583 from cvxgrp/renovate/astral-sh-uv-0.x
- Merge pull request #585 from cvxgrp/renovate/astral-sh-setup-uv-8.x
- Initial plan
- Merge pull request #587 from cvxgrp/copilot/fix-coverage-badge
- Delete REPOSITORY_ANALYSIS.md
- Bump version 1.6.0 → 1.6.1

## [1.6.0] - 2026-04-03

### Bug Fixes
- Ignore CVE-2026-4539 in pip-audit (no fix available for pygments 2.19.2)
- Add missing uv install step to pre-commit CI job

### Dependencies
- *(deps)* Update github/codeql-action action to v4.33.0
- *(deps)* Update dependency marimo to v0.21.0
- *(deps)* Update dependency astral-sh/uv to v0.10.11
- *(deps)* Update dependency mosek to v11.1.10
- *(deps)* Update astral-sh/setup-uv action to v7.6.0
- *(deps)* Update dependency marimo to v0.21.1
- *(deps)* Update dependency astral-sh/uv to v0.10.12
- *(deps)* Update github/codeql-action action to v4.34.0
- *(deps)* Update github/codeql-action action to v4.34.1
- *(deps)* Update dependency astral-sh/uv to v0.11.0
- *(deps)* Update dependency astral-sh/uv to v0.11.1
- *(deps)* Update github/codeql-action action to v4.35.0
- *(deps)* Update astral-sh/setup-uv action to v8
- *(deps)* Update github/codeql-action action to v4.35.1
- *(deps)* Update actions/deploy-pages action to v5
- *(deps)* Update dependency astral-sh/uv to v0.11.2
- *(deps)* Update dependency pandas to v3.0.2
- *(deps)* Update dependency marimo to v0.22.0
- *(deps)* Update dependency astral-sh/uv to v0.11.3
- *(deps)* Update docker/login-action action to v4.1.0

### Maintenance
- Chore(deps)(deps): bump pygments from 2.19.2 to 2.20.0
- Chore(deps)(deps): bump numpy in the python-dependencies group
- Sync with rhiza template v0.8.20
- Ignore ocaml/ directory

### Other Changes
- Merge pull request #555 from cvxgrp/tschm-patch-100
- Merge pull request #557 from cvxgrp/renovate/github-codeql-action-4.x
- Merge pull request #561 from cvxgrp/renovate/marimo-0.x
- Merge pull request #560 from cvxgrp/renovate/astral-sh-uv-0.x
- Merge pull request #558 from cvxgrp/renovate/mosek-11.x
- Merge pull request #559 from cvxgrp/renovate/astral-sh-setup-uv-7.x
- Update template.yml for version and template changes
- Resolve rhiza template sync to v0.8.13
- Merge pull request #562 from cvxgrp/tschm-patch-160
- Merge pull request #563 from cvxgrp/renovate/marimo-0.x
- Merge pull request #564 from cvxgrp/renovate/astral-sh-uv-0.x
- Merge pull request #565 from cvxgrp/renovate/github-codeql-action-4.x
- Merge pull request #566 from cvxgrp/renovate/github-codeql-action-4.x
- Merge pull request #568 from cvxgrp/renovate/astral-sh-uv-0.x
- Merge pull request #569 from cvxgrp/renovate/astral-sh-uv-0.x
- Merge pull request #572 from cvxgrp/renovate/github-codeql-action-4.x
- Merge pull request #576 from cvxgrp/dependabot/uv/pygments-2.20.0
- Merge pull request #574 from cvxgrp/renovate/astral-sh-setup-uv-8.x
- Merge pull request #573 from cvxgrp/renovate/github-codeql-action-4.x
- Merge pull request #570 from cvxgrp/renovate/actions-deploy-pages-5.x
- Merge pull request #571 from cvxgrp/renovate/astral-sh-uv-0.x
- Merge pull request #577 from cvxgrp/dependabot/uv/python-dependencies-1670832c3b
- Merge pull request #578 from cvxgrp/renovate/pandas-3.x
- Merge pull request #579 from cvxgrp/renovate/marimo-0.x
- Merge pull request #580 from cvxgrp/renovate/astral-sh-uv-0.x
- Merge pull request #581 from cvxgrp/renovate/docker-login-action-4.x
- Update reference version in template.yml
- Merge pull request #582 from cvxgrp/tschm-patch-200
- Bump version 1.5.4 → 1.6.0

## [1.5.4] - 2026-03-15

### Dependencies
- *(deps)* Update actions/download-artifact action to v8.0.1
- *(deps)* Update dependency mosek to v11.1.9
- *(deps)* Update astral-sh/setup-uv action to v7.5.0
- *(deps)* Update dependency astral-sh/uv to v0.10.10
- *(deps)* Update softprops/action-gh-release action to v2.5.1
- *(deps)* Update softprops/action-gh-release action to v2.5.2
- *(deps)* Update ncipollo/release-action action to v1.21.0

### Other Changes
- Merge pull request #542 from cvxgrp/renovate/actions-download-artifact-8.x
- Merge pull request #543 from cvxgrp/renovate/mosek-11.x
- Merge pull request #544 from cvxgrp/renovate/astral-sh-setup-uv-7.x
- Merge pull request #545 from cvxgrp/renovate/astral-sh-uv-0.x
- Merge pull request #546 from cvxgrp/renovate/softprops-action-gh-release-2.x
- Merge pull request #547 from cvxgrp/renovate/softprops-action-gh-release-2.x
- Remove mypy configuration from pyproject.toml
- Merge pull request #548 from cvxgrp/mypy_remove
- Make plotly and kaleido an optional plot extra
- Merge pull request #549 from cvxgrp/optionalPlot
- Add algorithmic explanation with source links to README
- Merge pull request #550 from cvxgrp/enhanceREADME
- Delete experiments/fusion2.py
- Merge pull request #551 from cvxgrp/tschm-patch-1
- Delete experiments/data directory
- Merge pull request #552 from cvxgrp/tschm-patch-2
- Update template.yml to ref v0.8.12 and add exclusions
- Ignore ocaml folder
- Sync
- Merge pull request #553 from cvxgrp/tschm-patch-100
- Merge pull request #554 from cvxgrp/renovate/ncipollo-release-action-1.x
- Bump version 1.5.3 → 1.5.4

## [1.5.3] - 2026-03-11

### Dependencies
- *(deps)* Update github/codeql-action action to v4.32.5
- *(deps)* Update dependency mosek to v11.1.7
- *(deps)* Update dependency astral-sh/uv to v0.10.8
- *(deps)* Update dependency marimo to v0.20.4
- *(deps)* Update docker/login-action action to v4
- *(deps)* Update dependency mosek to v11.1.8
- *(deps)* Update github/codeql-action action to v4.32.6
- *(deps)* Update dependency astral-sh/uv to v0.10.9
- *(deps)* Update astral-sh/setup-uv action to v7.4.0

### Maintenance
- Chore(deps)(deps): bump plotly in the python-dependencies group
- Chore(deps)(deps): bump numpy in the python-dependencies group

### Other Changes
- Merge pull request #530 from cvxgrp/renovate/github-codeql-action-4.x
- Merge pull request #529 from cvxgrp/renovate/mosek-11.x
- Merge pull request #531 from cvxgrp/dependabot/uv/python-dependencies-698941b3fe
- Merge pull request #533 from cvxgrp/renovate/astral-sh-uv-0.x
- Merge pull request #532 from cvxgrp/renovate/marimo-0.x
- Merge pull request #534 from cvxgrp/renovate/docker-login-action-4.x
- Merge pull request #535 from cvxgrp/renovate/mosek-11.x
- Merge pull request #536 from cvxgrp/renovate/github-codeql-action-4.x
- Merge pull request #537 from cvxgrp/renovate/astral-sh-uv-0.x
- Merge pull request #539 from cvxgrp/dependabot/uv/python-dependencies-9e55b512a1
- Merge pull request #540 from cvxgrp/renovate/astral-sh-setup-uv-7.x
- Update reference version to v0.8.9
- Sync rhiza template to v0.12.1, resolve merge conflicts
- Add sync-experimental Makefile target
- Merge pull request #541 from cvxgrp/tschm-patch-1
- Bump version 1.5.2 → 1.5.3

## [1.5.2] - 2026-02-28

### Bug Fixes
- Resolve all pre-commit hook issues
- Resolve mypy type checking errors
- Add explicit float cast to resolve mypy no-any-return error
- Replace np.min/np.max with built-in min/max for type safety
- Add --repo flag to gh release list in release workflow
- Use comment instead of classifier for Private :: Do Not Upload

### Dependencies
- *(deps)* Lock file maintenance (#459)
- *(deps)* Update dependency astral-sh/uv to v0.9.26 (#461)
- *(deps)* Update dependency marimo to v0.19.4
- *(deps)* Update dependency mosek to v11.1.3
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.26
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.14.13 (#465)
- *(deps)* Lock file maintenance (#466)
- *(deps)* Update dependency marimo to v0.19.6 (#468)
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.36.1
- *(deps)* Update dependency astral-sh/uv to v0.9.27 (#470)
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.27
- *(deps)* Update dependency pandas to v3
- *(deps)* Lock file maintenance (#473)
- *(deps)* Update dependency astral-sh/uv to v0.9.28 (#475)
- *(deps)* Update pre-commit hook abravalheri/validate-pyproject to v0.25
- *(deps)* Update dependency marimo to v0.19.7
- *(deps)* Update github/codeql-action action to v4.32.1
- *(deps)* Lock file maintenance (#479)
- *(deps)* Update dependency marimo to v0.19.9 (#481)
- *(deps)* Update dependency mosek to v11.1.5 (#482)
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.30
- *(deps)* Update github/codeql-action action to v4.32.2
- *(deps)* Update astral-sh/setup-uv action to v7.3.0
- *(deps)* Update dependency astral-sh/uv to v0.10.0
- *(deps)* Update pre-commit hook rhysd/actionlint to v1.7.11
- *(deps)* Update dependency astral-sh/uv to v0.10.2
- *(deps)* Lock file maintenance
- *(deps)* Lock file maintenance (#493)
- *(deps)* Update dependency astral-sh/uv to v0.10.3 (#495)
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.36.2
- *(deps)* Update pre-commit hook astral-sh/uv-pre-commit to v0.10.3
- *(deps)* Update actions/download-artifact action to v7
- *(deps)* Update dependency jebel-quant/rhiza to v0.8.0
- *(deps)* Update dependency astral-sh/uv to v0.10.4
- *(deps)* Update dependency pandas to v3.0.1 (#501)
- *(deps)* Update pre-commit hook astral-sh/uv-pre-commit to v0.10.4
- *(deps)* Update github/codeql-action action to v4.32.4 (#503)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.15.2 (#504)
- *(deps)* Update dependency marimo to v0.20.1 (#505)
- *(deps)* Update dependency marimo to v0.20.2 (#506)
- *(deps)* Lock file maintenance (#507)
- *(deps)* Update dependency astral-sh/uv to v0.10.6 (#510)
- *(deps)* Update dependency jebel-quant/rhiza to v0.8.3
- *(deps)* Update pre-commit hook astral-sh/uv-pre-commit to v0.10.6 (#512)
- *(deps)* Update pre-commit hook pycqa/bandit to v1.9.4
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.15.4 (#514)
- *(deps)* Update actions/attest-build-provenance action to v4
- *(deps)* Update dependency astral-sh/uv to v0.10.7
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.37.0
- *(deps)* Update dependency jebel-quant/rhiza to v0.8.4
- *(deps)* Update actions/attest-sbom action to v4
- *(deps)* Update github artifact actions
- *(deps)* Lock file maintenance
- *(deps)* Update astral-sh/setup-uv action to v7.3.1 (#522)
- *(deps)* Update dependency jebel-quant/rhiza to v0.8.5
- *(deps)* Update astral-sh/setup-uv action to v7.3.1

### Maintenance
- Update via rhiza
- Add deptry package_module_name_map configuration
- Update via rhiza
- Chore(deps-dev)(deps-dev): bump the python-dependencies group with 2 updates
- Replace assert statements with proper exception raising
- Update test to expect ValueError instead of AssertionError
- Increase test coverage to 100%
- Replace release.yml symlink with manual PyPI publish workflow

### Other Changes
- Delete .rhiza.env
- Merge pull request #462 from cvxgrp/renovate/marimo-0.x
- Merge pull request #463 from cvxgrp/renovate/mosek-11.x
- Merge pull request #464 from cvxgrp/renovate/ghcr.io-astral-sh-uv-0.x
- Merge pull request #467 from cvxgrp/rhiza/21342542406
- Merge pull request #469 from cvxgrp/renovate/python-jsonschema-check-jsonschema-0.x
- Merge pull request #471 from cvxgrp/renovate/ghcr.io-astral-sh-uv-0.x
- Merge pull request #472 from cvxgrp/renovate/pandas-3.x
- Merge pull request #474 from cvxgrp/rhiza/21573534992
- Merge pull request #478 from cvxgrp/renovate/abravalheri-validate-pyproject-0.x
- Merge pull request #476 from cvxgrp/renovate/marimo-0.x
- Merge pull request #477 from cvxgrp/renovate/github-codeql-action-4.x
- Merge pull request #483 from cvxgrp/renovate/ghcr.io-astral-sh-uv-0.x
- Merge pull request #484 from cvxgrp/renovate/github-codeql-action-4.x
- Merge pull request #485 from cvxgrp/renovate/astral-sh-setup-uv-7.x
- Merge pull request #486 from cvxgrp/renovate/astral-sh-uv-0.x
- Refactor template.yml for new repository format
- Sync
- Merge pull request #488 from cvxgrp/dependabot/uv/python-dependencies-b3f1a5853e
- Merge pull request #491 from cvxgrp/renovate/rhysd-actionlint-1.x
- Merge pull request #487 from cvxgrp/renovate/astral-sh-uv-0.x
- Merge pull request #492 from cvxgrp/renovate/lock-file-maintenance
- Merge pull request #497 from cvxgrp/renovate/python-jsonschema-check-jsonschema-0.x
- Merge pull request #496 from cvxgrp/renovate/astral-sh-uv-pre-commit-0.x
- Merge pull request #499 from cvxgrp/renovate/major-github-artifact-actions
- Sync
- Merge pull request #498 from cvxgrp/renovate/jebel-quant-rhiza-0.x
- Merge pull request #500 from cvxgrp/renovate/astral-sh-uv-0.x
- Merge pull request #502 from cvxgrp/renovate/astral-sh-uv-pre-commit-0.x
- Conftest
- Merge pull request #511 from cvxgrp/renovate/jebel-quant-rhiza-0.x
- Merge pull request #513 from cvxgrp/renovate/pycqa-bandit-1.x
- Merge pull request #515 from cvxgrp/renovate/actions-attest-build-provenance-4.x
- Merge pull request #516 from cvxgrp/renovate/astral-sh-uv-0.x
- Merge pull request #518 from cvxgrp/renovate/python-jsonschema-check-jsonschema-0.x
- Merge pull request #517 from cvxgrp/renovate/jebel-quant-rhiza-0.x
- Merge pull request #519 from cvxgrp/renovate/actions-attest-sbom-4.x
- Merge pull request #520 from cvxgrp/renovate/major-github-artifact-actions
- Merge pull request #521 from cvxgrp/renovate/lock-file-maintenance
- Sync
- Merge pull request #523 from cvxgrp/renovate/jebel-quant-rhiza-0.x
- Add renovate.json
- Merge pull request #525 from cvxgrp/renovate/configure
- Merge pull request #526 from cvxgrp/renovate/astral-sh-setup-uv-7.x
- Delete .rhiza/history
- Merge pull request #527 from cvxgrp/tschm-patch-1
- Analysis.md
- Release
- Bump version 1.5.1 → 1.5.2

## [1.5.1] - 2026-01-13

### Dependencies
- *(deps)* Lock file maintenance (#452)
- *(deps)* Lock file maintenance (#454)
- *(deps)* Update dependency astral-sh/uv to v0.9.24 (#456)
- *(deps)* Update dependency marimo to v0.19.2 (#457)
- *(deps)* Update dependency mosek to v11.1.2

### Maintenance
- Update via rhiza

### Other Changes
- Update README.md
- Add quick links section to README
- README badges
- Merge pull request #451 from cvxgrp/tschm-patch-1
- Merge pull request #455 from cvxgrp/rhiza/20904664345
- Sync
- Merge pull request #458 from cvxgrp/renovate/mosek-11.x
- Bump version 1.5.0 → 1.5.1

## [1.5.0] - 2026-01-02

### Other Changes
- Do not update rhiza_release

## [1.4.4] - 2026-01-02

### Bug Fixes
- *(deps)* Update dependency mosek to v11.0.30 (#406)
- *(deps)* Update dependency kaleido to v1.2.0 (#407)
- *(deps)* Update dependency pytest to v9
- *(deps)* Update dependency marimo to v0.18.0 (#408)
- *(deps)* Update dependency pre-commit to v4.5.0
- *(deps)* Update dependency marimo to v0.18.1
- *(deps)* Update dependency marimo to v0.18.3
- *(deps)* Update dependency pytest to v9.0.2
- *(deps)* Update dependency marimo to v0.18.4
- *(deps)* Update dependency pre-commit to v4.5.1 (#439)

### Dependencies
- *(deps)* Lock file maintenance (#403)
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.14 (#414)
- *(deps)* Update softprops/action-gh-release action to v2.5.0
- *(deps)* Lock file maintenance
- *(deps)* Lock file maintenance (#420)
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.16 (#422)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.14.8 (#423)
- *(deps)* Lock file maintenance (#432)
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.36.0 (#436)
- *(deps)* Lock file maintenance (#437)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.14.10 (#438)
- *(deps)* Lock file maintenance (#440)
- *(deps)* Update dependency astral-sh/uv to v0.9.20 (#442)
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.20 (#443)

### Maintenance
- Sync template files
- Sync template files
- Sync template files
- Sync template files
- Sync template files
- Sync template files
- Import rhiza templates
- Chore(deps)(deps): bump actions/checkout from 4 to 6

### Other Changes
- Merge pull request #404 from cvxgrp/template-updates
- Merge pull request #410 from cvxgrp/renovate/pytest-9.x
- Merge pull request #117 from tschm/template-updates
- Merge pull request #405 from tschm/main
- Merge pull request #409 from cvxgrp/renovate/pre-commit-4.x
- Merge pull request #118 from tschm/template-updates
- Merge pull request #412 from cvxgrp/template-updates
- Merge pull request #415 from cvxgrp/renovate/marimo-0.x
- Merge pull request #416 from cvxgrp/renovate/softprops-action-gh-release-2.x
- Delete .github/workflows/devcontainer.yml
- Merge pull request #413 from cvxgrp/tschm-patch-1
- Merge pull request #411 from cvxgrp/renovate/lock-file-maintenance
- Merge branch 'main' into main
- Merge pull request #417 from tschm/main
- Delete taskfiles directory
- Delete Taskfile.yml
- Delete tests/test_taskfile.py
- Merge pull request #119 from tschm/tschm-patch-1
- Merge pull request #418 from tschm/main
- Update template.yml to exclude taskfiles
- Merge pull request #419 from tschm/main
- Delete .github/workflows/docker.yml
- Exclude docker.yml from GitHub template
- Merge pull request #424 from cvxgrp/renovate/marimo-0.x
- Merge pull request #425 from cvxgrp/renovate/pytest-9.x
- Update template.yml
- Merge pull request #426 from cvxgrp/tschm-patch-1
- Initial plan
- Add comprehensive test coverage for cvxcla package
- Run make fmt - fix linting and formatting issues
- Update tests/test_cla.py
- Merge pull request #428 from cvxgrp/copilot/write-test-for-package
- Delete tests/test_readme.py
- Merge pull request #430 from cvxgrp/tschm-patch-3
- Delete tests/test_makefile.py
- Merge pull request #429 from cvxgrp/tschm-patch-2
- Merge pull request #431 from cvxgrp/template-updates
- Merge pull request #433 from cvxgrp/template-updates
- Update template repository in template.yml
- Delete .github/workflows/_devcontainer.yml
- Delete .github/workflows/structure.yml
- Delete .github/scripts/build-extras.sh
- Update cla.py
- Clean up commented code in cla.py
- Update LICENSE
- Merge pull request #435 from cvxgrp/renovate/marimo-0.x
- Fix include/exclude formatting in template.yml
- Rhiza
- Delete tests/test_config_templates directory
- Delete .github/scripts/sync.sh
- Rhiza
- Add CodeQL analysis workflow configuration
- Migrate
- Template
- Delete .github/CONFIG.md
- Delete .github/TOKEN_SETUP.md
- Merge pull request #445 from cvxgrp/dependabot/github_actions/actions/checkout-6
- Update template.yml
- Removing dependabot
- Rhiza sync
- Initial plan
- Fix package metadata error by adding fallback version handling
- Add test for package metadata fallback handling
- Fmt
- Merge pull request #447 from cvxgrp/copilot/fix-local-machine-bug
- Update template.yml
- Initial plan
- Add tests to increase coverage to 99% (263/265 lines)
- Finalize tests - 99% coverage achieved (261/263 lines)
- Merge pull request #449 from cvxgrp/copilot/increase-test-coverage-100
- Python-version
- Fmt
- Remove marimo runtime configuration
- Merge pull request #450 from cvxgrp/tschm-patch-1
- Update version

## [1.4.3] - 2025-11-23

### Bug Fixes
- *(deps)* Update dependency pre-commit to v4.3.0 (#310)
- *(deps)* Update dependency marimo to v0.14.17 (#313)
- *(deps)* Update dependency cvxbson to v0.2.0 (#322)
- *(deps)* Update dependency mosek to v11.0.28 (#325)
- *(deps)* Update dependency pandas to v2.3.2 (#326)
- *(deps)* Update dependency marimo to v0.15.0 (#327)
- *(deps)* Update dependency marimo to v0.15.2 (#332)
- *(deps)* Update dependency pytest to v8.4.2 (#337)
- *(deps)* Update dependency pytest-cov to v6.3.0 (#338)
- *(deps)* Update dependency marimo to v0.15.5 (#344)
- *(deps)* Update dependency pytest-cov to v7 (#351)
- *(deps)* Update dependency kaleido to v1.1.0 (#349)
- *(deps)* Update dependency marimo to v0.16.1 (#350)
- Fixing (#354)
- *(deps)* Update dependency marimo to v0.16.3 (#355)
- *(deps)* Update dependency mosek to v11.0.29 (#356)
- *(deps)* Update dependency pandas to v2.3.3 (#357)
- *(deps)* Update dependency marimo to v0.16.5 (#361)
- *(deps)* Update dependency marimo to v0.17.0 (#372)
- *(deps)* Update dependency marimo to v0.17.2 (#383)
- *(deps)* Update dependency python-dotenv to v1.2.1 (#384)
- *(deps)* Update dependency marimo to v0.17.6 (#387)
- *(deps)* Update dependency marimo to v0.17.7 (#394)
- *(deps)* Update dependency marimo to v0.17.8
- Fixing plotfrontier in README

### Dependencies
- *(deps)* Lock file maintenance (#305)
- *(deps)* Lock file maintenance (#308)
- *(deps)* Update pre-commit hook pre-commit/pre-commit-hooks to v6 (#311)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.12.8 (#309)
- *(deps)* Update actions/checkout action to v5 (#314)
- *(deps)* Update actions/checkout action to v5 (#315)
- *(deps)* Lock file maintenance (#318)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.12.9 (#320)
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.33.3 (#321)
- *(deps)* Lock file maintenance (#323)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.12.10 (#324)
- *(deps)* Update actions/upload-pages-artifact action to v4 (#328)
- *(deps)* Lock file maintenance (#329)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.12.11 (#331)
- *(deps)* Lock file maintenance (#333)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.12.12 (#335)
- *(deps)* Update softprops/action-gh-release action to v2.3.3 (#336)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.13.0 (#345)
- *(deps)* Lock file maintenance (#346)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.13.1 (#347)
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.34.0 (#348)
- *(deps)* Lock file maintenance (#352)
- *(deps)* Lock file maintenance (#358)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.13.3 (#359)
- *(deps)* Update softprops/action-gh-release action to v2.4.0 (#360)
- *(deps)* Lock file maintenance (#362)
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.34.1 (#364)
- *(deps)* Update pre-commit hook rhysd/actionlint to v1.7.8 (#365)
- *(deps)* Update softprops/action-gh-release action to v2.4.1 (#366)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.14.0 (#367)
- *(deps)* Update astral-sh/setup-uv action to v7 (#368)
- *(deps)* Lock file maintenance (#369)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.14.1 (#370)
- *(deps)* Update mcr.microsoft.com/devcontainers/python docker tag to v3.14 (#371)
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.5 (#373)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.14.2 (#374)
- *(deps)* Update dependency python to 3.14 (#375)
- *(deps)* Lock file maintenance (#376)
- *(deps)* Lock file maintenance (#382)
- *(deps)* Lock file maintenance (#385)
- *(deps)* Update jebel-quant/sync_template action to v0.4.1 (#388)
- *(deps)* Lock file maintenance (#389)
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.8 (#391)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.14.4 (#392)
- *(deps)* Lock file maintenance (#395)
- *(deps)* Update ghcr.io/astral-sh/uv docker tag to v0.9.10 (#397)

### Maintenance
- Sync config files from .config-templates (#312)
- Sync config files from .config-templates (#334)
- Sync config files from .config-templates (#343)
- Sync template files (#363)
- Sync template from tschm/.config-templates@main (#112) (#380)
- Sync template from tschm/.config-templates@main (#112) (#381)
- Sync template files (#113)
- Sync template files (#114)
- Sync template files
- Sync template files (#386)
- Sync template files (#390)
- Sync template files
- Sync template files

### Other Changes
- Renovate updates (#302)
- Update from tschm/cvxcla (#306)
- After sync (#307)
- Update sync.yml
- Renovate updates (#317)
- Delete .env
- Delete .env (#316)
- Tschm patch 1 (#330)
- Update from tschm (#340)
- Delete .github/taskfiles directory
- Delete .github/CONTRIBUTING.md
- Delete .github/CODE_OF_CONDUCT.md
- Update from tschm (#353)
- Update from tschm (#378)
- Merge from tschm (#379)
- Merge pull request #115 from tschm/template-updates
- Merge pull request #396 from cvxgrp/template-updates
- Merge pull request #398 from cvxgrp/renovate/marimo-0.x
- Merge branch 'main' into main
- Merge pull request #399 from tschm/main
- Delete tests/test_docs.py
- Merge pull request #400 from cvxgrp/template-updates
- Fmt
- Revise portfolio example code in README
- Revise output examples in README.md
- Merge pull request #116 from tschm/tschm-patch-1
- Merge pull request #402 from tschm/main

## [1.4.2] - 2025-08-05

### Dependencies
- *(deps)* Lock file maintenance (#299)

### Other Changes
- Updates (#298)
- Update from tschm/cvxcla (#300)
- Renovate updates and config templates (#301)

## [1.4.1] - 2025-07-25

### Other Changes
- Remove spicy (#297)

## [1.4.0] - 2025-07-25

### Other Changes
- Remove cvxpy to compute the first point (#296)

## [1.3.3] - 2025-07-25

### Other Changes
- Pandas (#294)

## [1.3.2] - 2025-07-25

### Maintenance
- Test (#293)

### Other Changes
- Update cla.py
- Remove ruff traces from pyproject.toml (#292)
- Addressing dependencies (#295)

## [1.3.1] - 2025-07-25

### Other Changes
- Without px.line (#291)

## [1.3.0] - 2025-07-24

### Dependencies
- *(deps)* Update tschm/.config-templates action to v0.1.6 (#286)

### Maintenance
- Sync config files from .config-templates (#287)

### Other Changes
- Updated notebook
- Update cla.py
- Updates via .config-templates (#288)
- Updates renovate & .config-templates (#289)

## [1.2.0] - 2025-07-21

### Bug Fixes
- *(deps)* Update dependency pytest-cov to v6.2.1 (#253)
- *(deps)* Update dependency mosek to v11.0.23 (#9)
- *(deps)* Update dependency pytest to v8.4.1 (#10)
- *(deps)* Update dependency mosek to v11.0.24 (#14)
- *(deps)* Update dependency marimo to v0.14.9
- *(deps)* Update dependency kaleido to v1
- *(deps)* Update dependency mosek to v11.0.23
- *(deps)* Update dependency pytest to v8.4.1
- *(deps)* Update dependency marimo to v0.14.7
- *(deps)* Update dependency kaleido to v1
- *(deps)* Update dependency marimo to v0.14.9
- *(deps)* Update dependency mosek to v11.0.24
- *(deps)* Update dependency marimo to v0.14.12 (#284)

### Dependencies
- *(deps)* Lock file maintenance (#252)
- *(deps)* Lock file maintenance (#7)
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.33.1 (#8)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.12.0 (#11)
- *(deps)* Lock file maintenance (#12)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.12.1 (#13)
- *(deps)* Update pre-commit hook crate-ci/typos to v1.34.0
- *(deps)* Lock file maintenance
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.33.1
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.12.0 (#259)
- *(deps)* Lock file maintenance (#262)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.12.1
- *(deps)* Update pre-commit hook crate-ci/typos to v1.34.0
- *(deps)* Lock file maintenance (#270)
- *(deps)* Update jebel-quant/marimushka action to v0.1.4 (#271)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.12.2 (#272)
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.33.2 (#273)
- *(deps)* Update tschm/cradle action to v0.2.1 (#274)
- *(deps)* Lock file maintenance (#275)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.12.3 (#277)
- *(deps)* Update tschm/cradle action to v0.3.01 (#278)
- *(deps)* Lock file maintenance (#279)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.12.4 (#282)
- *(deps)* Update tschm/cradle action to v0.3.04 (#283)
- *(deps)* Lock file maintenance (#285)

### Other Changes
- Coverage for the correct folder
- Makefile revisited
- Update README.md
- Merge branch 'cvxgrp:main' into main
- Merge pull request #15 from tschm/renovate/crate-ci-typos-1.x
- Merge pull request #16 from tschm/renovate/marimo-0.x
- Merge pull request #17 from tschm/renovate/kaleido-1.x
- Merge pull request #254 from tschm/main
- Merge pull request #255 from cvxgrp/renovate/lock-file-maintenance
- Merge pull request #257 from cvxgrp/renovate/mosek-11.x
- Merge pull request #258 from cvxgrp/renovate/pytest-8.x
- Merge pull request #256 from cvxgrp/renovate/python-jsonschema-check-jsonschema-0.x
- Merge pull request #260 from cvxgrp/renovate/marimo-0.x
- Merge pull request #261 from cvxgrp/renovate/kaleido-1.x
- Merge pull request #264 from cvxgrp/renovate/marimo-0.x
- Merge pull request #263 from cvxgrp/renovate/astral-sh-ruff-pre-commit-0.x
- Merge pull request #265 from cvxgrp/renovate/mosek-11.x
- Merge pull request #266 from cvxgrp/renovate/crate-ci-typos-1.x
- Merge branch 'cvxgrp:main' into main
- Update book.yml
- Fmt
- Merge pull request #267 from tschm/main
- Update underlying book (#268)
- Empty (#269)
- Update pre-commit.yml
- Update README.md
- Potential fix for code scanning alert no. 9: Workflow does not contain permissions (#276)
- Update book.yml
- Update via .config-templates (#281)
- Double publish?!

## [1.1.11] - 2025-06-13

### Other Changes
- Update cla.py
- Update cla.py
- Update pre-commit.yml
- Fmt
- Update Makefile
- Update Makefile

## [1.1.10] - 2025-06-13

### Bug Fixes
- *(deps)* Update dependency pytest-cov to v6.2.1 (#4)

### Dependencies
- *(deps)* Lock file maintenance (#5)

### Other Changes
- Update release.yml
- Trying to import cvxcla

## [1.1.9] - 2025-06-13

### Bug Fixes
- Fixing deptry job

### Dependencies
- *(deps)* Lock file maintenance
- *(deps)* Lock file maintenance
- *(deps)* Update tschm/cradle action to v0.1.72

### Other Changes
- D flag
- Refactor Makefile
- Bring in ty
- Bring in ty
- Update pre-commit.yml
- Update pre-commit.yml
- Merge pull request #243 from cvxgrp/tschm-patch-1
- Merge pull request #242 from cvxgrp/renovate/lock-file-maintenance
- Update cla.py
- Create marimo.yml
- Update marimo.yml
- Fmt
- Merge pull request #244 from cvxgrp/tschm-patch-2
- Update book.yml
- Fmt
- Update cla.py
- Update cla.py
- Update cla.py
- Update cla.py
- Cla
- Fmt
- Merge pull request #245 from cvxgrp/tschm-patch-1
- New package
- Corrected tests
- Conftest
- Notebook fixing
- Fmt
- Merge pull request #247 from cvxgrp/246-move-to-cvxcla
- Merge pull request #3 from tschm/renovate/lock-file-maintenance
- Merge pull request #2 from tschm/renovate/tschm-cradle-0.x
- Merge pull request #248 from tschm/main
- Adding dependency preamble
- Adding dependency preamble
- Merge pull request #250 from cvxgrp/249-update-marimo
- Trying to import cvxcla
- Merge pull request #251 from cvxgrp/249-update-marimo

## [1.1.8] - 2025-06-06

### Bug Fixes
- *(deps)* Update dependency pytest to v8.4.0 (#236)
- *(deps)* Update dependency mosek to v11.0.22

### Dependencies
- *(deps)* Lock file maintenance
- *(deps)* Update pre-commit hook crate-ci/typos to v1.33.1 (#235)
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.11.13

### Maintenance
- Testing code in README
- Testing code in README

### Other Changes
- Plot as attribute of frontier
- Merge pull request #233 from cvxgrp/renovate/lock-file-maintenance
- Update book.yml
- Merge pull request #237 from cvxgrp/tschm-patch-1
- Ruff for docstrings
- Update pyproject.toml
- Comments and
- Comments
- Merge pull request #239 from cvxgrp/renovate/mosek-11.x
- Merge pull request #238 from cvxgrp/renovate/astral-sh-ruff-pre-commit-0.x
- Towards a proper book
- Book using marimo export

## [1.1.7] - 2025-05-31

### Bug Fixes
- *(deps)* Update dependency plotly to v6.1.2
- *(deps)* Update dependency mosek to v11.0.21

### Dependencies
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.11.12 (#221)

### Other Changes
- Update README.md
- Install marimo
- Wasm export
- Merge pull request #220 from cvxgrp/219-wasm
- Merge pull request #223 from cvxgrp/renovate/plotly-6.x
- Wasm export
- Merge pull request #224 from cvxgrp/219-wasm
- Merge pull request #222 from cvxgrp/renovate/mosek-11.x
- Wasm export
- Notebook layouts
- Notebook layouts
- Merge pull request #225 from cvxgrp/219-wasm
- Plotly as first class citizen
- Merge branch '219-wasm' into main
- Update cla.py
- Install with micropip
- Install with micropip
- Install with micropip
- Install with micropip
- Hidden cell?
- Hidden cell?
- Merge pull request #230 from cvxgrp/229-micropip-install-in-notebook
- Pandas first class citizen
- Pandas first class citizen
- Plot as attribute of frontier
- Plot as attribute of frontier
- Plot as attribute of frontier
- Plot as attribute of frontier
- Plot as attribute of frontier
- Merge pull request #232 from cvxgrp/231-pandas-and-plot

## [1.1.6] - 2025-05-29

### Other Changes
- Relaxed versions for pyodide
- Version test
- Merge pull request #217 from cvxgrp/216-relax-numpy

## [1.1.5] - 2025-05-27

### Maintenance
- Refactor cla
- Refactor cla

### Other Changes
- Merge pull request #214 from cvxgrp/refactor
- Cached property
- Merge pull request #215 from cvxgrp/refactor

## [1.1.4] - 2025-05-26

### Bug Fixes
- Fixing experiment

### Other Changes
- README
- Plot
- Remove CLAUX
- Remove CLAUX
- Remove CLAUX
- Make plotly an extra
- Deptry issues
- Merge pull request #213 from cvxgrp/experiment
- Update README.md
- Update README.md

## [1.1.3] - 2025-05-26

### Documentation
- Documented tests

### Other Changes
- README corrected
- Workflows commented
- Index pointing to README
- Hooks
- Dependencies
- Cleaner Makefile
- Cleaner Makefile
- Mosek as dev dependency
- Experiments
- Remove sphinx
- Merge pull request #211 from cvxgrp/clean
- README
- Merge pull request #212 from cvxgrp/clean

## [1.1.2] - 2025-05-26

### Other Changes
- Update release.yml

## [1.1.1] - 2025-05-26

### Dependencies
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.11.2
- *(deps)* Update pre-commit hook abravalheri/validate-pyproject to v0.24.1
- *(deps)* Update pre-commit hook crate-ci/typos to v1.31.0
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.32.1
- *(deps)* Update pre-commit hook crate-ci/typos to v1.31.1
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.11.3
- *(deps)* Lock file maintenance
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.11.4
- *(deps)* Update pre-commit hook astral-sh/ruff-pre-commit to v0.11.5
- *(deps)* Lock file maintenance
- *(deps)* Lock file maintenance (#205)
- *(deps)* Update pre-commit hook python-jsonschema/check-jsonschema to v0.33.0 (#206)

### Maintenance
- *(config)* Migrate config .github/renovate.json

### Other Changes
- Bump pytest-cov from 5.0.0 to 6.0.0
- Merge pull request #128 from cvxgrp/dependabot/pip/pytest-cov-6.0.0
- Bump cvxpy from 1.5.3 to 1.6.0
- Merge pull request #129 from cvxgrp/dependabot/pip/cvxpy-1.6.0
- Update ci.yml
- Merge pull request #130 from cvxgrp/tschm-patch-1
- Update pyproject.toml
- Lock
- Merge pull request #131 from cvxgrp/tschm-patch-2
- Update pyproject.toml
- Remove cvxpy
- Merge pull request #132 from cvxgrp/tschm-patch-1
- Update pyproject.toml
- Merge pull request #133 from cvxgrp/tschm-patch-1
- Bump cvxbson from 0.0.7 to 0.0.8
- Merge pull request #134 from cvxgrp/dependabot/pip/cvxbson-0.0.8
- Towards uv
- Fmt
- Cla marimo
- Uv.lock
- Loc poetry
- Fmt
- Workflow uv
- Merge pull request #135 from cvxgrp/uv
- Remove jupyter traces
- Marimo book tested
- Cla page
- Page for book
- Merge pull request #137 from cvxgrp/136-remove-jupyter
- Update README.md
- Merge pull request #138 from cvxgrp/tschm-patch-1
- Fmt
- Startup with uv
- Env for remote task file
- Remove Makefile
- Remove .vscode
- Update pyproject.toml
- Taskfile
- Project url
- Project url
- Contributing
- Actions
- Merge pull request #139 from cvxgrp/actions
- Update pre-commit hooks
- Merge pull request #140 from cvxgrp/actions
- Actions
- Update startup.sh
- Merge pull request #141 from cvxgrp/tschm-patch-1
- Update .pre-commit-config.yaml
- Typos
- Actions
- Merge pull request #142 from cvxgrp/tschm-patch-2
- Fmt
- Update pre-commit.yml
- Merge pull request #144 from cvxgrp/tschm-patch-2
- Explicit checkouts in each workflow
- V2
- V2.0.0
- V2.0.0
- Merge pull request #145 from cvxgrp/v2
- Update book.yml
- Merge pull request #146 from cvxgrp/tschm-patch-2
- [pre-commit.ci] pre-commit autoupdate
- Typos
- Merge pull request #147 from cvxgrp/pre-commit-ci-update-config
- Bump cvxgrp/.github from 2.0.0 to 2.0.3
- Merge pull request #148 from cvxgrp/dependabot/github_actions/cvxgrp/dot-github-2.0.3
- [pre-commit.ci] pre-commit autoupdate
- Merge pull request #149 from cvxgrp/pre-commit-ci-update-config
- Update .pre-commit-config.yaml
- Merge pull request #150 from cvxgrp/tschm-patch-1
- Bump cvxgrp/.github from 2.0.3 to 2.0.6
- Merge pull request #151 from cvxgrp/dependabot/github_actions/cvxgrp/dot-github-2.0.6
- [pre-commit.ci] pre-commit autoupdate
- Merge pull request #152 from cvxgrp/pre-commit-ci-update-config
- Update release.yml
- [pre-commit.ci] auto fixes from pre-commit.com hooks
- Fmt
- Merge pull request #153 from cvxgrp/tschm-patch-1
- Bump cvxgrp/.github from 2.0.6 to 2.0.8
- Merge pull request #154 from cvxgrp/dependabot/github_actions/cvxgrp/dot-github-2.0.8
- [pre-commit.ci] pre-commit autoupdate
- Merge pull request #155 from cvxgrp/pre-commit-ci-update-config
- Automated release
- Automated release
- Merge branch 'tschm-patch-1' into main
- Dispatch
- Revist cvxcla
- Update book production
- Release without explicit checkout
- Update pre-commit hook crate-ci/typos to v1
- Merge pull request #161 from cvxgrp/renovate/major-pre-commit
- Update pre-commit hooks
- Merge pull request #160 from cvxgrp/renovate/pre-commit
- Workflows
- Workflows
- Workflows
- Merge pull request #163 from cvxgrp/workflows
- Lock file maintenance
- Merge pull request #162 from cvxgrp/renovate/lock-file-maintenance
- Release
- Merge pull request #164 from cvxgrp/workflows2
- Book
- Lock file maintenance
- Merge pull request #165 from cvxgrp/renovate/lock-file-maintenance
- Update cvxgrp/.github action to v2.1.1
- Merge pull request #158 from cvxgrp/renovate/cvxgrp-.github-2.x
- Update release.yml
- Merge pull request #166 from cvxgrp/tschm-patch-1
- Update ci.yml
- Fmt
- Merge pull request #167 from cvxgrp/tschm-patch-2
- Update cvxgrp/.github action to v2.2.1
- Update pre-commit.yml
- Update ci.yml
- Update book.yml
- Update ci.yml
- Merge pull request #168 from cvxgrp/renovate/cvxgrp-.github-2.x
- Lock file maintenance
- Merge pull request #169 from cvxgrp/renovate/lock-file-maintenance
- Update ci.yml
- Merge pull request #171 from cvxgrp/tschm-patch-3
- Update cvxgrp/.github action to v2.2.2
- Merge pull request #170 from cvxgrp/renovate/cvxgrp-.github-2.x
- Update cvxgrp/.github action to v2.2.3
- Merge pull request #172 from cvxgrp/renovate/cvxgrp-.github-2.x
- Update pre-commit hook astral-sh/ruff-pre-commit to v0.9.5
- Merge pull request #173 from cvxgrp/renovate/pre-commit
- Lock file maintenance
- Merge pull request #174 from cvxgrp/renovate/lock-file-maintenance
- Update pre-commit hooks
- Merge pull request #175 from cvxgrp/renovate/pre-commit
- Lock file maintenance
- Merge pull request #176 from cvxgrp/renovate/lock-file-maintenance
- Update cvxgrp/.github action to v2.2.4
- Merge pull request #177 from cvxgrp/renovate/cvxgrp-.github-2.x
- Lock file maintenance
- Merge pull request #178 from cvxgrp/renovate/lock-file-maintenance
- Update cvxgrp/.github action to v2.2.5
- Merge pull request #179 from cvxgrp/renovate/cvxgrp-.github-2.x
- Update pre-commit hooks
- Merge pull request #181 from cvxgrp/renovate/pre-commit
- Lock file maintenance
- Merge pull request #180 from cvxgrp/renovate/lock-file-maintenance
- Lock file maintenance
- Merge pull request #182 from cvxgrp/renovate/lock-file-maintenance
- Update cvxgrp/.github action to v2.2.6
- Merge pull request #183 from cvxgrp/renovate/cvxgrp-.github-2.x
- Update pre-commit hooks
- Merge pull request #185 from cvxgrp/renovate/pre-commit
- Lock file maintenance
- Merge pull request #186 from cvxgrp/renovate/lock-file-maintenance
- Update cvxgrp/.github action to v2.2.7
- Merge pull request #184 from cvxgrp/renovate/cvxgrp-.github-2.x
- Lock file maintenance
- Merge pull request #187 from cvxgrp/renovate/lock-file-maintenance
- Update cvxgrp/.github action to v2.2.8
- Merge pull request #188 from cvxgrp/renovate/cvxgrp-.github-2.x
- Lock file maintenance
- Merge pull request #189 from cvxgrp/renovate/lock-file-maintenance
- Lock file maintenance
- Merge pull request #190 from cvxgrp/renovate/lock-file-maintenance
- Lock file maintenance
- Merge pull request #191 from cvxgrp/renovate/lock-file-maintenance
- Update renovate.json
- Merge pull request #192 from tschm/main
- Merge pull request #194 from cvxgrp/renovate/astral-sh-ruff-pre-commit-0.x
- Merge pull request #193 from cvxgrp/renovate/abravalheri-validate-pyproject-0.x
- Update pyproject.toml
- Fmt
- Merge pull request #195 from cvxgrp/tschm-patch-1
- Merge pull request #196 from cvxgrp/renovate/crate-ci-typos-1.x
- Merge pull request #197 from cvxgrp/renovate/python-jsonschema-check-jsonschema-0.x
- Merge pull request #198 from cvxgrp/renovate/crate-ci-typos-1.x
- Merge pull request #199 from cvxgrp/renovate/astral-sh-ruff-pre-commit-0.x
- Merge pull request #201 from cvxgrp/renovate/lock-file-maintenance
- Merge pull request #200 from cvxgrp/renovate/astral-sh-ruff-pre-commit-0.x
- Update renovate.json
- Merge pull request #202 from cvxgrp/renovate/astral-sh-ruff-pre-commit-0.x
- Merge pull request #204 from cvxgrp/renovate/migrate-config
- Merge pull request #203 from cvxgrp/renovate/lock-file-maintenance
- Update README.md
- Makefile
- README with emojis
- Comments
- Merge pull request #207 from cvxgrp/makebranch
- Potential fix for code scanning alert no. 4: Workflow does not contain permissions
- Merge pull request #208 from cvxgrp/alert-autofix-4
- Potential fix for code scanning alert no. 6: Workflow does not contain permissions
- Merge pull request #209 from cvxgrp/alert-autofix-6
- Potential fix for code scanning alert no. 1: Workflow does not contain permissions
- Merge pull request #210 from cvxgrp/alert-autofix-1

## [1.1.0] - 2024-10-27

### Other Changes
- Update basic.yml
- Update __init__.py
- Update claux.py
- Update first.py
- Update cla.py
- Update pyproject.toml
- Lock file with typing extensions
- Cvxpy
- Fmt
- Merge pull request #127 from cvxgrp/deptry

## [1.0.0] - 2024-10-12

### Maintenance
- Testing

### Other Changes
- Update cla.py
- Delete slides directory
- Bump numpy from 1.26.0 to 1.26.1
- Merge pull request #69 from cvxgrp/dependabot/pip/numpy-1.26.1
- Bump cvxpy from 1.4.0 to 1.4.1
- Merge pull request #68 from cvxgrp/dependabot/pip/cvxpy-1.4.1
- Bump pandas from 2.1.1 to 2.1.2
- Merge pull request #72 from cvxgrp/dependabot/pip/pandas-2.1.2
- Bump plotly from 5.17.0 to 5.18.0
- Merge pull request #70 from cvxgrp/dependabot/pip/plotly-5.18.0
- Bump pytest from 7.4.2 to 7.4.3
- Merge pull request #71 from cvxgrp/dependabot/pip/pytest-7.4.3
- Bump pyarrow from 13.0.0 to 14.0.1
- Merge pull request #73 from cvxgrp/dependabot/pip/pyarrow-14.0.1
- Bump numpy from 1.26.1 to 1.26.2
- Merge pull request #75 from cvxgrp/dependabot/pip/numpy-1.26.2
- Bump pandas from 2.1.2 to 2.1.3
- Merge pull request #74 from cvxgrp/dependabot/pip/pandas-2.1.3
- Bump scipy from 1.11.3 to 1.11.4
- Merge pull request #76 from cvxgrp/dependabot/pip/scipy-1.11.4
- Bump pandas from 2.1.3 to 2.1.4
- Merge pull request #77 from cvxgrp/dependabot/pip/pandas-2.1.4
- Bump cvxbson from 0.0.4 to 0.0.6
- Merge pull request #78 from cvxgrp/dependabot/pip/cvxbson-0.0.6
- Bump pytest from 7.4.3 to 7.4.4
- Merge pull request #79 from cvxgrp/dependabot/pip/pytest-7.4.4
- Bump numpy from 1.26.2 to 1.26.3
- Merge pull request #81 from cvxgrp/dependabot/pip/numpy-1.26.3
- Bump cvxbson from 0.0.6 to 0.0.7
- Merge pull request #82 from cvxgrp/dependabot/pip/cvxbson-0.0.7
- Bump jinja2 from 3.1.2 to 3.1.3
- Merge pull request #83 from cvxgrp/dependabot/pip/jinja2-3.1.3
- Bump scipy from 1.11.4 to 1.12.0
- Merge pull request #84 from cvxgrp/dependabot/pip/scipy-1.12.0
- Bump cvxpy from 1.4.1 to 1.4.2
- Merge pull request #86 from cvxgrp/dependabot/pip/cvxpy-1.4.2
- Bump pandas from 2.1.4 to 2.2.0
- Merge pull request #85 from cvxgrp/dependabot/pip/pandas-2.2.0
- Bump pytest from 7.4.4 to 8.0.0
- Merge pull request #87 from cvxgrp/dependabot/pip/pytest-8.0.0
- Bump numpy from 1.26.3 to 1.26.4
- Merge pull request #88 from cvxgrp/dependabot/pip/numpy-1.26.4
- Update dependabot.yml
- Bump actions/checkout from 3 to 4
- Merge pull request #89 from cvxgrp/dependabot/github_actions/actions/checkout-4
- Bump pre-commit/action from 3.0.0 to 3.0.1
- Merge pull request #90 from cvxgrp/dependabot/github_actions/pre-commit/action-3.0.1
- Bump plotly from 5.18.0 to 5.19.0
- Merge pull request #91 from cvxgrp/dependabot/pip/plotly-5.19.0
- Bump pytest from 8.0.0 to 8.0.1
- Merge pull request #92 from cvxgrp/dependabot/pip/pytest-8.0.1
- Bump pytest from 8.0.1 to 8.0.2
- Merge pull request #94 from cvxgrp/dependabot/pip/pytest-8.0.2
- Bump pandas from 2.2.0 to 2.2.1
- Merge pull request #93 from cvxgrp/dependabot/pip/pandas-2.2.1
- Bump pytest from 8.0.2 to 8.1.0
- Merge pull request #95 from cvxgrp/dependabot/pip/pytest-8.1.0
- Bump pytest from 8.1.0 to 8.1.1
- Merge pull request #96 from cvxgrp/dependabot/pip/pytest-8.1.1
- Bump plotly from 5.19.0 to 5.20.0
- Merge pull request #97 from cvxgrp/dependabot/pip/plotly-5.20.0
- Bump pytest-cov from 4.1.0 to 5.0.0
- Merge pull request #98 from cvxgrp/dependabot/pip/pytest-cov-5.0.0
- Update pyproject.toml
- Bump scipy from 1.12.0 to 1.13.0
- Merge pull request #99 from cvxgrp/dependabot/pip/scipy-1.13.0
- Bump pandas from 2.2.1 to 2.2.2
- Merge pull request #100 from cvxgrp/dependabot/pip/pandas-2.2.2
- Bump cvxpy from 1.4.2 to 1.4.3
- Merge pull request #101 from cvxgrp/dependabot/pip/cvxpy-1.4.3
- Bump plotly from 5.20.0 to 5.21.0
- Merge pull request #102 from cvxgrp/dependabot/pip/plotly-5.21.0
- Bump pytest from 8.1.1 to 8.2.0
- Merge pull request #103 from cvxgrp/dependabot/pip/pytest-8.2.0
- Bump jinja2 from 3.1.3 to 3.1.4
- Merge pull request #105 from cvxgrp/dependabot/pip/jinja2-3.1.4
- Bump plotly from 5.21.0 to 5.22.0
- Merge pull request #104 from cvxgrp/dependabot/pip/plotly-5.22.0
- Bump cvxpy from 1.4.3 to 1.5.1
- Merge pull request #106 from cvxgrp/dependabot/pip/cvxpy-1.5.1
- Bump pytest from 8.2.0 to 8.2.1
- Merge pull request #107 from cvxgrp/dependabot/pip/pytest-8.2.1
- Bump pytest from 8.2.1 to 8.2.2
- Merge pull request #109 from cvxgrp/dependabot/pip/pytest-8.2.2
- Update basic.yml
- Update test_frontier.py
- Update test_frontier.py
- Merge pull request #110 from cvxgrp/tschm-patch-1
- Update ci.yml
- Merge pull request #113 from cvxgrp/tschm-patch-2
- Bump cvxpy from 1.5.1 to 1.5.2
- Fmt
- Merge pull request #112 from cvxgrp/dependabot/pip/cvxpy-1.5.2
- Update pyproject.toml
- Clarabel
- Merge pull request #115 from cvxgrp/tschm-patch-2
- Bump numpy from 2.0.0 to 2.0.1
- Merge pull request #118 from cvxgrp/dependabot/pip/numpy-2.0.1
- Bump pytest from 8.2.2 to 8.3.1
- Merge pull request #117 from cvxgrp/dependabot/pip/pytest-8.3.1
- Bump pytest from 8.3.1 to 8.3.2
- Merge pull request #120 from cvxgrp/dependabot/pip/pytest-8.3.2
- Bump plotly from 5.22.0 to 5.23.0
- Merge pull request #119 from cvxgrp/dependabot/pip/plotly-5.23.0
- Bump cvxpy-base from 1.5.2 to 1.5.3
- Merge pull request #121 from cvxgrp/dependabot/pip/cvxpy-base-1.5.3
- Bump plotly from 5.23.0 to 5.24.0
- Merge pull request #122 from cvxgrp/dependabot/pip/plotly-5.24.0
- Bump numpy from 2.0.1 to 2.0.2
- Merge pull request #123 from cvxgrp/dependabot/pip/numpy-2.0.2
- Bump pandas from 2.2.2 to 2.2.3
- Merge pull request #126 from cvxgrp/dependabot/pip/pandas-2.2.3
- Bump plotly from 5.24.0 to 5.24.1
- Merge pull request #124 from cvxgrp/dependabot/pip/plotly-5.24.1
- Bump pytest from 8.3.2 to 8.3.3
- Merge pull request #125 from cvxgrp/dependabot/pip/pytest-8.3.3
- Proper interpolation
- Notebook cleaning

## [0.0.4] - 2023-10-10

### Maintenance
- Testing numpy
- Testing
- Testing
- Testing

### Other Changes
- Removing the Niedermayer trick
- Remove minvariance with cvxpy
- Revisit basic workflow
- Remove Solver
- Moving algebra out
- Merge pull request #55 from cvxgrp/research
- Towards README
- README revisited
- Remove logging
- Notebook
- Update README.md
- Update pyproject.toml
- Poetry updates
- Fmt
- Merge pull request #59 from cvxgrp/poetry
- Update README.md
- Code of conduct
- Conduct
- Bump numpy from 1.25.2 to 1.26.0
- Merge pull request #62 from cvxgrp/dependabot/pip/numpy-1.26.0
- Bump plotly from 5.16.1 to 5.17.0
- Merge pull request #61 from cvxgrp/dependabot/pip/plotly-5.17.0
- Update pyproject.toml
- Update Makefile
- Python 3.11
- Bump pandas from 2.1.0 to 2.1.1
- Merge pull request #63 from cvxgrp/dependabot/pip/pandas-2.1.1
- Update conf.py
- Update conf.py
- Book
- Bump scipy from 1.11.2 to 1.11.3
- Merge pull request #64 from cvxgrp/dependabot/pip/scipy-1.11.3
- Pyportfolioopt
- Merge pull request #66 from cvxgrp/65-bring-cyportfolioopt
- Benchmark test
- Axis labels corrected
- Ignore ruff
- Types and their testing
- Ecos solver for testing
- Remove mosek
- Notebook
- Loguru not first-hand citizen
- Update cvxpy
- Loguru not first-hand citizen

## [0.0.3] - 2023-09-11

### Bug Fixes
- Fixing devcontainer
- Fix pyyaml
- Fix kernel name
- Fix tests
- Fix tolerance for adding turning point
- Fixing minvar example

### Maintenance
- Test coverage
- Test different os
- Test different os
- Tests passing?
- Test solver fixed
- Testing

### Other Changes
- Initial commit
- Update README.md
- 3 bring over code (#4)
- Remove yfinance and quantstats
- Notebooks
- README.md (#6)
- Pre-commit in dev dependencies
- Devcontainer
- Link for codespaces
- Config for ruff
- Install black
- Ignore htmlcov and .pytest_cache
- Makefile
- Update book.yml
- Pre-commit
- Update black
- Linting group
- Dependencies
- Pre commit
- Update frontier.py
- Update README.md
- Deactivate coveralls
- Update .pre-commit-config.yaml (#9)
- Update .pre-commit-config.yaml (#10)
- 12 introduce types (#13)
- Update cla.py (#14)
- 15 introduce schur class (#17)
- 20 add original code (#21)
- Update index.md
- Update LICENSE
- 20 add original code (#23)
- 24 cleaning (#25)
- Getting to 100%
- Getting to 100%
- Coverage
- Update README.md
- Getting to 100%
- Remove init_algo lp
- Remove test for init algo lp
- Remove test for init algo lp
- Readme
- Remove test for init algo lp
- Readme
- Index for book
- Dependency on matplotlib
- 27 support plotting (#28)
- 29 make jupyter (#30)
- Link to Markowitz paper
- Update README.md
- 31 bring test coverage to 100 (#32)
- Update README.md
- Update README.md
- Update README.md
- Fmt
- Remove hist code (#35)
- 33 add markowitz paper (#37)
- 33 add markowitz paper (#38)
- Prepare empty packages
- Adding to CLAux
- 39 integrate solution by philipp schiele (#41)
- Poetry updates
- Update ci.yml
- Update Makefile
- Create dependabot.yml
- Poetry updates
- Without pyyaml
- Update pyyaml
- Update ci.yml (#46)
- Remove book
- Update Makefile
- Update Makefile
- Update Makefile (#47)
- Bump scipy from 1.11.1 to 1.11.2 (#48)
- Elegant download of book artifacts
- Niedermayer test (#50)
- Actions
- Fmt
- Update ci.yml
- Update pyproject.toml
- Update conftest.py
- Rename aux to claux.py
- Rename aux to claux.py
- Revisit workflows
- Jupyter in readme
- Update .pre-commit-config.yaml
- Np.block :-)
- Rename cla in Niedermayer
- Niedermayer CLA
- Lock file
- Bailey revisited
- LICENSE
- Update basic.yml
- Copyright file
- Bump loguru from 0.7.0 to 0.7.1
- Merge pull request #51 from cvxgrp/dependabot/pip/loguru-0.7.1
- Bump pytest from 7.4.0 to 7.4.1
- Lock fmt
- Merge pull request #52 from cvxgrp/dependabot/pip/pytest-7.4.1
- Make fmt update
- Unconstrained
- Minvar portfolio
- Remove TestCLA
- Remove TestCLA
- Cla Markowitz cleaned
- Deactivae poetry pre-commit
- Merge pull request #53 from cvxgrp/markowitz
- Make fmt update
- Explicit ecos for minvar
- Less strict testing
- Bump pytest from 7.4.1 to 7.4.2
- Merge pull request #54 from cvxgrp/dependabot/pip/pytest-7.4.2
- Update README.md

<!-- generated by git-cliff -->
