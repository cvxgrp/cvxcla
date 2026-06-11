## [1.7.0] - 2026-06-11

### 🚀 Features

- Covariance backend abstraction (Phase 1 of #646) (#647)

### 💼 Other

- Bump version 1.6.3 → 1.7.0

### ⚙️ Miscellaneous Tasks

- Update CHANGELOG.md for v1.6.3 [skip ci]
- Update rhiza template to v0.18.8 (#643)
## [1.6.3] - 2026-06-04

### 🚀 Features

- Add profiles: github-project to template.yml

### 🐛 Bug Fixes

- Replace broken badge.fury.io PyPI badge with shields.io
- *(deps)* Update dependency kaleido to v1.3.0
- Resolve sync conflicts (accept template version)
- Update workflow references from rhiza v0.14.0 to v0.15.2
- Pin rhiza workflows to fix commit (setup-uv v7.6.0)
- Strip whitespace from PYTHON_VERSION and update test assertion
- Add classifiers and doctest to resolve skipped rhiza tests
- Add lower bound for pytest to fix lowest-direct resolution
- Skip plot tests when plotly is not installed

### 💼 Other

- Bump version 1.6.2 → 1.6.3

### 📚 Documentation

- Add docstrings to nested functions to reach 100% docs coverage

### ⚙️ Miscellaneous Tasks

- Bump rhiza template version v0.9.5 → v0.10.3
- Sync rhiza template v0.9.5 → v0.10.3
- Merge main into rhiza/update-0.10.3, resolve conflicts
- Update template.lock sync timestamp
- Bump rhiza template to v0.11.0
- Sync with rhiza v0.11.0
- Restore workflow files removed by previous sync
- Sync .github workflows from cvxrisk
- Sync .rhiza/tests from cvxrisk
- Bump rhiza to v0.15.1
- Sync with rhiza template v0.15.2
- Bump rhiza to v0.15.3
- Apply rhiza sync v0.15.3
- Bump rhiza to v0.17.0
- Apply rhiza sync v0.17.0
- Bump rhiza to v0.18.4
- Apply rhiza sync v0.18.4
- Add pip dependabot entry for .rhiza/requirements
- Update uv.lock with latest dependency versions
## [1.6.2] - 2026-04-14

### 🐛 Bug Fixes

- Add mkdocs.yml to set correct site name and repo for cvxcla
- Add markdown="1" to README div for proper rendering

### 💼 Other

- Bump version 1.6.1 → 1.6.2

### ⚙️ Miscellaneous Tasks

- Update rhiza template to v0.9.5
- Sync with rhiza template v0.9.5
- Sync rhiza template files
- Move marimo notebooks into book/marimo/notebooks/
## [1.6.1] - 2026-04-03

### 🐛 Bug Fixes

- Update coverage badge to point to gh-pages SVG

### 💼 Other

- Bump version 1.6.0 → 1.6.1

### ⚙️ Miscellaneous Tasks

- Remove devcontainer and fix SECURITY.md for cvxcla
## [1.6.0] - 2026-04-03

### 🐛 Bug Fixes

- Ignore CVE-2026-4539 in pip-audit (no fix available for pygments 2.19.2)
- Add missing uv install step to pre-commit CI job

### 💼 Other

- Bump version 1.5.4 → 1.6.0

### ⚙️ Miscellaneous Tasks

- Sync with rhiza template v0.8.20
- Ignore ocaml/ directory
## [1.5.4] - 2026-03-15

### 💼 Other

- Bump version 1.5.3 → 1.5.4
## [1.5.3] - 2026-03-11

### 💼 Other

- Bump version 1.5.2 → 1.5.3
## [1.5.2] - 2026-02-28

### 🐛 Bug Fixes

- Resolve all pre-commit hook issues
- Resolve mypy type checking errors
- Add explicit float cast to resolve mypy no-any-return error
- Replace np.min/np.max with built-in min/max for type safety
- Add --repo flag to gh release list in release workflow
- Use comment instead of classifier for Private :: Do Not Upload

### 💼 Other

- Bump version 1.5.1 → 1.5.2

### 🚜 Refactor

- Replace assert statements with proper exception raising

### 🧪 Testing

- Update test to expect ValueError instead of AssertionError
- Increase test coverage to 100%

### ⚙️ Miscellaneous Tasks

- Update via rhiza
- Add deptry package_module_name_map configuration
- Update via rhiza
- Replace release.yml symlink with manual PyPI publish workflow
## [1.5.1] - 2026-01-13

### 💼 Other

- Bump version 1.5.0 → 1.5.1

### ⚙️ Miscellaneous Tasks

- Update via rhiza
## [1.5.0] - 2026-01-02

### ⚙️ Miscellaneous Tasks

- Bump version to 1.5.0
## [1.4.4] - 2026-01-02

### 🐛 Bug Fixes

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

### ⚙️ Miscellaneous Tasks

- Sync template files
- Sync template files
- Sync template files
- Sync template files
- Sync template files
- Sync template files
- Import rhiza templates
- Bump version to 1.4.4
## [1.4.3] - 2025-11-23

### 🐛 Bug Fixes

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

### ⚙️ Miscellaneous Tasks

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
## [1.3.0] - 2025-07-24

### ⚙️ Miscellaneous Tasks

- Sync config files from .config-templates (#287)
## [1.2.0] - 2025-07-21

### 🐛 Bug Fixes

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
## [1.1.10] - 2025-06-13

### 🐛 Bug Fixes

- *(deps)* Update dependency pytest-cov to v6.2.1 (#4)
## [1.1.8] - 2025-06-06

### 🐛 Bug Fixes

- *(deps)* Update dependency pytest to v8.4.0 (#236)
- *(deps)* Update dependency mosek to v11.0.22
## [1.1.7] - 2025-05-31

### 🐛 Bug Fixes

- *(deps)* Update dependency plotly to v6.1.2
- *(deps)* Update dependency mosek to v11.0.21
## [1.1.1] - 2025-05-26

### ⚙️ Miscellaneous Tasks

- *(config)* Migrate config .github/renovate.json
## [u0.0.1] - 2023-08-06
