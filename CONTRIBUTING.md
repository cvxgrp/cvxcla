# Contributing to cvxcla

This document is a guide to contributing to cvxcla

We welcome all contributions. You don't need to be an expert (in optimization)
to help out.

## Checklist

Contributions are made through
[pull requests](https://help.github.com/articles/using-pull-requests/).
Before sending a pull request, make sure you do the following:

- Run 'make fmt' to make sure your code adheres to our [coding style](#code-style).
  This step also includes our license on top of your new files.
- [Write unit tests](#writing-unit-tests)
- Run the [unit tests](#running-unit-tests) and check that they're passing

## Building cvxcla from source

You'll need to build cvxcla locally in order to start editing code.
Please install [task](https://taskfile.dev)

To install cvxcla from source, clone the Github
repository, navigate to its root, and run the following command:

```bash
task cvxcla:install
```

## Contributing code

To contribute to cvxcla, send us pull requests.
For those new to contributing, check out Github's
[guide](https://help.github.com/articles/using-pull-requests/).

Once you've made your pull request, a member of the cvxcla
development team will assign themselves to review it. You might have a few
back-and-forths with your reviewer before it is accepted, which is completely normal.
Your pull request will trigger continuous integration tests for many different
Python versions and different platforms. If these tests start failing, please
fix your code and send another commit, which will re-trigger the tests.

If you'd like to add a new feature to cvxcla, please do propose your
change on a GitHub issue, to make sure that your priorities align with ours.

If you'd like to contribute code but don't know where to start, try one of the
following:

- Read the cvxcla source and enhance the documentation,
  or address TODOs
- Browse the [issue tracker](https://github.com/cvxgrp/cvxcla/issues),
  and look for the issues tagged "help wanted".

## License

A license is added to new files automatically as a pre-commit hook.

## Code style

We use black and ruff to enforce our Python coding style.
Before sending us a pull request, navigate to the project root
and run

```bash
task cvxcla:fmt
```

to make sure that your changes abide by our style conventions. Please fix any
errors that are reported before sending the pull request.

## Writing unit tests

Most code changes will require new unit tests. Even bug fixes require unit tests,
since the presence of bugs usually indicates insufficient tests.
cvxcla tests live in the directory `tests`,
which contains many files, each of which contains many unit tests.
When adding tests, try to find a file in which your tests should belong;
if you're testing a new feature, you might want to create a new test file.

We use the popular Python [pytest](https://docs.pytest.org/en/) framework for our
tests.

## Running unit tests

We use `pytest` to run our unit tests.
To run all unit tests run the following command:

```bash
task cvxcla:test
```

We keep a close eye on our coverage via

```bash
task cvxcla:coverage
```

Please make sure that your change doesn't cause any of the unit tests to fail.
