# Contributing to sidescanpy

Bug reports, sample files and pull requests are welcome.

## Reporting a problem

Open an issue. For a file that doesn't read, or reads wrongly, say which
sonar and acquisition software wrote it, and include the log at DEBUG
level (`sidescanpy.log.configure_logging("DEBUG")`). Format bugs are much
easier to fix with a short sample file: if you can share one under an
open licence, the "File format problem" issue template says how.

## Changing the code

- Setup, commands and conventions: [`docs/development.md`](docs/development.md).
- Architecture rules, the layering the tests enforce, and known issues:
  [`AGENTS.md`](AGENTS.md). It is written for coding agents but is the
  shortest complete guide for people too.
- Add tests for new behaviour, on synthetic data with realistic shapes
  (`tests/unit/synthetic.py`). Never commit survey data.
- Run `make test`, `make lint` and `make docs-fast` before opening a pull
  request. `make docs` fails on any Sphinx warning, including a malformed
  docstring.
- Commit messages follow [Conventional Commits](https://www.conventionalcommits.org/)
  with a scope, e.g. `fix(io/xtf): ...`. One logical change per commit.

By contributing you agree that your contribution is licensed under the
project's MIT licence, and to follow the [code of conduct](CODE_OF_CONDUCT.md).
