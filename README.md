# Anonymous framework artifact

This repository contains the framework implementation supplied for anonymous
review. Documentation-site assets, release automation, package-index links,
and project-specific attribution metadata are intentionally omitted.

## Install

Python 3.11–3.13 is supported.

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
```

The import package and command-line entry point are both `anonframework`.

## Verify

```bash
ruff check src/ tests/
ruff format --check src/ tests/
mypy src/
pytest
```

The full test suite covers the controller, event channel, persistence,
reporting, security-domain scopes, and public interfaces.
