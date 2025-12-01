# Copilot Instructions for pySPFM

## Repository Overview

pySPFM is a Python implementation of AFNI's 3dPFM and 3dMEPFM algorithms for Paradigm Free Mapping in fMRI data analysis. It provides sparse deconvolution methods for estimating neuronal activity from BOLD signals, supporting both single-echo and multi-echo fMRI data with optional spatial regularization.

**Type:** Python scientific computing library  
**Python versions:** 3.10, 3.11, 3.12 (CI tests all three; use 3.10 for local development)  
**Key dependencies:** numpy, scipy, nibabel, nilearn, pylops, pyproximal, dask, jax

## Build & Development Commands

### Environment Setup (ALWAYS do this first)

```bash
# Install uv if not available
pip install uv

# Install Python 3.10 (recommended - some dependencies like jaxlib may have delayed wheel availability for newer Python versions)
uv python install 3.10

# Sync dependencies with test extras using Python 3.10
uv sync --python 3.10 --extra tests

# For integration tests, also install setuptools (for pkg_resources)
uv pip install setuptools
```

### Running Tests

```bash
# Run unit tests (skip integration tests - they require network access)
uv run pytest pySPFM/tests/ --skipintegration -v

# Run all tests including integration (requires network for OSF downloads)
uv run pytest pySPFM/tests/ -v

# Run specific test file
uv run pytest pySPFM/tests/test_fista.py -v

# Run with coverage
uv run pytest --cov=pySPFM --cov-report=xml pySPFM/tests/ --skipintegration
```

### Linting

```bash
# Run Black formatter check
uv run black --check pySPFM

# Run flake8 linter
uv run flake8 pySPFM

# Run isort check
uv run isort --check-only pySPFM

# Run all pre-commit hooks (install pre-commit first if needed: pip install pre-commit)
pre-commit run --all-files
```

### Building Documentation

```bash
uv sync --extra doc
uv run sphinx-build -W -b html docs docs/_build/html
```

## Project Structure

```
pySPFM/
├── pySPFM/                    # Main source package
│   ├── __init__.py            # Package initialization
│   ├── __about__.py           # Version and metadata
│   ├── io.py                  # I/O utilities (read/write NIfTI, JSON)
│   ├── utils.py               # Logging, dask scheduler, helpers
│   ├── deconvolution/         # Core deconvolution algorithms
│   │   ├── fista.py           # FISTA solver
│   │   ├── lars.py            # LARS solver
│   │   ├── hrf_generator.py   # HRF matrix generation
│   │   ├── debiasing.py       # Debiasing utilities
│   │   ├── select_lambda.py   # Lambda selection methods
│   │   ├── spatial_regularization.py
│   │   └── stability_selection.py
│   ├── workflows/             # CLI entry points
│   │   ├── pySPFM.py          # Main workflow (pySPFM command)
│   │   ├── auc_to_estimates.py # AUC workflow
│   │   └── parser_utils.py    # Argument parsing utilities
│   └── tests/                 # Test suite
│       ├── conftest.py        # Pytest fixtures, OSF data fetching
│       └── test_*.py          # Test modules
├── docs/                      # Sphinx documentation
├── pyproject.toml             # Project config, dependencies, tool settings
├── tox.ini                    # Tox configuration for multi-Python testing
├── Makefile                   # Make targets for lint/unittest/integrationtest
├── .pre-commit-config.yaml    # Pre-commit hooks (black, isort, trailing whitespace)
└── .circleci/config.yml       # CI pipeline configuration
```

## Configuration Files

| File | Purpose |
|------|---------|
| `pyproject.toml` | Dependencies, build config, tool settings (black, isort, flake8, coverage) |
| `tox.ini` | Multi-Python test environments |
| `.pre-commit-config.yaml` | Pre-commit hooks: trailing whitespace, EOF, YAML, black, isort |
| `.circleci/config.yml` | CI: tests on Python 3.10/3.11/3.12, linting, docs build |
| `.codecov.yml` | Code coverage settings |
| `uv.lock` | Locked dependency versions |

## Code Style Requirements

- **Formatter:** Black (line-length: 99)
- **Import sorting:** isort (profile: black)
- **Linter:** flake8 with plugins (docstrings, unused-arguments, use-fstring)
- **Docstrings:** NumPy style convention
- **Target Python:** 3.10+

## CI/CD Pipeline (CircleCI)

The CI runs:
1. **py_env**: Creates conda environments for Python 3.10, 3.11, 3.12
2. **style_check**: Runs `make lint` (black --check, flake8)
3. **py_unittest**: Runs `make unittest` (pytest --skipintegration)
4. **py_integration**: Runs `make integrationtest` (full test suite)
5. **build_docs**: Builds Sphinx documentation

## Important Notes

1. **Use Python 3.10 for local development** - some dependencies (like jaxlib) may have delayed wheel availability for newer Python versions. CI tests all three versions (3.10, 3.11, 3.12).
2. **Integration tests require network access** to download test data from OSF
3. **Run linting before commits**: `uv run black pySPFM && uv run isort pySPFM`
4. **Version is managed by hatch-vcs** - don't manually edit `_version.py`
5. **Tests download fixtures from OSF** - see `conftest.py` for `fetch_file()` function
6. **Entry points**: `pySPFM` and `auc_to_estimates` CLI commands defined in `pyproject.toml`

## Common Development Tasks

### Adding a new deconvolution method
1. Create module in `pySPFM/deconvolution/`
2. Add tests in `pySPFM/tests/test_<module>.py`
3. Import in `pySPFM/deconvolution/__init__.py` if needed
4. Update workflow in `pySPFM/workflows/pySPFM.py` if adding to CLI

### Adding a new CLI argument
1. Edit `_get_parser()` in the relevant workflow file
2. Add parameter to the main function signature
3. Add tests in corresponding test file

### Running a quick validation
```bash
uv run black --check pySPFM && uv run flake8 pySPFM && uv run pytest pySPFM/tests/ --skipintegration -x
```

These instructions provide a comprehensive overview. Search the codebase for additional context when needed or if discrepancies are found.
