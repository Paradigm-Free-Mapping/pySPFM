# template-package
A template repo to use for new repositories.

[![Latest Version](https://img.shields.io/pypi/v/template-package.svg)](https://pypi.python.org/pypi/template-package/)
[![PyPI - Python Version](https://img.shields.io/pypi/pyversions/template-package.svg)](https://pypi.python.org/pypi/template-package/)
[![DOI](https://zenodo.org/badge/111111.svg)](https://zenodo.org/badge/latestdoi/111111)
[![License](https://img.shields.io/badge/License-LGPL%202.1-blue.svg)](https://opensource.org/licenses/LGPL-2.1)
[![CircleCI](https://circleci.com/gh/ME-ICA/template-package.svg?style=shield)](https://circleci.com/gh/ME-ICA/template-package)
[![Documentation Status](https://readthedocs.org/projects/template-package/badge/?version=latest)](http://template-package.readthedocs.io/en/latest/?badge=latest)
[![Codecov](https://codecov.io/gh/me-ica/template-package/branch/main/graph/badge.svg)](https://codecov.io/gh/me-ica/template-package)

## Instructions

1. Replace `template-package` with the new repo name across the whole repository.
1. Enable the GitHub repository on Zenodo.
1. Set up the GitHub repository on CircleCI.
1. Set up the GitHub repository on ReadTheDocs.
1. Make the first release on GitHub.
    - The PyPi deployment Action will fail.
1. Deploy to PyPi (instructions below based on [this page](https://realpython.com/pypi-publish-python-package/#publishing-to-pypi)):
    1. `pip install twine`
    1. `python setup.py sdist bdist_wheel`
    1. Upload to TestPyPi:
        1. `twine upload --repository-url https://test.pypi.org/legacy/ dist/*`
        1. Enter TestPyPi username
        1. Enter TestPyPi password
    1. Upload to PyPi (if TestPyPi worked):
        1. `twine upload dist/*`
        1. Enter PyPi username
        1. Enter PyPi password
    1. Future GitHub releases should now deploy to PyPi via the Action without issue.
1. Update the Zenodo badge now that there's a real release.
    - You must do this _after_ deploying to PyPi because any new commits
      after the first release will change the versioneer-managed version string.
1. Add all important CI steps to the branch protection rules for the `main` branch.
1. Add Integrations for the following:
    - AllContributors
    - Welcome
    - CodeCov
    - circleci-artifacts-redirector
    - Release Drafter? Not sure if the Action can suitably replace the Integration.

## Information about this configuration

### Continuous integration via CircleCI

The default configuration uses CircleCI and make to manage testing.
After tests are run, code coverage information is pushed to CodeCov.
CircleCI will also build the documentation as part of CI, and an artifact redirector
(`circleci-artifacts-redirector`) is necessary to view the rendered documentation from each PR easily.

### Versioning with versioneer

Versioneer is used to automatically track and update version strings.

### Linting with flake8, black, and isort

flake8, black, and isort are used to manage code style.

### Reference management with duecredit

duecredit is used to build reference lists for the codebase.
duecredit is included as a required dependency.

### Documentation with Sphinx and ReadTheDocs

The package documentation is built with Sphinx and we assume that the documentation will be hosted by ReadTheDocs.

### Deployment to PyPi

The package is designed to be pip installable and hosted on PyPi.
New releases are pushed to PyPi automatically via a GitHub Action.
