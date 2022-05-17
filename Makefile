.PHONY: all lint

all_tests: lint unittest integrationtest

help:
	@echo "Please use 'make <target>' where <target> is one of:"
	@echo "  lint			to run flake8 on all Python files"
	@echo "  unittest		to run unit tests on pySPFM"
	@echo "  integrationtest		to run integration tests"

lint:
	@flake8 pySPFM

unittest:
	@py.test -m "not integration" --cov-append --cov-report xml --cov-report term-missing --cov=pySPFM pySPFM

integrationtest:
	@py.test -m "integration" --cov-append --cov-report xml --cov-report term-missing --cov=pySPFM pySPFM
