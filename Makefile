install:
	pip install -e .

test:
	python -m pip install -e ".[test]"
	pytest -q

NOTEBOOKS_DIR := $(dir $(abspath $(lastword $(MAKEFILE_LIST))))notebooks

.PHONY: install test docs clean-docs notebooks

notebooks:
	find "$(NOTEBOOKS_DIR)" -type f -name '*.py' -not -path '*/.ipynb_checkpoints/*' -print0 | xargs -0 -r jupytext --to ipynb

docs:
	python -m sphinx -W --keep-going -b html docs docs/_build/html

clean-docs:
	python -m sphinx -M clean docs docs/_build
