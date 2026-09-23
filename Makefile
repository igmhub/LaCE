install:
	python scripts/update_version.py
	pip install -e .

test:
	python -m pip install -e ".[test]"
	pytest -q

.PHONY: install test docs clean-docs

docs:
	python -m sphinx -W --keep-going -b html docs docs/_build/html

clean-docs:
	python -m sphinx -M clean docs docs/_build
