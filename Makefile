install:
	python scripts/update_version.py
	pip install -e .

.PHONY: docs clean-docs

docs:
	python -m sphinx -W --keep-going -b html docs docs/_build/html

clean-docs:
	python -m sphinx -M clean docs docs/_build
