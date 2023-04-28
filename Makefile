.PHONY: test lint docs release install

test:
	poetry run pytest

lint:
	ruff . --fix

docs:
	mkdocs gh-deploy --force

release:
	poetry version patch
	poetry publish --build

install:
	pip install poetry
	poetry install
