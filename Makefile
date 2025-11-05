init:
	python -m pip install --upgrade pip
	pip install -e .[dev]
	pre-commit install

lint:
	ruff check .
	black --check .
	mypy src

format:
	ruff check --fix .
	black .

test:
	pytest

build:
	python -m build

publish:
	twine upload dist/*

