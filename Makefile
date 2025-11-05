init:
	@command -v uv >/dev/null 2>&1 || { echo "uv is not installed. Installing uv..."; curl -LsSf https://astral.sh/uv/install.sh | sh; }
	uv pip install -e .[dev]
	uv run pre-commit install

lint:
	uv run ruff check .
	uv run black --check .
	uv run mypy src

format:
	uv run ruff check --fix .
	uv run black .

test:
	uv run pytest

build:
	uv build

publish:
	uvx twine upload dist/*

