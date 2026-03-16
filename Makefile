.PHONY: lint types test check install

lint:
	ruff check src/ tests/

types:
	mypy src/

test:
	pytest

check: lint types test

install:
	uv pip install -e ".[dev]"
