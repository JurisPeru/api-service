.PHONY: install format lint typecheck test run coverage docker-build

install:
	poetry install

format:
	poetry run black --line-length 100 src tests

lint:
	poetry run ruff check src tests

typecheck:
	poetry run mypy src

test:
	poetry run pytest tests --cov=src --cov-report=term-missing --cov-fail-under=70

coverage:
	poetry run pytest tests --cov=src --cov-report=html --cov-fail-under=70

run:
	poetry run uvicorn src.app.main:app --host 0.0.0.0 --reload


all: install format lint typecheck test
