.PHONY: install format lint typecheck test run coverage docker-build docker-run

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

docker-build:
	DOCKER_BUILDKIT=1 docker buildx build \
					--secret id=github_token,env=GITHUB_TOKEN \
					--network=host -f Dockerfile.prod -t jurisperu-api .
docker-run:
	docker run -d -p 8000:8000 --env-file .env --name api jurisperu-api

all: install format lint typecheck test
