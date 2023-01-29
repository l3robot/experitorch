.PHONY: format install lock lint pre-commit

format:
	poetry run black --line-length 120 .
	poetry run isort --line-length 120 --project experitorch .

install:
	poetry install

lock:
	poetry lock

lint:
	poetry run mypy experitorch

pre-commit:
	poetry run pre-commit run