.PHONY: install lock lint pre-commit

install:
	poetry install

lock:
	poetry lock

lint:
	poetry run pylint experitorch

pre-commit:
	poetry run pre-commit run