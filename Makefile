SHELL=/bin/bash
POETRY_VERSION=1.1.10

POETRY = pip3 install -q poetry==${POETRY_VERSION}

poetry:
	${POETRY}

develop: # usually use this command
	${POETRY} \
	&& poetry install \

develop_no_venv:
	${POETRY} \
	&& poetry config virtualenvs.create false --local \
	&& poetry install

pip_export:
	pip3 freeze > requirements.txt

poetry_export:
	poetry export -f requirements.txt --output requirements.txt --without-hashes

develop_by_requirements:
	for package in $(cat requirements.txt); do poetry add "${package}"; done