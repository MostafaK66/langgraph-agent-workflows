.PHONY: install install-agents test lint format typecheck quality

install:
	python -m pip install -e '.[dev]'

install-agents:
	python -m pip install -e '.[agents]'

test:
	python -m pytest --cov=langgraph_agents --cov-report=term-missing

lint:
	python -m ruff check src tests

format:
	python -m ruff format src tests

typecheck:
	python -m mypy

quality: lint typecheck test
	python -m compileall -q src tests
