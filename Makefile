PYTHON ?= python3.11
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: install install-all lint format test clean

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -q --upgrade pip

install: $(BIN)/python  ## Core + dev dependencies
	$(BIN)/pip install -q -e ".[dev]"

install-all: $(BIN)/python  ## Everything (ml, api, dashboard, notebooks, dev); CPU-only PyTorch
	$(BIN)/pip install -q torch --index-url https://download.pytorch.org/whl/cpu
	$(BIN)/pip install -q -e ".[ml,api,dashboard,notebooks,dev]"

lint:
	$(BIN)/ruff check src tests
	$(BIN)/ruff format --check src tests
	$(BIN)/mypy

format:
	$(BIN)/ruff check --fix src tests
	$(BIN)/ruff format src tests

test:
	$(BIN)/pytest

clean:
	rm -rf .pytest_cache .mypy_cache .ruff_cache build *.egg-info src/*.egg-info
