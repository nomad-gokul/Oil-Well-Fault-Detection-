PYTHON ?= python3.11
VENV ?= .venv
BIN := $(VENV)/bin

.PHONY: install install-all lint format test clean serve dashboard publish-bundle up down

$(BIN)/python:
	$(PYTHON) -m venv $(VENV)
	$(BIN)/pip install -q --upgrade pip

install: $(BIN)/python  ## Core + dev dependencies
	$(BIN)/pip install -q -e ".[dev]"

install-all: $(BIN)/python  ## Everything (ml, api, dashboard, notebooks, dev); CPU-only PyTorch
	# Prefer the small CPU-only wheel; fall back to PyPI where that index is unreachable.
	$(BIN)/pip install -q torch --index-url https://download.pytorch.org/whl/cpu || true
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

serve:  ## API on http://localhost:8000/docs (newest bundle in artifacts/serving)
	$(BIN)/uvicorn owfd.api.main:app --reload --port 8000

dashboard:  ## Dashboard on http://localhost:8501
	$(BIN)/streamlit run src/owfd/dashboard/app.py

publish-bundle:  ## Copy the newest artifacts/serving bundle to deploy/bundle (commit it)
	$(BIN)/python scripts/publish_bundle.py

up:  ## Build and start the stack (dashboard: http://localhost:8501)
	docker compose up -d --build --wait

down:
	docker compose down
