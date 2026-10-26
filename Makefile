.PHONY: test figures lint format demo

test:
	uv run pytest -q

figures:
	@for s in scripts/run_phase*.py; do echo "== $$s"; uv run python $$s || exit 1; done

lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

format:
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts

demo:
	uv run python scripts/run_final_demo.py
