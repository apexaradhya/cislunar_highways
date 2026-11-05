.PHONY: test figures lint format demo explorer

# Order matters: later scripts read data written by earlier ones.
FIGURE_SCRIPTS = \
	run_phase2_integrators run_phase3_lagrange run_phase4_hill run_phase5_linear run_phase6_stm \
	run_phase7_lyapunov run_phase8_halo run_phase9_manifolds run_phase10_poincare \
	run_phase11_transfer run_phase11_insertion run_phase11_figures \
	run_phase11b_sweep run_phase11b_figures run_phase11c_predictability run_phase12_visuals

test:
	uv run pytest -q

figures:
	@for s in $(FIGURE_SCRIPTS); do echo "== $$s"; uv run python scripts/$$s.py || exit 1; done

explorer:
	uv run python scripts/export_explorer_data.py
	uv run python scripts/build_explorer.py

lint:
	uv run ruff check src tests scripts
	uv run ruff format --check src tests scripts

format:
	uv run ruff format src tests scripts
	uv run ruff check --fix src tests scripts

demo:
	uv run python scripts/run_final_demo.py
