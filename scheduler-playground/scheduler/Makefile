test:
	uv run -m tests

test-genetic:
	uv run python -m tests.test_genetic

test-genetic-2:
	uv run python -m tests.test_genetic_2

test-genetic-real:
	uv run python -m tests.test_genetic_real $(ARGS)

export-data:
	uv run python export_real_data.py

dev:
	uv run -m api

clean:
	rm -rf .venv
	find . -type d -name "__pycache__" -exec rm -rf {} +

.PHONY: test test-genetic test-genetic-2 test-genetic-real export-data dev clean