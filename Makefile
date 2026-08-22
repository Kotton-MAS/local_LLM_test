.PHONY: sync test cov lint fmt fmt-check type lock-check ci pre-commit clean help

help:
	@echo "Available targets:"
	@echo "  sync         - Install dependencies (uv sync --locked)"
	@echo "  test         - Run tests (uv run pytest -q)"
	@echo "  cov          - Run tests with coverage"
	@echo "  lint         - Run ruff check"
	@echo "  fmt          - Run ruff format (modifies files)"
	@echo "  fmt-check    - Check formatting without modifying"
	@echo "  type         - Run mypy"
	@echo "  ci           - Full CI check: lint + fmt-check + type + test"
	@echo "  pre-commit   - Run pre-commit on all files"
	@echo "  clean        - Remove caches and build artifacts"

sync:
	uv sync --locked

test:
	uv run pytest -q

cov:
	uv run pytest --cov

lint:
	uv run ruff check .

fmt:
	uv run ruff format .

fmt-check:
	uv run ruff format --check .

type:
	uv run mypy .

lock-check:
	uv lock --check

# Stop hook (verify-ci.sh) と GitHub Actions が両方これを呼ぶ。
# ここを単一の真実とすることで、ローカルと CI の検証ロジック乖離を防ぐ。
ci: lock-check lint fmt-check type test

pre-commit:
	uv run pre-commit run --all-files

clean:
	rm -rf .pytest_cache .ruff_cache .mypy_cache .coverage htmlcov dist build
	find . -type d -name __pycache__ -exec rm -rf {} + 2>/dev/null || true