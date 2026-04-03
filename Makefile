.PHONY: sync
sync:
	uv sync

.PHONY: format
format:
	uv run ruff format
	uv run ruff check --fix

.PHONY: type
type:
	uv run pyright

.PHONY: check
check: format type
