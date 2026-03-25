.PHONY: up down install ingest benchmark api test

# ── Infrastructure ────────────────────────────────────
up:
	docker compose up -d

down:
	docker compose down

# ── Setup ─────────────────────────────────────────────
install:
	poetry install

# ── Data ──────────────────────────────────────────────
ingest:
	poetry run python scripts/ingest_sample_data.py

# ── Evaluation ────────────────────────────────────────
benchmark:
	poetry run python scripts/run_benchmark.py

# ── API ───────────────────────────────────────────────
api:
	poetry run uvicorn whatisrag.api.main:app --host 0.0.0.0 --port 8000 --reload

# ── Test ──────────────────────────────────────────────
test:
	poetry run pytest tests/ -v
