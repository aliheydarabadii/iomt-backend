#!/bin/sh
set -eu

if [ "${RUN_MIGRATIONS_ON_START:-true}" = "true" ]; then
  alembic upgrade head
fi

if [ "${RUN_SEED_ON_START:-true}" = "true" ]; then
  python -m scripts.seed
fi

exec uvicorn app.main:app --host 0.0.0.0 --port 8000
