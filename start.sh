#!/usr/bin/env bash
# Render start script — uses PORT env var (default 10000)
PORT="${PORT:-10000}"
echo "Starting Active Scholar on port $PORT"
exec uvicorn main:app --host 0.0.0.0 --port "$PORT"
