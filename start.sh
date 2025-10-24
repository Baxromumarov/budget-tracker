#!/bin/sh
set -eu

SERVICE="${SERVICE:-backend}"
PORT="${PORT:-8000}"

case "$SERVICE" in
  backend)
    echo "🚀 Starting backend service..."
    cd backend
    PYTHON_BIN="$(command -v python3 || command -v python || true)"
    if [ -z "$PYTHON_BIN" ]; then
      echo "❌ Python is not installed in the environment."
      exit 1
    fi
    "$PYTHON_BIN" -m ensurepip --upgrade >/dev/null 2>&1 || true
    "$PYTHON_BIN" -m pip install --upgrade pip
    "$PYTHON_BIN" -m pip install -r requirements.txt
    uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
    ;;

  frontend)
    echo "🌐 Starting frontend service..."
    cd frontend
    npm install
    npm run build
    npx serve -s dist -l "$PORT"
    ;;

  *)
    echo "❌ Unknown SERVICE: $SERVICE. Set SERVICE=backend or SERVICE=frontend."
    exit 1
    ;;
esac
