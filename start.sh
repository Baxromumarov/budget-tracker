#!/bin/sh
set -eu

SERVICE="${SERVICE:-backend}"
PORT="${PORT:-8080}"

echo "🚀 Starting backend service..."
cd backend
PYTHON_BIN="$(command -v python3 || command -v python || true)"
if [ -z "$PYTHON_BIN" ]; then
  echo "⚙️  Python not detected. Installing python3..."
  apt-get update && apt-get install -y python3 python3-venv python3-pip
  PYTHON_BIN="$(command -v python3 || true)"
  if [ -z "$PYTHON_BIN" ]; then
    echo "❌ Failed to install python3."
    exit 1
  fi
fi

if [ ! -d ".venv" ]; then
  echo "📦 Creating virtual environment..."
  "$PYTHON_BIN" -m venv .venv
fi

VENV_PY=".venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "❌ Virtual environment not created correctly."
  exit 1
fi

"$VENV_PY" -m pip install --upgrade pip
"$VENV_PY" -m pip install -r requirements.txt
exec "$VENV_PY" -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
