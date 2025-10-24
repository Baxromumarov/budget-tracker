#!/bin/sh
set -eu

SERVICE="${SERVICE:-backend}"
PORT="${PORT:-8000}"

case "$SERVICE" in
  backend)
    echo "🚀 Starting backend service..."
    cd backend
    if command -v pip3 >/dev/null 2>&1; then
      PIP_CMD="pip3"
    else
      PIP_CMD="pip"
    fi
    "$PIP_CMD" install --upgrade pip
    "$PIP_CMD" install -r requirements.txt
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
