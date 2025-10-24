#!/bin/bash
set -euo pipefail

SERVICE="${SERVICE:-backend}"
PORT="${PORT:-8000}"

case "$SERVICE" in
  backend)
    echo "🚀 Starting backend service..."
    cd backend
    pip install --upgrade pip
    pip install -r requirements.txt
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
