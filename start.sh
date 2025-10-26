#!/bin/sh
set -eu

SERVICE="${SERVICE:-backend}"
PORT="${PORT:-8000}"

echo "🚀 Preparing environment with uv..."
cd backend

if ! command -v uv >/dev/null 2>&1; then
  echo "⚙️  uv not detected. Installing..."
  curl -LsSf https://astral.sh/uv/install.sh | sh
  export PATH="$HOME/.local/bin:$PATH"
fi

UV_BIN="$(command -v uv || true)"
if [ -z "$UV_BIN" ]; then
  echo "❌ Failed to install uv."
  exit 1
fi

if [ ! -d ".venv" ]; then
  echo "📦 Creating virtual environment with uv..."
  "$UV_BIN" venv .venv
fi

VENV_PY=".venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "❌ Virtual environment not created correctly."
  exit 1
fi

echo "📚 Installing dependencies..."
"$UV_BIN" pip install --python "$VENV_PY" -r requirements.txt

case "$SERVICE" in
  backend)
    echo "🌐 Launching API on port $PORT..."
    exec "$VENV_PY" -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
    ;;
  telegram-bot|bot)
    echo "🤖 Starting Telegram bot..."
    exec "$VENV_PY" -m app.telegram_bot
    ;;
  *)
    echo "❌ Unknown SERVICE value: $SERVICE"
    exit 1
    ;;
esac
