#!/bin/sh
set -eu

SERVICE="${SERVICE:-backend}"
PORT="${PORT:-8000}"

echo "🚀 Preparing environment..."
cd backend

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi

  echo "⚙️  uv not detected. Attempting installation..."
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  elif command -v python3 >/dev/null 2>&1; then
    python3 -m pip install --user --upgrade uv
  elif command -v python >/dev/null 2>&1; then
    python -m pip install --user --upgrade uv
  else
    echo "ℹ️  No curl/wget/python available to bootstrap uv."
    return 1
  fi

  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  command -v uv >/dev/null 2>&1
}

if ensure_uv; then
  USE_UV=1
  UV_BIN="$(command -v uv)"
else
  USE_UV=0
  echo "ℹ️  Continuing with Python's venv/pip fallback."
fi

if [ "$USE_UV" -eq 1 ]; then
  if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment with uv..."
    "$UV_BIN" venv .venv
  fi
else
  PYTHON_BOOTSTRAP="$(command -v python3 || command -v python || true)"
  if [ -z "$PYTHON_BOOTSTRAP" ]; then
    echo "❌ Python is required to create a virtual environment."
    exit 1
  fi
  if [ ! -d ".venv" ]; then
    echo "📦 Creating virtual environment with $PYTHON_BOOTSTRAP ..."
    "$PYTHON_BOOTSTRAP" -m venv .venv
  fi
fi

VENV_PY=".venv/bin/python"
if [ ! -x "$VENV_PY" ]; then
  echo "❌ Virtual environment not created correctly."
  exit 1
fi

echo "📚 Installing dependencies..."
if [ "${USE_UV:-0}" -eq 1 ]; then
  "$UV_BIN" pip install --python "$VENV_PY" -r requirements.txt
else
  "$VENV_PY" -m pip install --upgrade pip
  "$VENV_PY" -m pip install -r requirements.txt
fi

case "$SERVICE" in
  backend)
    echo "🌐 Launching API on port $PORT..."
    exec "$VENV_PY" -m uvicorn app.main:app --host 0.0.0.0 --port "$PORT"
    ;;
  telegram-bot|bot)
    echo "🤖 Starting Telegram bot..."
    cat <<'EOF'
Bot controls:
• Send text such as "Coffee 5.20" or drop a receipt photo.
• Commands: /report for summaries, /recent for last entries, /help for tips.
• Inline buttons offer quick monthly breakdowns.
EOF
    exec "$VENV_PY" -m app.telegram_bot
    ;;
  *)
    echo "❌ Unknown SERVICE value: $SERVICE"
    exit 1
    ;;
esac
