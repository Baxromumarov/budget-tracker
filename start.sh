#!/bin/sh
set -eu

SERVICE="${SERVICE:-backend}"
PORT="${PORT:-8000}"

echo "🚀 Preparing environment..."
cd backend

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"

ensure_python() {
  local py_bin
  py_bin="$(command -v python3 || command -v python || true)"
  if [ -n "$py_bin" ]; then
    echo "$py_bin"
    return 0
  fi

  echo "⚙️  Python not detected. Attempting installation..."
  if command -v apt-get >/dev/null 2>&1; then
    if ! apt-get update; then
      echo "ℹ️  apt-get update failed."
      return 1
    fi
    if ! apt-get install -y python3 python3-venv python3-pip; then
      echo "ℹ️  apt-get install failed."
      return 1
    fi
  elif command -v apk >/dev/null 2>&1; then
    if ! apk add --no-cache python3 py3-pip; then
      echo "ℹ️  apk add failed."
      return 1
    fi
  elif command -v dnf >/dev/null 2>&1; then
    if ! dnf install -y python3 python3-pip; then
      echo "ℹ️  dnf install failed."
      return 1
    fi
  elif command -v yum >/dev/null 2>&1; then
    if ! yum install -y python3 python3-pip; then
      echo "ℹ️  yum install failed."
      return 1
    fi
  else
    echo "ℹ️  No supported package manager found to install Python automatically."
    return 1
  fi

  py_bin="$(command -v python3 || command -v python || true)"
  if [ -z "$py_bin" ]; then
    echo "❌ Unable to install Python automatically. Please install Python 3 and rerun."
    return 1
  fi
  echo "$py_bin"
  return 0
}

ensure_uv() {
  if command -v uv >/dev/null 2>&1; then
    return 0
  fi

  echo "⚙️  uv not detected. Attempting installation..."
  if command -v curl >/dev/null 2>&1; then
    curl -LsSf https://astral.sh/uv/install.sh | sh
  elif command -v wget >/dev/null 2>&1; then
    wget -qO- https://astral.sh/uv/install.sh | sh
  elif [ -n "$PYTHON_BOOTSTRAP" ]; then
    if ! "$PYTHON_BOOTSTRAP" -m pip --version >/dev/null 2>&1; then
      "$PYTHON_BOOTSTRAP" -m ensurepip --upgrade >/dev/null 2>&1 || true
    fi
    if ! "$PYTHON_BOOTSTRAP" -m pip install --user --upgrade uv; then
      echo "ℹ️  pip install uv failed."
      return 1
    fi
  else
    echo "ℹ️  No curl/wget/python available to bootstrap uv."
    return 1
  fi

  export PATH="$HOME/.local/bin:$HOME/.cargo/bin:$PATH"
  command -v uv >/dev/null 2>&1
}

ensure_tesseract() {
  if command -v tesseract >/dev/null 2>&1; then
    return 0
  fi

  echo "⚙️  Tesseract OCR not detected. Attempting installation..."
  if command -v apt-get >/dev/null 2>&1; then
    if ! apt-get update; then
      echo "ℹ️  apt-get update failed while installing Tesseract."
      return 1
    fi
    if ! apt-get install -y tesseract-ocr libtesseract-dev; then
      echo "ℹ️  apt-get install tesseract-ocr failed."
      return 1
    fi
  elif command -v apk >/dev/null 2>&1; then
    if ! apk add --no-cache tesseract-ocr; then
      echo "ℹ️  apk add tesseract-ocr failed."
      return 1
    fi
  elif command -v brew >/dev/null 2>&1; then
    if ! brew install tesseract; then
      echo "ℹ️  brew install tesseract failed."
      return 1
    fi
  else
    echo "ℹ️  Please install Tesseract OCR manually (https://tesseract-ocr.github.io/) for receipt parsing."
    return 1
  fi
  command -v tesseract >/dev/null 2>&1
}

PYTHON_BOOTSTRAP="$(command -v python3 || command -v python || true)"
if [ -z "$PYTHON_BOOTSTRAP" ]; then
  if PYTHON_BOOTSTRAP="$(ensure_python)"; then
    :
  else
    echo "❌ Python is required to continue."
    exit 1
  fi
fi

if ensure_uv; then
  USE_UV=1
  UV_BIN="$(command -v uv)"
else
  USE_UV=0
  echo "ℹ️  Continuing with Python's venv/pip fallback."
fi

if ! ensure_tesseract; then
  echo "⚠️  Tesseract installation failed; image receipts may not be parsed."
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
• I auto-detect category/type/amount and store it against your Telegram profile.
• Commands: /report for summaries by period, /recent for last entries, /help for tips.
• Drop receipts without text—the bot reads them via local Tesseract OCR.
• Inline buttons offer quick monthly breakdowns (1, 3, 6, 12 months, YTD).
EOF
    exec "$VENV_PY" -m app.telegram_bot
    ;;
  *)
    echo "❌ Unknown SERVICE value: $SERVICE"
    exit 1
    ;;
esac
