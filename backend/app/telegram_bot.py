"""Telegram bot automation for the budget tracker."""

from __future__ import annotations

import asyncio
import base64
import calendar
import json
import logging
import re
import secrets
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

from openai import OpenAI, OpenAIError
from telegram import InlineKeyboardButton, InlineKeyboardMarkup, Update
from telegram.constants import ParseMode
from telegram.ext import (
    Application,
    ApplicationBuilder,
    CallbackQueryHandler,
    CommandHandler,
    ContextTypes,
    MessageHandler,
    filters,
)

from . import crud
from .config import get_settings
from .db import SessionLocal
from .schemas import TransactionCreate, TransactionType, UserRegister
from .security import hash_password

logger = logging.getLogger(__name__)

settings = get_settings()

if not settings.telegram_bot_token:
    logger.warning("Telegram bot token is not configured. Bot cannot start without TELEGRAM_BOT.")

openai_client = OpenAI(api_key=settings.openai_api_key) if settings.openai_api_key else None

DEFAULT_CATEGORIES = [
    "Food",
    "Grocery",
    "Transport",
    "Housing",
    "Utilities",
    "Insurance",
    "Health",
    "Education",
    "Entertainment",
    "Shopping",
    "Salary",
    "Investment",
    "Travel",
    "Other",
]

REPORT_PRESETS: dict[str, int] = {
    "report:1": 1,   # current month
    "report:3": 3,   # last 3 months
    "report:6": 6,   # last 6 months
    "report:12": 12,  # last 12 months
    "report:ytd": -1,  # year to date
}

CONFIDENCE_THRESHOLD = 0.65


@dataclass(frozen=True)
class BotUser:
    """Lightweight representation of a backend user bound to Telegram."""

    id: int
    name: str
    username: str
    telegram_id: str


def _sanitize_username(raw: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9_]", "", raw.lower())
    return cleaned or "telegram_user"


def _build_unique_username(base: str, telegram_id: int) -> str:
    candidate = _sanitize_username(base) if base else f"tg_{telegram_id}"
    with SessionLocal() as db:
        suffix = 1
        unique_candidate = candidate
        while crud.get_user_by_username(db, unique_candidate):
            suffix += 1
            unique_candidate = f"{candidate}_{suffix}"
        return unique_candidate


def _compose_display_name(first_name: str | None, last_name: str | None, fallback: str) -> str:
    parts = [part for part in (first_name, last_name) if part]
    name = " ".join(parts)
    return name or fallback


def ensure_bot_user(telegram_user: Any) -> tuple[BotUser, bool]:
    """Ensure there is a user/profile for the Telegram sender. Returns BotUser and if created."""
    if telegram_user is None:
        raise ValueError("Received update without telegram user attached.")

    telegram_id = str(telegram_user.id)
    with SessionLocal() as db:
        profile = crud.get_telegram_profile(db, telegram_id)
        now = datetime.now(timezone.utc)

        if profile:
            crud.update_telegram_profile(
                db,
                profile,
                username=telegram_user.username,
                first_name=telegram_user.first_name,
                last_name=telegram_user.last_name,
                language_code=getattr(telegram_user, "language_code", None),
                last_interaction_at=now,
            )
            user = profile.user
            return (
                BotUser(
                    id=user.id,
                    name=user.name,
                    username=user.username,
                    telegram_id=telegram_id,
                ),
                False,
            )

        preferred_username = telegram_user.username or f"tg_{telegram_user.id}"
        username = _build_unique_username(preferred_username, telegram_user.id)
        display_name = _compose_display_name(
            telegram_user.first_name,
            telegram_user.last_name,
            fallback=username,
        )

        random_password = secrets.token_urlsafe(12)
        hashed_password = hash_password(random_password)
        register_data = UserRegister(
            name=display_name,
            username=username,
            email=None,
            password=random_password,
        )
        user = crud.create_user(db, register_data, password_hash=hashed_password)
        crud.create_telegram_profile(
            db,
            user,
            telegram_id=telegram_id,
            username=telegram_user.username,
            first_name=telegram_user.first_name,
            last_name=telegram_user.last_name,
            language_code=getattr(telegram_user, "language_code", None),
        )

        logger.info("Created new Telegram user %s (%s)", username, telegram_id)
        return (
            BotUser(
                id=user.id,
                name=user.name,
                username=user.username,
                telegram_id=telegram_id,
            ),
            True,
        )


def _parse_json_payload(raw: str) -> dict[str, Any]:
    try:
        return json.loads(raw)
    except json.JSONDecodeError as exc:
        logger.debug("Failed to parse AI JSON payload: %s", exc)
        return {}


def _extract_ai_text(response: Any) -> str:
    if not response:
        return ""
    text_chunks: list[str] = []
    try:
        output = getattr(response, "output", None)
        if output:
            for item in output:
                contents = getattr(item, "content", None)
                if not contents:
                    continue
                for content in contents:
                    if content.get("type") == "output_text":
                        text_chunks.append(content.get("text", ""))
        else:
            choices = getattr(response, "choices", None)
            if choices:
                message = choices[0].get("message", {})
                text_chunks.append(message.get("content", ""))
    except Exception as exc:  # noqa: BLE001
        logger.debug("Unable to extract text from OpenAI response: %s", exc)
    return "\n".join(chunk for chunk in text_chunks if chunk).strip()


def _fallback_parse(text: str) -> dict[str, Any]:
    amount_match = re.search(r"(-?\d+[.,]?\d*)", text)
    amount = float(amount_match.group(1).replace(",", ".")) if amount_match else None
    lowered = text.lower()
    type_: TransactionType = "expense"
    if any(keyword in lowered for keyword in ("income", "salary", "paid", "deposit", "received")):
        type_ = "income"
    category = "Salary" if type_ == "income" else "Other"
    if "food" in lowered or "lunch" in lowered or "dinner" in lowered:
        category = "Food"
    elif "grocery" in lowered or "market" in lowered:
        category = "Grocery"
    elif "uber" in lowered or "taxi" in lowered or "bus" in lowered:
        category = "Transport"
    description = text.strip()[:120] if text else "Telegram entry"
    return {
        "amount": amount,
        "currency": None,
        "type": type_,
        "category": category,
        "description": description,
        "date": date.today().isoformat(),
        "confidence": 0.3 if amount is not None else 0.0,
    }


def _call_openai(
    *,
    text: str | None = None,
    image_bytes: bytes | None = None,
) -> dict[str, Any]:
    if not openai_client:
        return _fallback_parse(text or "")

    try:
        instructions = (
            "You are an assistant that extracts structured finance data from receipts and chat messages. "
            "Respond with strict JSON using keys: amount (number), currency (string or null), "
            "type (\"income\" or \"expense\"), category (string), description (string <= 120 chars), "
            "date (ISO-8601, yyyy-mm-dd if available), confidence (0-1). "
            f"Prefer categories from: {', '.join(DEFAULT_CATEGORIES)}."
        )

        user_content: list[dict[str, Any]] = []
        if text:
            user_content.append({"type": "input_text", "text": text})
        if image_bytes:
            encoded = base64.b64encode(image_bytes).decode("utf-8")
            user_content.append({"type": "input_image", "image_base64": encoded})

        response = openai_client.responses.create(
            model=settings.ai_model,
            input=[
                {"role": "system", "content": [{"type": "input_text", "text": instructions}]},
                {"role": "user", "content": user_content},
            ],
            response_format={"type": "json_object"},
        )
        extracted = _extract_ai_text(response)
        data = _parse_json_payload(extracted)
        if not data.get("amount"):
            return _fallback_parse(text or "")
        return data
    except (OpenAIError, json.JSONDecodeError) as exc:
        logger.error("OpenAI parsing failed: %s", exc)
        return _fallback_parse(text or "")


def _coerce_transaction(data: dict[str, Any]) -> tuple[TransactionCreate, dict[str, Any]] | None:
    amount = data.get("amount")
    try:
        amount_value = round(float(amount), 2) if amount is not None else None
    except (TypeError, ValueError):
        amount_value = None

    if amount_value is None or amount_value <= 0:
        return None

    type_raw = str(data.get("type", "expense")).lower()
    type_value: TransactionType = "income" if type_raw == "income" else "expense"

    category = data.get("category") or ("Salary" if type_value == "income" else "Other")
    normalized_category = category.title()
    if normalized_category not in DEFAULT_CATEGORIES:
        normalized_category = normalized_category[:120]

    description = data.get("description") or "Captured from Telegram"
    description = description[:120]

    raw_date = data.get("date")
    tx_date = date.today()
    if isinstance(raw_date, str) and raw_date:
        try:
            tx_date = date.fromisoformat(raw_date[:10])
        except ValueError:
            pass

    extra = {
        "currency": data.get("currency") or settings.default_currency,
        "confidence": float(data.get("confidence") or 0.0),
    }

    tx = TransactionCreate(
        amount=abs(amount_value),
        date=tx_date,
        category=normalized_category,
        type=type_value,
        description=description,
    )
    return tx, extra


def _store_transaction(user_id: int, transaction: TransactionCreate, extra: dict[str, Any]) -> int:
    with SessionLocal() as db:
        description = transaction.description
        if extra.get("currency") and extra["currency"] != settings.default_currency:
            currency_note = f" [{extra['currency']}]"
            description = f"{description}{currency_note}"[:255]
        if extra.get("confidence") < CONFIDENCE_THRESHOLD:
            description = f"{description} (needs review)"[:255]
        tx = crud.create_transaction(
            db,
            user_id=user_id,
            data=TransactionCreate(
                amount=transaction.amount,
                date=transaction.date,
                category=transaction.category,
                type=transaction.type,
                description=description,
            ),
        )
        return tx.id


async def _process_payload(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    text: str | None = None,
    image_bytes: bytes | None = None,
) -> None:
    user, created = await asyncio.to_thread(ensure_bot_user, update.effective_user)
    data = await asyncio.to_thread(_call_openai, text=text, image_bytes=image_bytes)
    parsed = _coerce_transaction(data)

    if not parsed:
        await update.effective_chat.send_message(
            "I couldn't understand that entry. Please send something like "
            "`Spent 25 on groceries yesterday`.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    transaction, extra = parsed
    tx_id = await asyncio.to_thread(_store_transaction, user.id, transaction, extra)

    confidence_note = ""
    if extra["confidence"] < CONFIDENCE_THRESHOLD:
        confidence_note = "\n⚠️ Confidence was low, please double-check in the app."

    await update.effective_chat.send_message(
        (
            f"✅ Saved *{transaction.type.capitalize()}* #{tx_id}\n"
            f"• Amount: {transaction.amount:.2f} {extra['currency']}\n"
            f"• Category: {transaction.category}\n"
            f"• Date: {transaction.date.isoformat()}\n"
            f"• Description: {transaction.description}"
            f"{confidence_note}"
        ),
        parse_mode=ParseMode.MARKDOWN,
    )


def _subtract_months(base: date, months: int) -> date:
    # Adapted manual month subtraction.
    new_month_index = base.year * 12 + base.month - 1 - months
    year = new_month_index // 12
    month = new_month_index % 12 + 1
    day = min(base.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _period_summary(user_id: int, months: int) -> tuple[str, list[tuple[str, float, float]]]:
    today = date.today()
    if months == -1:  # year to date
        start_date = today.replace(month=1, day=1)
    else:
        first_day_current = today.replace(day=1)
        start_date = _subtract_months(first_day_current, months - 1)
    with SessionLocal() as db:
        transactions = crud.list_transactions(db, user_id=user_id, start_date=start_date, end_date=today)

    income = sum(t.amount for t in transactions if t.kind.value == "income")
    expenses = sum(t.amount for t in transactions if t.kind.value == "expense")
    balance = income - expenses

    categories: dict[str, float] = {}
    monthly: dict[str, tuple[float, float]] = {}
    for tx in transactions:
        month_key = tx.date.strftime("%Y-%m")
        income_sum, expense_sum = monthly.get(month_key, (0.0, 0.0))
        if tx.kind.value == "income":
            income_sum += tx.amount
        else:
            expense_sum += tx.amount
        monthly[month_key] = (income_sum, expense_sum)
        categories[tx.category] = categories.get(tx.category, 0.0) + tx.amount

    top_categories = sorted(categories.items(), key=lambda item: item[1], reverse=True)[:3]
    top_categories_text = "\n".join(f"• {name}: {amount:.2f}" for name, amount in top_categories) or "• No data yet"
    monthly_rows = sorted(monthly.items())

    header = (
        f"📊 *Summary* ({start_date:%Y-%m-%d} → {today:%Y-%m-%d})\n"
        f"Income: {income:.2f} {settings.default_currency}\n"
        f"Expenses: {expenses:.2f} {settings.default_currency}\n"
        f"Balance: {balance:.2f} {settings.default_currency}\n\n"
        f"🥇 *Top categories*\n{top_categories_text}"
    )
    return header, [(month, inc, exp) for month, (inc, exp) in monthly_rows]


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user, created = await asyncio.to_thread(ensure_bot_user, update.effective_user)
    welcome = (
        "🎉 Welcome to your Budget Tracker assistant!\n"
        "Send me amounts like `Spent 25 on groceries` or drop a receipt photo."
    )
    if created:
        welcome = (
            "👋 Hello and welcome! I've set up a secure account for you. "
            "Send a message like `Coffee 5.20` or drop a receipt photo to get started."
        )
    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Current month", callback_data="report:1"),
                InlineKeyboardButton("Last 3 months", callback_data="report:3"),
            ],
            [
                InlineKeyboardButton("Last 6 months", callback_data="report:6"),
                InlineKeyboardButton("Year to date", callback_data="report:ytd"),
            ],
            [
                InlineKeyboardButton("Latest entries", callback_data="recent:5"),
            ],
        ]
    )
    await update.message.reply_text(
        welcome + "\n\nCommands:\n"
        "/report – summaries\n"
        "/recent – last 5 entries\n"
        "/help – tips & shortcuts",
        reply_markup=keyboard,
        parse_mode=ParseMode.MARKDOWN,
    )
    context.user_data["bot_user_id"] = user.id


async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    await update.message.reply_text(
        "📌 Tips:\n"
        "• Send text like `Salary 1200 received today` or `Expense 18 lunch`\n"
        "• Drop receipt photos – I'll read them for you\n"
        "• /report to see spending trends\n"
        "• /recent to double-check last entries\n"
        "• Reply with `adjust <amount> category` if you spot an error (coming soon)",
        parse_mode=ParseMode.MARKDOWN,
    )


async def report_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton("Current month", callback_data="report:1"),
                InlineKeyboardButton("Last 3 months", callback_data="report:3"),
            ],
            [
                InlineKeyboardButton("Last 6 months", callback_data="report:6"),
                InlineKeyboardButton("Year to date", callback_data="report:ytd"),
            ],
        ]
    )
    await update.message.reply_text("Choose a range:", reply_markup=keyboard)


async def recent_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user, _ = await asyncio.to_thread(ensure_bot_user, update.effective_user)
    with SessionLocal() as db:
        transactions = crud.list_transactions(db, user.id)[:5]
    if not transactions:
        await update.message.reply_text("No transactions yet. Send one now!")
        return
    lines = [
        f"#{tx.id} {tx.date:%Y-%m-%d} • {tx.kind.value.capitalize()} • {tx.amount:.2f} – {tx.category}"
        for tx in transactions
    ]
    await update.message.reply_text("📝 Last 5 entries:\n" + "\n".join(lines))


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text or ""
    trimmed = text.strip().lower()
    if trimmed in {"report", "summary"}:
        await report_command(update, context)
        return
    await _process_payload(update, context, text=text)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not openai_client:
        await update.message.reply_text(
            "Image analysis requires an OpenAI API key. Please set OPENAI_API_KEY in your environment."
        )
        return
    photo = update.message.photo[-1]
    file = await context.bot.get_file(photo.file_id)
    image_bytes = await file.download_as_bytearray()
    await _process_payload(update, context, image_bytes=image_bytes)


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    user, _ = await asyncio.to_thread(ensure_bot_user, query.from_user)
    data = query.data or ""
    if data.startswith("report:"):
        months = REPORT_PRESETS.get(data)
        if months is None:
            await query.edit_message_text("Unsupported report selection.")
            return
        header, monthly_rows = await asyncio.to_thread(_period_summary, user.id, months)
        monthly_text = "\n".join(
            f"• {month}: income {inc:.2f}, expenses {exp:.2f}"
            for month, inc, exp in monthly_rows
        ) or "No monthly data yet."
        await query.edit_message_text(
            f"{header}\n\n📆 *Monthly breakdown*\n{monthly_text}",
            parse_mode=ParseMode.MARKDOWN,
        )
    elif data.startswith("recent:"):
        count = int(data.split(":")[1])
        with SessionLocal() as db:
            transactions = crud.list_transactions(db, user.id)[:count]
        if not transactions:
            await query.edit_message_text("No transactions recorded yet.")
            return
        lines = [
            f"#{tx.id} {tx.date:%Y-%m-%d} • {tx.kind.value.capitalize()} • {tx.amount:.2f} – {tx.category}"
            for tx in transactions
        ]
        await query.edit_message_text("📝 Latest entries:\n" + "\n".join(lines))


def build_application() -> Application:
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT is missing from configuration.")

    application = ApplicationBuilder().token(settings.telegram_bot_token).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("report", report_command))
    application.add_handler(CommandHandler("recent", recent_command))
    application.add_handler(CallbackQueryHandler(handle_callback))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_text))
    return application


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if not settings.telegram_bot_token:
        raise SystemExit("Please set TELEGRAM_BOT in the environment to run the bot.")
    application = build_application()
    logger.info("Starting Telegram bot...")
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
