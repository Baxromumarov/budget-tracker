"""Telegram bot automation for the budget tracker."""

from __future__ import annotations

import asyncio
import calendar
import json
import logging
import re
import secrets
from io import BytesIO
from dataclasses import dataclass
from datetime import date, datetime, timezone
from typing import Any

try:
    from PIL import Image, ImageOps, UnidentifiedImageError
    import pytesseract
    TESSERACT_AVAILABLE = True
except ImportError:
    Image = None
    ImageOps = None
    UnidentifiedImageError = Exception
    pytesseract = None
    TESSERACT_AVAILABLE = False
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
    logger.warning(
        "Telegram bot token is not configured. Bot cannot start without TELEGRAM_BOT.")


_openai_client: OpenAI | None = None
_openai_client_disabled = False


def get_openai_client() -> OpenAI | None:
    global _openai_client, _openai_client_disabled
    if _openai_client_disabled or not settings.openai_api_key:
        return None
    if _openai_client is not None:
        return _openai_client
    try:
        _openai_client = OpenAI(api_key=settings.openai_api_key)
        return _openai_client
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to initialise OpenAI client: %s", exc)
        _openai_client_disabled = True
        return None


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

AMOUNT_KEYWORDS = [
    "total",
    "amount",
    "grand",
    "balance",
    "due",
    "payable",
    "paid",
    "subtotal",
]

CATEGORY_KEYWORDS = {
    "restaurant": "Food",
    "tandoori": "Food",
    "chicken": "Food",
    "salad": "Food",
    "bowl": "Food",
    "health": "Food",
    "food": "Food",
    "grocer": "Grocery",
    "market": "Grocery",
    "supermarket": "Grocery",
    "rent": "Housing",
    "lease": "Housing",
    "utility": "Utilities",
    "electricity": "Utilities",
    "water": "Utilities",
    "gst": "Other",
    "internet": "Utilities",
    "phone": "Utilities",
    "uber": "Transport",
    "taxi": "Transport",
    "bus": "Transport",
    "train": "Transport",
    "flight": "Travel",
    "hotel": "Travel",
    "medicine": "Health",
    "pharmacy": "Health",
    "hospital": "Health",
    "salary": "Salary",
    "bonus": "Salary",
    "wage": "Salary",
    "shop": "Shopping",
    "store": "Shopping",
}

DATE_PATTERNS = [
    r"\b(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})\b",
    r"\b(\d{4}[/-]\d{1,2}[/-]\d{1,2})\b",
    r"\b(\d{1,2}\s+[A-Za-z]{3,9}\s+\d{2,4})\b",
]

CURRENCY_SYMBOL_MAP = {
    "$": "USD",
    "€": "EUR",
    "£": "GBP",
    "₹": "INR",
    "¥": "JPY",
    "₩": "KRW",
    "₽": "RUB",
    "₺": "TRY",
    "₫": "VND",
    "₦": "NGN",
    "R$": "BRL",
    "A$": "AUD",
    "C$": "CAD",
}
CURRENCY_CODE_REGEX = re.compile(
    r"\b(USD|EUR|GBP|INR|JPY|AUD|CAD|CHF|NZD|SEK|NOK|DKK|SGD|HKD|RUB|TRY|BRL|ZAR|UZS)\b",
    re.IGNORECASE,
)

AMOUNT_KEYWORD_WEIGHTS = {
    "invoice amount": 800,
    "total amount": 750,
    "grand total": 750,
    "invoice": 550,
    "total": 500,
    "amount": 450,
    "balance": 300,
    "payable": 300,
    "due": 250,
    "sgst": 150,
    "cgst": 150,
}

REPORT_PRESETS: dict[str, int] = {
    "report:1": 1,   # current month
    "report:3": 3,   # last 3 months
    "report:6": 6,   # last 6 months
    "report:12": 12,  # last 12 months
    "report:ytd": -1,  # year to date
}

CONFIDENCE_THRESHOLD = 0.55
PENDING_TX_KEY = "pending_transaction"
LAST_SAVED_TX_KEY = "last_saved_transaction"


def _main_menu_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "📊 View Reports", callback_data="menu:reports"
                ),
                InlineKeyboardButton(
                    "📝 My Transactions", callback_data="menu:transactions"
                ),
            ],
            [
                InlineKeyboardButton(
                    "🗑 Clear History", callback_data="clear:prompt"
                ),
            ],
            [
                InlineKeyboardButton("📥 Latest 5", callback_data="recent:5"),
                InlineKeyboardButton("💡 Help", callback_data="menu:help"),
            ],
        ]
    )


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
        user = crud.create_user(
            db, register_data, password_hash=hashed_password)
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


def _fallback_parse(text: str) -> dict[str, Any]:
    amount_match = re.search(r"(-?\d+[.,]?\d*)", text)
    amount = float(amount_match.group(1).replace(
        ",", ".")) if amount_match else None
    lowered = text.lower()
    type_: TransactionType = "expense"
    if any(keyword in lowered for keyword in ("income", "salary", "paid", "deposit", "received")):
        type_ = "income"
    category = "Salary" if type_ == "income" else "Other"
    if "food" in lowered or "lunch" in lowered or "dinner" in lowered or "cafe" in lowered or "restaurant" in lowered:
        category = "Food"
    elif "grocery" in lowered or "groceries" in lowered or "market" in lowered or "supermarket" in lowered:
        category = "Grocery"
    elif "uber" in lowered or "taxi" in lowered or "bus" in lowered or "train" in lowered or "metro" in lowered or "gas" in lowered or "fuel" in lowered:
        category = "Transport"
    elif "rent" in lowered or "house" in lowered or "apartment" in lowered:
        category = "Housing"
    elif "utility" in lowered or "electricity" in lowered or "water" in lowered or "internet" in lowered or "phone" in lowered:
        category = "Utilities"
    elif "doctor" in lowered or "hospital" in lowered or "pharmacy" in lowered or "medicine" in lowered or "health" in lowered:
        category = "Health"
    elif "cinema" in lowered or "movie" in lowered or "concert" in lowered or "game" in lowered or "entertainment" in lowered:
        category = "Entertainment"
    elif "shopping" in lowered or "store" in lowered or "shop" in lowered or "purchase" in lowered:
        category = "Shopping"
    currency = _detect_currency(text) or settings.default_currency
    description = "" if text else "Telegram entry"
    # Determine confidence based on how well we matched
    confidence = 0.85 if amount is not None and category != "Other" else 0.5
    return {
        "amount": amount,
        "currency": currency,
        "type": type_,
        "category": category,
        "description": description,
        "date": date.today().isoformat(),
        "confidence": confidence,
    }


def _call_openai_structured(text: str) -> dict[str, Any] | None:
    client = get_openai_client()
    if not client:
        return None

    instructions = (
        "You extract structured expense data from receipts or chat messages. "
        "Return ONLY valid JSON matching this schema: {\"amount\": number, \"currency\": string|null, \"type\": \"expense\"|\"income\", \"category\": string|null, \"description\": string|null, \"date\": string|null}. "
        "Interpret shorthand like 2k=2000 or 1.5m=1500000, and detect currency symbols/codes returning ISO codes (USD, EUR, INR, UZS, etc.). "
        "Use ISO dates (YYYY-MM-DD), default type to \"expense\" unless clearly income, and keep description <=120 characters."
    )

    try:
        response = client.chat.completions.create(
            model=settings.ai_model,
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": text},
            ],
            temperature=0,
        )
        content = response.choices[0].message.content or ""
        start = content.find("{")
        end = content.rfind("}")
        if start == -1 or end == -1:
            raise ValueError("No JSON object in OpenAI response")
        payload = content[start : end + 1]
        data = json.loads(payload)
        amount = data.get("amount")
        if amount is not None:
            try:
                data["amount"] = round(float(amount), 2)
            except (TypeError, ValueError):
                data["amount"] = None
        if data.get("date"):
            try:
                data["date"] = date.fromisoformat(str(data["date"])[:10]).isoformat()
            except ValueError:
                data["date"] = None
        data["description"] = ""
        if not data.get("type"):
            data["type"] = "expense"
        if data.get("currency"):
            data["currency"] = str(data["currency"]).upper()
        else:
            data["currency"] = _detect_currency(text) or settings.default_currency
        data["confidence"] = 0.9
        logger.info("OpenAI refined data: %s", data)
        return data
    except (OpenAIError, json.JSONDecodeError, ValueError, KeyError) as exc:
        logger.error("OpenAI parsing failed: %s", exc)
        return None


def _normalise_amount(raw: str) -> float | None:
    candidate = raw.replace(",", "")
    candidate_no_space = candidate.replace(" ", "")
    suffix_match = re.fullmatch(r"(\d+(?:\.\d+)?)([kmb])", candidate_no_space.lower())
    if suffix_match:
        value = float(suffix_match.group(1))
        multiplier = {"k": 1_000, "m": 1_000_000, "b": 1_000_000_000}[suffix_match.group(2)]
        return round(value * multiplier, 2)
    try:
        return round(float(candidate_no_space), 2)
    except ValueError:
        return None


def _detect_currency(text: str) -> str | None:
    lowered = text.lower()
    for symbol, code in CURRENCY_SYMBOL_MAP.items():
        if symbol in text:
            return code
    match = CURRENCY_CODE_REGEX.search(text)
    if match:
        return match.group(1).upper()
    if "uzs" in lowered or "so'm" in lowered or "sum" in lowered:
        return "UZS"
    return None


def _extract_amount_from_text(text: str) -> tuple[float | None, float]:
    pattern = re.compile(
        r"(\d{1,3}(?:[,\s]\d{3})+(?:\.\d{1,2})|\d+(?:\.\d{1,2})?)(?:\s*([kmb]))?",
        re.IGNORECASE,
    )
    best_value: float | None = None
    best_score = float("-inf")

    lines = text.splitlines()
    length = len(lines)
    for idx, raw_line in enumerate(lines):
        line = raw_line.strip()
        if not line:
            continue
        lower = line.lower()
        keyword_bonus = 0
        for keyword, weight in AMOUNT_KEYWORD_WEIGHTS.items():
            if keyword in lower:
                keyword_bonus = max(keyword_bonus, weight)
        for match in pattern.finditer(line):
            base_value = match.group(1)
            suffix = match.group(2) or ""
            token = f"{base_value}{suffix}"
            value = _normalise_amount(token)
            if value is None:
                continue
            digits_only = re.sub(r"\D", "", base_value)
            if len(digits_only) >= 8 and "." not in token:
                continue
            if value > 1_000_000:
                continue
            tokens = set(re.findall(r"[A-Za-z]+", lower))
            if {"approval", "auth", "authorization"} & tokens:
                continue
            if 1800 <= value <= 2100 and {"date", "terminal", "time", "am", "pm"} & tokens:
                continue
            if idx > 0:
                prev_tokens = set(re.findall(r"[A-Za-z]+", lines[idx - 1].lower()))
                if {"approval", "auth", "authorization"} & prev_tokens:
                    continue
            if "thank" in lower and keyword_bonus == 0:
                continue
            score = value
            if value >= 10_000 and keyword_bonus == 0:
                score *= 0.6
            score += keyword_bonus
            score += (length - idx) * 2
            if score > best_score:
                best_score = score
                best_value = value
    if best_value is None:
        return None, 0.0
    confidence = 0.9 if keyword_bonus else 0.65
    return best_value, confidence


def _extract_date_from_text(text: str) -> str | None:
    for pattern in DATE_PATTERNS:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if not match:
            continue
        raw = match.group(1)
        cleaned = raw.replace("/", "-").replace(".", "-")
        for fmt in ("%d-%m-%Y", "%d-%m-%y", "%Y-%m-%d", "%d-%b-%Y", "%d-%B-%Y", "%d %b %Y", "%d %B %Y"):
            try:
                parsed = datetime.strptime(cleaned, fmt)
                return parsed.date().isoformat()
            except ValueError:
                continue
    return None


def _infer_category_from_text(text: str) -> str:
    lowered = text.lower()
    for keyword, category in CATEGORY_KEYWORDS.items():
        if keyword in lowered:
            return category
    return "Other"


def _summarise_description(text: str, total: float | None) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]

    merchant = ""
    for line in lines[:5]:
        words = re.findall(r"[A-Za-z]{3,}", line)
        if len(words) >= 2 and not re.search(r"\d{2,}", line):
            merchant = " ".join(words)[:40]
            if not merchant.lower().startswith("receipt"):
                break
            merchant = ""
    if not merchant:
        for line in lines:
            if "receipt" in line.lower():
                candidate = line.title()[:40]
                if candidate.lower().startswith("receipt") and len(lines) > 1:
                    continue
                merchant = candidate
                break
    if not merchant:
        for line in lines[:3]:
            letters = re.findall(r"[A-Za-z]{3,}", line)
            if letters:
                candidate = " ".join(letters)[:40]
                if candidate:
                    merchant = candidate
                    break

    item_lines: list[str] = []
    for line in lines:
        if re.search(r"\b\d+\s*x\b", line.lower()) and re.search(r"\d+\.\d{2}", line):
            item_lines.append(re.sub(r"\s+", " ", line))
    if not item_lines:
        for line in lines:
            if re.search(r"\d+\.\d{2}", line):
                item_lines.append(re.sub(r"\s+", " ", line))
                if len(item_lines) >= 3:
                    break

    parts: list[str] = []
    if merchant:
        parts.append(merchant)
    if total is not None:
        parts.append(f"Total {total:.2f}")
    if item_lines:
        parts.append("; ".join(item_lines[:2]))

    summary = " | ".join(parts) if parts else ""
    return summary[:120]


def _heuristic_parse(text: str) -> dict[str, Any]:
    amount, confidence = _extract_amount_from_text(text)
    if amount is None:
        return {}

    category = _infer_category_from_text(text)
    currency = _detect_currency(text) or settings.default_currency
    type_: TransactionType = "expense"
    lowered = text.lower()
    if any(term in lowered for term in ("salary", "income", "credited", "payment received")):
        type_ = "income"
        if category == "Other":
            category = "Salary"

    potential_date = _extract_date_from_text(text) or date.today().isoformat()

    confidence = max(confidence, 0.85 if category != "Other" else 0.6)

    return {
        "amount": amount,
        "currency": currency,
        "type": type_,
        "category": category,
        "description": "",
        "date": potential_date,
        "confidence": confidence,
    }


def _merge_results(ai_data: dict[str, Any] | None, heuristic_data: dict[str, Any] | None) -> dict[str, Any] | None:
    if ai_data and ai_data.get("amount"):
        if not heuristic_data or not heuristic_data.get("amount"):
            return ai_data
        heuristic_conf = float(heuristic_data.get("confidence", 0.0))
        ai_amount = float(ai_data.get("amount", 0.0) or 0.0)
        heuristic_amount = float(heuristic_data.get("amount", 0.0) or 0.0)
        significant_gap = abs(ai_amount - heuristic_amount) > max(5.0, 0.1 * max(ai_amount, 1.0))
        if heuristic_conf < 0.75 or significant_gap:
            return ai_data
        merged = heuristic_data.copy()
        for key in ("amount", "category", "type", "date", "currency"):
            if (not merged.get(key) or merged.get(key) == "Other") and ai_data.get(key):
                merged[key] = ai_data[key]
        merged["description"] = ""
        merged["confidence"] = max(heuristic_conf, float(ai_data.get("confidence", 0.0)))
        return merged
    return heuristic_data


def _extract_text_from_image(image_bytes: bytes) -> str:
    if not TESSERACT_AVAILABLE or not Image or not pytesseract:
        logger.warning("Tesseract OCR is not available in this environment.")
        return ""
    try:
        image = Image.open(BytesIO(image_bytes))
    except (UnidentifiedImageError, OSError) as exc:
        logger.warning("Failed to open image for OCR: %s", exc)
        return ""

    # Basic preprocessing to improve OCR results
    if max(image.width, image.height) < 1600:
        scale = 1600 / max(image.width, image.height)
        image = image.resize(
            (int(image.width * scale), int(image.height * scale)),
            Image.Resampling.LANCZOS,
        )

    image = image.convert("L")  # grayscale
    if ImageOps:
        image = ImageOps.autocontrast(image)
    image = image.point(lambda x: 0 if x < 140 else 255, "1")

    config = "--psm 6 --oem 3"
    text = pytesseract.image_to_string(image, lang="eng", config=config)
    normalised = text.replace("\r", "\n")
    normalised = re.sub(r"[ \t]+", " ", normalised)
    lines = [line.strip() for line in normalised.split("\n") if line.strip()]
    cleaned = "\n".join(lines)
    preview = cleaned if len(cleaned) <= 500 else f"{cleaned[:500]}…"
    logger.info("OCR extracted text: %s", preview)
    return cleaned


def _analyse_text_blob(text: str) -> dict[str, Any]:
    cleaned = text.strip()
    if not cleaned:
        return {}

    ai_data = None
    if settings.openai_api_key:
        ai_data = _call_openai_structured(cleaned)
        if ai_data and ai_data.get("amount"):
            logger.info("Parsed receipt data (via OpenAI): %s", ai_data)
            ai_data["description"] = ""
            if not ai_data.get("currency"):
                ai_data["currency"] = _detect_currency(cleaned) or settings.default_currency
            return ai_data

    heuristic_data = _heuristic_parse(cleaned)
    if heuristic_data:
        logger.info("Parsed receipt data (heuristic): %s", heuristic_data)
        heuristic_data["description"] = ""
        return heuristic_data

    fallback = _fallback_parse(cleaned)
    fallback["description"] = ""
    logger.info("Parsed receipt data (fallback): %s", fallback)
    return fallback


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

    category = data.get("category") or (
        "Salary" if type_value == "income" else "Other")
    normalized_category = category.title()
    if normalized_category not in DEFAULT_CATEGORIES:
        normalized_category = normalized_category[:120]

    description = ""

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


def _store_transaction(user_id: int, transaction: TransactionCreate, extra: dict[str, Any]) -> tuple[int, int]:
    with SessionLocal() as db:
        description = transaction.description or ""
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
                description=description or "",
            ),
        )
        total = crud.count_transactions(db, user_id)
        return tx.id, total


def _clear_user_transactions(user_id: int) -> int:
    with SessionLocal() as db:
        return crud.delete_transactions_for_user(db, user_id)


def _delete_transaction(user_id: int, transaction_id: int) -> bool:
    with SessionLocal() as db:
        return crud.delete_transaction_by_id(db, transaction_id, user_id)


def _post_save_keyboard() -> InlineKeyboardMarkup:
    return InlineKeyboardMarkup(
        [
            [InlineKeyboardButton("➕ Add Another", callback_data="action:add")],
            [InlineKeyboardButton("📊 This Month", callback_data="action:report_current")],
            [InlineKeyboardButton("↩️ Undo", callback_data="action:undo")],
        ]
    )

async def _process_payload(
    update: Update,
    context: ContextTypes.DEFAULT_TYPE,
    *,
    text: str | None = None,
    image_bytes: bytes | None = None,
    require_confirmation: bool = False,
) -> None:
    user, created = await asyncio.to_thread(ensure_bot_user, update.effective_user)

    collected_parts: list[str] = []
    if text and text.strip():
        collected_parts.append(text.strip())

    if image_bytes:
        if not TESSERACT_AVAILABLE:
            await update.effective_chat.send_message(
                "Image OCR is not available on this server. Please install Tesseract OCR or send the details as text.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return
        ocr_text = await asyncio.to_thread(_extract_text_from_image, bytes(image_bytes))
        if not ocr_text:
            await update.effective_chat.send_message(
                "I couldn't read the receipt clearly. Try taking a sharper photo or reply with the amount and category in text.",
                parse_mode=ParseMode.MARKDOWN,
            )
            return
        collected_parts.append(ocr_text)

    combined_text = "\n".join(collected_parts).strip()
    if not combined_text:
        await update.effective_chat.send_message(
            "I need a short description or a readable receipt to log the transaction. Please try again.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    data = await asyncio.to_thread(_analyse_text_blob, combined_text)
    logger.info("Combined text for analysis: %s", combined_text)
    logger.info("Analysis result: %s", data)
    parsed = _coerce_transaction(data)

    if not parsed:
        await update.effective_chat.send_message(
            "I couldn't understand that entry. Please send something like `Spent 25 on groceries yesterday`.",
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    transaction, extra = parsed
    if not extra.get("currency"):
        extra["currency"] = settings.default_currency
    if require_confirmation:
        context.user_data[PENDING_TX_KEY] = {
            "user_id": user.id,
            "transaction": transaction.model_dump(),
            "extra": extra,
        }
        summary = (
            f"🧾 *Receipt draft*\n"
            f"• Type: {transaction.type.capitalize()}\n"
            f"• Amount: {transaction.amount:.2f} {extra['currency']}\n"
            f"• Category: {transaction.category}\n"
            f"• Date: {transaction.date.isoformat()}\n"
            f"• Description: {transaction.description}\n\n"
            "Is this correct?"
        )
        keyboard = InlineKeyboardMarkup(
            [
                [
                    InlineKeyboardButton("✅ Looks good", callback_data="confirm:yes"),
                    InlineKeyboardButton("✏️ Needs edits", callback_data="confirm:no"),
                ]
            ]
        )
        await update.effective_chat.send_message(
            summary,
            reply_markup=keyboard,
            parse_mode=ParseMode.MARKDOWN,
        )
        return

    tx_id, ordinal = await asyncio.to_thread(_store_transaction, user.id, transaction, extra)
    context.user_data[LAST_SAVED_TX_KEY] = {"transaction_id": tx_id, "user_id": user.id, "ordinal": ordinal}

    confidence_note = ""
    if extra["confidence"] < CONFIDENCE_THRESHOLD:
        confidence_note = "\n⚠️ Confidence was low, please double-check in the app."

    await update.effective_chat.send_message(
        (
            f"✅ Saved *{transaction.type.capitalize()}* #{ordinal}\n"
            f"• Amount: {transaction.amount:.2f} {extra['currency']}\n"
            f"• Category: {transaction.category}\n"
            f"• Date: {transaction.date.isoformat()}"
            f"{confidence_note}\n\n"
            "Choose a next step below."
        ),
        parse_mode=ParseMode.MARKDOWN,
        reply_markup=_post_save_keyboard(),
    )


def _subtract_months(base: date, months: int) -> date:
    # Adapted manual month subtraction.
    new_month_index = base.year * 12 + base.month - 1 - months
    year = new_month_index // 12
    month = new_month_index % 12 + 1
    day = min(base.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


def _format_currency_totals(income_map: dict[str, float], expense_map: dict[str, float]) -> list[str]:
    lines: list[str] = []
    currencies = sorted(set(income_map) | set(expense_map))
    for code in currencies:
        income = income_map.get(code, 0.0)
        expenses = expense_map.get(code, 0.0)
        balance = income - expenses
        lines.append(f"{code}: income {income:.2f}, expenses {expenses:.2f}, balance {balance:.2f}")
    return lines


def _period_summary(user_id: int, months: int) -> tuple[str, list[str]]:
    today = date.today()
    if months == -1:  # year to date
        start_date = today.replace(month=1, day=1)
    else:
        first_day_current = today.replace(day=1)
        start_date = _subtract_months(first_day_current, months - 1)
    with SessionLocal() as db:
        transactions = crud.list_transactions(
            db, user_id=user_id, start_date=start_date, end_date=today)

    aggregated = crud.aggregate_transactions(transactions)
    income_map: dict[str, float] = aggregated["income_by_currency"]
    expense_map: dict[str, float] = aggregated["expenses_by_currency"]
    category_totals: dict[str, float] = aggregated["category_totals"]
    monthly_totals: dict[str, dict[str, dict[str, float]]] = aggregated["monthly_totals"]

    header_lines = [f"📊 *Summary* ({start_date:%Y-%m-%d} → {today:%Y-%m-%d})"]
    currency_lines = _format_currency_totals(income_map, expense_map)
    if currency_lines:
        header_lines.extend(currency_lines)
    else:
        header_lines.append("No transactions recorded yet.")

    top_categories = sorted(
        category_totals.items(), key=lambda item: item[1], reverse=True)[:3]
    top_categories_text = "\n".join(
        f"• {name}: {amount:.2f}" for name, amount in top_categories) or "• No data yet"
    header_lines.append("\n🥇 *Top categories*")
    header_lines.append(top_categories_text)
    header_lines.append("\n(No currency conversion applied; values remain in their native currencies.)")

    monthly_rows: list[str] = []
    for month_key in sorted(monthly_totals.keys()):
        currency_parts = []
        for code in sorted(monthly_totals[month_key].keys()):
            info = monthly_totals[month_key][code]
            currency_parts.append(
                f"{code} income {info['income']:.2f}, expenses {info['expenses']:.2f}"
            )
        monthly_rows.append(f"• {month_key}: " + "; ".join(currency_parts))

    return "\n".join(header_lines), monthly_rows


async def start_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    user, created = await asyncio.to_thread(ensure_bot_user, update.effective_user)
    welcome = (
        "🎉 Welcome to your Budget Tracker assistant!\n"
        "Send me amounts like `Spent 25 on groceries today` or drop a receipt photo."
    )
    if created:
        welcome = (
            "👋 Hello and welcome! I've set up a secure account for you. "
            "Send a message like `Coffee 5.20` or drop a receipt photo to get started."
        )
    keyboard = _main_menu_keyboard()
    await update.message.reply_text(
        welcome + "\n\n*Main Menu:*\n"
        "• Send text or photo to add a transaction\n"
        "• Use buttons below for reports and transactions\n"
        "• Or use commands: /report, /transactions, /help",
        reply_markup=_main_menu_keyboard(),
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
                InlineKeyboardButton(
                    "Current month", callback_data="report:1"),
                InlineKeyboardButton(
                    "Last 3 months", callback_data="report:3"),
            ],
            [
                InlineKeyboardButton(
                    "Last 6 months", callback_data="report:6"),
                InlineKeyboardButton(
                    "Year to date", callback_data="report:ytd"),
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


async def transactions_command(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    keyboard = InlineKeyboardMarkup(
        [
            [
                InlineKeyboardButton(
                    "Last 10", callback_data="transactions:10"),
                InlineKeyboardButton(
                    "Last 20", callback_data="transactions:20"),
            ],
            [
                InlineKeyboardButton(
                    "This month", callback_data="transactions:month"),
                InlineKeyboardButton(
                    "All time", callback_data="transactions:all"),
            ],
            [
                InlineKeyboardButton(
                    "🔙 Back to Menu", callback_data="menu:back"),
            ],
        ]
    )
    await update.message.reply_text(
        "📝 *View Transactions*\n"
        "Choose how many transactions to view:",
        reply_markup=keyboard,
        parse_mode=ParseMode.MARKDOWN,
    )


async def handle_text(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    text = update.message.text or ""
    trimmed = text.strip().lower()
    if trimmed in {"report", "summary"}:
        await report_command(update, context)
        return

    pending = context.user_data.get(PENDING_TX_KEY)
    if pending and pending.get("awaiting_manual"):
        context.user_data.pop(PENDING_TX_KEY, None)
        await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="TYPING")
        await _process_payload(update, context, text=text)
        return

    # Send typing indicator for immediate feedback
    await context.bot.send_chat_action(chat_id=update.effective_chat.id, action="TYPING")

    # Process the text
    await _process_payload(update, context, text=text)


async def handle_photo(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    if not TESSERACT_AVAILABLE:
        await update.message.reply_text(
            "Image OCR isn't available because Tesseract is not installed on the server. "
            "Please install it or send the transaction details as text."
        )
        return

    status_msg = await update.message.reply_text("📸 Processing receipt image... Please wait.")

    try:
        photo = update.message.photo[-1]
        file = await context.bot.get_file(photo.file_id)
        image_bytes = bytes(await file.download_as_bytearray())

        await status_msg.edit_text("📝 Extracting text from receipt...")
        caption = update.message.caption or None
        await _process_payload(
            update,
            context,
            text=caption,
            image_bytes=image_bytes,
            require_confirmation=True,
        )
        await status_msg.delete()
    except Exception as exc:
        logger.error("Error processing photo: %s", exc)
        await status_msg.edit_text(
            "❌ Failed to process the image. Please try again with a clearer photo or send the details as text.",
            parse_mode=ParseMode.MARKDOWN,
        )


async def handle_callback(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    query = update.callback_query
    await query.answer()

    user, _ = await asyncio.to_thread(ensure_bot_user, query.from_user)
    data = query.data or ""

    if data.startswith("menu:"):
        menu_action = data.split(":")[1]
        if menu_action == "reports":
            keyboard = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "Current month", callback_data="report:1"),
                        InlineKeyboardButton(
                            "Last 3 months", callback_data="report:3"),
                    ],
                    [
                        InlineKeyboardButton(
                            "Last 6 months", callback_data="report:6"),
                        InlineKeyboardButton(
                            "Year to date", callback_data="report:ytd"),
                    ],
                    [
                        InlineKeyboardButton(
                            "🔙 Back to Menu", callback_data="menu:back"),
                    ],
                ]
            )
            await query.edit_message_text("📊 *Reports*\nChoose a time period:", reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)
        elif menu_action == "transactions":
            keyboard = InlineKeyboardMarkup(
                [
                    [
                        InlineKeyboardButton(
                            "Last 10", callback_data="transactions:10"),
                        InlineKeyboardButton(
                            "Last 20", callback_data="transactions:20"),
                    ],
                    [
                        InlineKeyboardButton(
                            "This month", callback_data="transactions:month"),
                        InlineKeyboardButton(
                            "All time", callback_data="transactions:all"),
                    ],
                    [
                        InlineKeyboardButton(
                            "🔙 Back to Menu", callback_data="menu:back"),
                    ],
                ]
            )
            await query.edit_message_text("📝 *My Transactions*\nChoose how many to view:", reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)
        elif menu_action == "help":
            help_text = (
                "📌 *Tips & Shortcuts*\n\n"
                "• Send text like `Salary 1200` or `Coffee 5.20`\n"
                "• Drop receipt photos – I'll read them automatically\n"
                "• Use /report for spending summaries\n"
                "• Use /transactions to view your entries\n"
                "• Use /recent for the last 5 transactions\n\n"
                "All commands are also available via the menu buttons!"
            )
            keyboard = InlineKeyboardMarkup(
                [[InlineKeyboardButton(
                    "🔙 Back to Menu", callback_data="menu:back")]]
            )
            await query.edit_message_text(help_text, reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)
        elif menu_action == "back":
            keyboard = _main_menu_keyboard()
            await query.edit_message_text("🏠 *Main Menu*", reply_markup=keyboard, parse_mode=ParseMode.MARKDOWN)
    elif data == "clear:prompt":
        keyboard = InlineKeyboardMarkup(
            [
                [InlineKeyboardButton("✅ Yes, delete all", callback_data="clear:yes")],
                [InlineKeyboardButton("❌ Cancel", callback_data="clear:cancel")],
            ]
        )
        await query.edit_message_text(
            "⚠️ *Delete all transactions?*\nThis will remove every entry in your account.",
            reply_markup=keyboard,
            parse_mode=ParseMode.MARKDOWN,
        )
    elif data == "clear:cancel":
        await query.edit_message_text(
                "Deletion cancelled.", reply_markup=_main_menu_keyboard(), parse_mode=ParseMode.MARKDOWN
        )
    elif data == "clear:yes":
        deleted = await asyncio.to_thread(_clear_user_transactions, user.id)
        await query.edit_message_text(
            f"🗑 Removed {deleted} transaction{'s' if deleted != 1 else ''}.",
            parse_mode=ParseMode.MARKDOWN,
        )
        if query.message:
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text="✅ Your history is clean.",
                reply_markup=_main_menu_keyboard(),
            )
    elif data == "action:add":
        await query.edit_message_text(
            "👍 Ready! Send the next transaction as text or drop another receipt.",
            reply_markup=None,
        )
    elif data == "action:report_current":
        header, monthly_rows = await asyncio.to_thread(_period_summary, user.id, 1)
        monthly_text = "\n".join(
            f"• {month}: income {inc:.2f}, expenses {exp:.2f}" for month, inc, exp in monthly_rows
        ) or "No data yet."
        keyboard = InlineKeyboardMarkup(
            [[InlineKeyboardButton("🔙 Back", callback_data="menu:back")]]
        )
        await query.edit_message_text(
            f"{header}\n\n📆 *Monthly breakdown*\n{monthly_text}",
            reply_markup=keyboard,
            parse_mode=ParseMode.MARKDOWN,
        )
    elif data == "action:undo":
        last_tx = context.user_data.get(LAST_SAVED_TX_KEY)
        if not last_tx:
            await query.edit_message_text(
                "No recent transaction to undo.", reply_markup=None
            )
            return
        removed = await asyncio.to_thread(
            _delete_transaction, last_tx["user_id"], last_tx["transaction_id"]
        )
        if removed:
            await query.edit_message_text(
                "↩️ Last transaction removed.", reply_markup=None
            )
            context.user_data.pop(LAST_SAVED_TX_KEY, None)
            await context.bot.send_message(
                chat_id=query.message.chat_id,
                text="You can continue logging expenses.",
                reply_markup=_main_menu_keyboard(),
            )
        else:
            await query.edit_message_text(
                "Couldn't undo the last transaction (maybe already cleared).",
                reply_markup=None,
            )
    elif data.startswith("report:"):
        months = REPORT_PRESETS.get(data)
        if months is None:
            await query.edit_message_text("Unsupported report selection.")
            return
        header, monthly_rows = await asyncio.to_thread(_period_summary, user.id, months)
        monthly_text = "\n".join(
            f"• {month}: income {inc:.2f}, expenses {exp:.2f}"
            for month, inc, exp in monthly_rows
        ) or "No monthly data yet."
        keyboard = InlineKeyboardMarkup(
            [[InlineKeyboardButton("🔙 Back to Reports",
                                   callback_data="menu:reports")]]
        )
        await query.edit_message_text(
            f"{header}\n\n📆 *Monthly breakdown*\n{monthly_text}",
            reply_markup=keyboard,
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
        keyboard = InlineKeyboardMarkup(
            [[InlineKeyboardButton(
                "🔙 Back to Menu", callback_data="menu:back")]]
        )
        await query.edit_message_text("📝 Latest entries:\n" + "\n".join(lines), reply_markup=keyboard)
    elif data.startswith("transactions:"):
        filter_type = data.split(":")[1]
        with SessionLocal() as db:
            if filter_type == "all":
                transactions = crud.list_transactions(db, user.id)
            elif filter_type == "month":
                today = date.today()
                start_date = today.replace(day=1)
                transactions = crud.list_transactions(
                    db, user.id, start_date=start_date, end_date=today)
            else:
                count = int(filter_type)
                transactions = crud.list_transactions(db, user.id)[:count]

        if not transactions:
            keyboard = InlineKeyboardMarkup(
                [[InlineKeyboardButton("🔙 Back", callback_data="menu:transactions")]])
            await query.edit_message_text("No transactions found.", reply_markup=keyboard)
            return

        lines = [
            f"#{tx.id} {tx.date:%Y-%m-%d} • {tx.kind.value.capitalize()} • {tx.amount:.2f} {settings.default_currency} – {tx.category}\n  {tx.description[:50]}"
            for tx in transactions[:20]  # Limit to 20 for display
        ]
        keyboard = InlineKeyboardMarkup(
            [[InlineKeyboardButton(
                "🔙 Back to Transactions", callback_data="menu:transactions")]]
        )
        title = f"📝 Your Transactions ({len(transactions)} total)"
        await query.edit_message_text(title + "\n" + "\n".join(lines), reply_markup=keyboard)
    elif data == "confirm:yes":
        pending = context.user_data.pop(PENDING_TX_KEY, None)
        if not pending:
            await query.edit_message_text("No pending transaction to confirm.")
            return
        transaction = TransactionCreate(**pending["transaction"])
        extra = pending["extra"]
        if not extra.get("currency"):
            extra["currency"] = settings.default_currency
        tx_id, ordinal = await asyncio.to_thread(_store_transaction, pending["user_id"], transaction, extra)
        context.user_data[LAST_SAVED_TX_KEY] = {
            "transaction_id": tx_id,
            "user_id": pending["user_id"],
            "ordinal": ordinal,
        }
        confidence_note = ""
        if extra.get("confidence", 1.0) < CONFIDENCE_THRESHOLD:
            confidence_note = "\n⚠️ Confidence was low, please double-check in the app."
        await query.edit_message_text(
            (
                f"✅ Saved *{transaction.type.capitalize()}* #{ordinal}\n"
                f"• Amount: {transaction.amount:.2f} {extra['currency']}\n"
                f"• Category: {transaction.category}\n"
                f"• Date: {transaction.date.isoformat()}"
                f"{confidence_note}\n\nChoose a next step below."
            ),
            parse_mode=ParseMode.MARKDOWN,
            reply_markup=_post_save_keyboard(),
        )
    elif data == "confirm:no":
        pending = context.user_data.get(PENDING_TX_KEY)
        if not pending:
            await query.edit_message_text("No pending transaction to edit.")
            return
        context.user_data[PENDING_TX_KEY]["awaiting_manual"] = True
        await query.edit_message_text(
            "No problem. Send the corrected details in text, for example:\n`Spent 42 on dinner yesterday`",
            parse_mode=ParseMode.MARKDOWN,
        )


def build_application() -> Application:
    if not settings.telegram_bot_token:
        raise RuntimeError("TELEGRAM_BOT is missing from configuration.")

    application = ApplicationBuilder().token(settings.telegram_bot_token).build()

    application.add_handler(CommandHandler("start", start_command))
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("report", report_command))
    application.add_handler(CommandHandler("recent", recent_command))
    application.add_handler(CommandHandler(
        "transactions", transactions_command))
    application.add_handler(CallbackQueryHandler(handle_callback))
    application.add_handler(MessageHandler(filters.PHOTO, handle_photo))
    application.add_handler(MessageHandler(
        filters.TEXT & ~filters.COMMAND, handle_text))
    return application


def main() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    if not settings.telegram_bot_token:
        raise SystemExit(
            "Please set TELEGRAM_BOT in the environment to run the bot.")
    application = build_application()
    logger.info("Starting Telegram bot...")
    application.run_polling(drop_pending_updates=True)


if __name__ == "__main__":
    main()
