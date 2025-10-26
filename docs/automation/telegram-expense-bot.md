# Telegram Expense Automation

This automation adds a Telegram entry point for the expense tracker. Users can send a message or receipt photo to a bot, the workflow extracts structured data, categorises the expense, stores the transaction in PostgreSQL, and replies with the result.  
➡️ The repository now ships a native Python bot (`SERVICE=bot ./start.sh`). Use this n8n flow when you prefer a low-code orchestrator or need to fan out actions to external systems.

## Prerequisites

- **Telegram Bot Token** – Create a bot via [@BotFather](https://core.telegram.org/bots#6-botfather) and note the token.
- **n8n instance** – Self-hosted or cloud. Ensure it can reach both Telegram and the project database.
- **LLM provider** – The workflow uses two OpenAI Chat nodes by default (one for OCR/extraction and one for category classification). Substitute with another n8n-compatible provider if preferred.
- **Database access** – Reuse the Postgres database that backs the FastAPI app. Credentials live in `.env` (`DATABASE_URL`) and in `docker-compose.yml`.

## Database Preparation

The workflow writes directly into the `transactions` table using the same schema as the backend. Create a dedicated bot user in the `users` table so that new rows are associated with a real account:

```sql
INSERT INTO users (name, username, email, password_hash)
VALUES ('Telegram Bot', 'telegram-bot', 'bot@example.com', '<bcrypt-hash>');
```

The hash can be generated with the FastAPI app (`passlib` PBKDF2) or by calling `/api/auth/register` once and reusing the stored password hash.

Note the resulting `id`; set it inside n8n so all automated expenses share the same owner.

## Workflow Outline

The workflow mirrors the classroom example and the diagram provided:

1. **Telegram Trigger** – `getUpdates` mode. Captures both text messages and photo attachments.
2. **Switch (Image or Not)** – Checks if the message contains a `photo` array.
   - **Image path**: download the largest image using `Telegram -> Get File` and hand the public URL to the first LLM node.
   - **Text path**: pass the raw message text directly to the second LLM node.
3. **Analyze Image** *(LLM)* – Prompt example:
   ```
   Extract merchant, total amount, currency (if any), purchase date, and a short description from this receipt image. Return JSON with keys: amount (numeric), currency, date (ISO 8601), category_guess, description.
   ```
4. **Categorize Expense** *(LLM with memory)* – Combines the extracted data (from step 3 or the raw message) with a category taxonomy you define. Prompt example:
   ```
   You are an expense classifier. Decide on the best category from: ["Food", "Transport", "Rent", "Utilities", "Entertainment", "Health", "Other"]. Return JSON with keys: category, type ("income" or "expense"), confidence (0-1).
   ```
   Enable Simple Memory in n8n to accumulate previous user-specific decisions and improve consistency.
5. **Insert rows in a table (Postgres)** – Map the LLM output to the schema:
   - `user_id`: static ID of the bot account.
   - `amount`: parsed float.
   - `date`: ISO string from extraction (fallback to `now()` if absent).
   - `category`: output of the categoriser.
   - `kind`: `expense` or `income`.
   - `description`: brief summary (include OCR'ed merchant if available).
6. **Respond on Telegram** – Send a confirmation message back to the chat, including the detected category and amount. On failure branches, return a helpful error and skip the DB insert.

## Suggested Enhancements

- **Confidence thresholds** – If the categoriser returns a low confidence, ask the user to confirm via Telegram buttons (`Reply Markup`).
- **Receipt storage** – Persist the receipt image to S3/Supabase and store the URL in the `description` column or a dedicated table.
- **Multi-user support** – Maintain a mapping table (`telegram_chat_id → user_id`) so each chatter gets their own ledger.
- **API integration** – Instead of writing directly to Postgres, authenticate as the bot user and call `POST /api/me/transactions`. This keeps business rules centralised in the FastAPI service.

## Importing the Workflow

1. In n8n, click **Import from File** and select `automation/n8n-telegram-expense-workflow.json`.
2. Populate the following credentials/secrets:
   - Telegram API credentials.
   - OpenAI (or alt provider) key.
   - Postgres connection (host, port, database, user, password).
3. Update the `user_id` value in the Postgres node if you created a unique bot user.
4. Activate the workflow. Send a message or receipt to the bot to validate end-to-end ingestion.

Once activated, every new Telegram message is captured, enriched, and made available to the web/REST clients with no manual data entry.
