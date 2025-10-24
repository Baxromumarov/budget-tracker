# Expense Tracker / Budget Manager

Full-stack expense tracking platform based on the provided project brief. The system lets users register, record income and expenses, inspect running balances, and export monthly reports in CSV or JSON formats. It demonstrates the requested object-oriented principles (encapsulation, inheritance, polymorphism, abstraction) in the Python backend while offering a modern React front end and a PostgreSQL datastore. Everything is containerised for easy deployment.

## Project Structure

- `backend/` – FastAPI application with SQLAlchemy models, domain entities, and REST endpoints.
- `frontend/` – Vite + React (TypeScript) single-page application for managing transactions and reports.
- `docker-compose.yml` – One-command deployment for PostgreSQL, backend API, and frontend static site.
- `.env.example` – Sample environment configuration for local development.

## Backend (FastAPI)

### Requirements

- Python 3.11+
- PostgreSQL 13+

### Local setup

```bash
cd backend
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp ../.env.example .env  # adjust DATABASE_URL if needed
uvicorn app.main:app --reload --port 8000
```

The API becomes available at `http://localhost:8000`. Interactive docs live at `http://localhost:8000/docs`.

### Key Endpoints

- `POST /api/users` – Create a user.
- `GET /api/users/{user_id}` – Retrieve profile details.
- `POST /api/users/{user_id}/transactions` – Add income or expense.
- `GET /api/users/{user_id}/transactions` – List and filter transactions.
- `PUT /api/users/{user_id}/transactions/{transaction_id}` – Update an entry.
- `DELETE /api/users/{user_id}/transactions/{transaction_id}` – Remove an entry.
- `GET /api/users/{user_id}/summary?month=10&year=2025` – Monthly roll-up (income, expenses, balance, top category).
- `GET /api/users/{user_id}/reports/{format}?month=10&year=2025` – Download CSV/JSON exports.

### Data Model & OOP Mapping

- `TransactionModel`, `UserModel` persist state in Postgres.
- Domain layer (`app/domain/entities.py`) implements `Transaction`, `IncomeTransaction`, `ExpenseTransaction`, `User`, `BudgetManager`, `ReportGenerator`, `CSVReportGenerator`, `JSONReportGenerator` mirroring the specification.
- REST routers orchestrate persistence (SQLAlchemy) and domain logic (report generation) to showcase encapsulation, inheritance, polymorphism, and abstraction.

## Frontend (React + Vite)

### Requirements

- Node.js 18+ (20+ recommended)

### Local setup

```bash
cd frontend
npm install
npm run dev
```

By default the SPA runs on `http://localhost:5173` and expects the API at `http://localhost:8000/api`. Use `VITE_API_URL` to point at a different backend when building:

```bash
VITE_API_URL="https://your-api.example.com/api" npm run build
```

### Features

- User onboarding (create/select users).
- Transaction CRUD with quick category chips.
- Interactive filters (category, type, date range).
- Monthly summaries with balance/income/expense/top category cards.
- One-click CSV/JSON downloads backed by the domain report generators.

## Docker Deployment

Ensure Docker and Docker Compose are installed, then from the repository root run:

```bash
docker compose up --build
```

Services:

- `db` – PostgreSQL instance with persistent volume (`postgres_data`).
- `backend` – FastAPI application exposed on `http://localhost:8000`.
- `frontend` – Nginx-served SPA on `http://localhost:5173`.

Override environment values by creating a `.env` file (based on `.env.example`) or editing the compose file.

## Running Tests / Validation

Currently there are no automated tests. Recommended smoke checks:

1. Start the stack (`docker compose up --build`).
2. Visit `http://localhost:5173`, create a user, add transactions, and verify summary updates in real time.
3. Download CSV/JSON reports and inspect the contents.
4. Hit `http://localhost:8000/docs` to explore and test endpoints directly.

## Next Steps

- Add authentication and per-user sessions.
- Introduce automated unit/integration tests (e.g., pytest & React Testing Library).
- Extend reporting (e.g., yearly dashboards, charts).
- Integrate background job processing for scheduled exports or alerts.
