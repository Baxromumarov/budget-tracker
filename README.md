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
cp ../.env.example .env  # set DATABASE_URL / JWT_SECRET as needed
uvicorn app.main:app --reload --port 8000
```

The API becomes available at `http://localhost:8000`. Interactive docs live at `http://localhost:8000/docs`.

### Key Endpoints

- `POST /api/auth/register` – Create an account (name, username, optional email, password).
- `POST /api/auth/login` – Exchange credentials for a JWT access token.
- `GET /api/users/me` – Return the authenticated user profile.
- `GET /api/me/transactions` – List and filter the current user's transactions.
- `POST /api/me/transactions` – Add a new income/expense entry.
- `PUT /api/me/transactions/{transaction_id}` – Update an existing entry.
- `DELETE /api/me/transactions/{transaction_id}` – Delete an entry.
- `GET /api/me/summary?month=10&year=2025` – Monthly roll-up (income, expenses, balance, top category).
- `GET /api/me/reports/{format}?month=10&year=2025` – Download CSV/JSON exports for the selected month.

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

- Registration & login landing pages backed by JWT authentication.
- Personalised dashboard with profile card, monthly overview, and logout control.
- Transaction CRUD with quick category chips, smarter filters, and improved table styling.
- Real-time monthly summaries with balance/income/expense/top category cards.
- One-click CSV/JSON downloads powered by the domain report generators.

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
2. Visit `http://localhost:5173`, register a new account, and verify login redirects to the dashboard.
3. Add, edit, and delete transactions; confirm the table and monthly summary update immediately.
4. Export CSV/JSON reports for the active month and inspect the files.
5. Hit `http://localhost:8000/docs` to explore and test endpoints directly (authorise with the JWT returned from login).

## Next Steps

- Add refresh tokens or session revocation lists for stronger security.
- Introduce automated unit/integration tests (e.g., pytest & React Testing Library).
- Extend reporting (e.g., yearly dashboards, charts).
- Integrate background job processing for scheduled exports or alerts.
