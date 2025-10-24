import { useEffect, useMemo, useState } from "react";
import { createUser, fetchUsers } from "./api/users";
import {
  createTransaction,
  deleteTransaction,
  fetchTransactions,
  updateTransaction,
} from "./api/transactions";
import { fetchMonthlySummary } from "./api/reports";
import TransactionForm from "./components/TransactionForm";
import TransactionTable from "./components/TransactionTable";
import SummaryCards from "./components/SummaryCards";
import ReportExport from "./components/ReportExport";
import { MonthlySummary, Transaction, TransactionInput, TransactionType, User } from "./types";

const currentDate = new Date();
const defaultMonth = currentDate.getMonth() + 1;
const defaultYear = currentDate.getFullYear();

interface Filters {
  category: string;
  type: TransactionType | "";
  start_date: string;
  end_date: string;
}

const initialFilters: Filters = {
  category: "",
  type: "",
  start_date: "",
  end_date: "",
};

export default function App() {
  const [users, setUsers] = useState<User[]>([]);
  const [selectedUserId, setSelectedUserId] = useState<number | null>(null);
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [summary, setSummary] = useState<MonthlySummary | null>(null);
  const [month, setMonth] = useState<number>(defaultMonth);
  const [year, setYear] = useState<number>(defaultYear);
  const [filters, setFilters] = useState<Filters>(initialFilters);
  const [loading, setLoading] = useState<boolean>(false);
  const [editingTransaction, setEditingTransaction] = useState<Transaction | null>(null);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    const loadUsers = async () => {
      try {
        const data = await fetchUsers();
        setUsers(data);
        if (data.length && !selectedUserId) {
          setSelectedUserId(data[0].id);
        }
      } catch (err) {
        console.error(err);
        setError("Unable to load users. Please ensure the backend is running.");
      }
    };
    void loadUsers();
    // We intentionally run this only once on mount.
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  useEffect(() => {
    if (!selectedUserId) {
      return;
    }

    const loadTransactions = async () => {
      setLoading(true);
      try {
        const data = await fetchTransactions(selectedUserId, {
          category: filters.category || undefined,
          type: filters.type || undefined,
          start_date: filters.start_date || undefined,
          end_date: filters.end_date || undefined,
        });
        setTransactions(data);
      } catch (err) {
        console.error(err);
        setError("Unable to fetch transactions.");
      } finally {
        setLoading(false);
      }
    };

    const loadSummary = async () => {
      try {
        const data = await fetchMonthlySummary(selectedUserId, month, year);
        setSummary(data);
      } catch (err) {
        console.error(err);
        setSummary(null);
      }
    };

    void loadTransactions();
    void loadSummary();
  }, [selectedUserId, filters, month, year]);

  const handleCreateUser = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    const formData = new FormData(event.currentTarget);
    const name = (formData.get("name") as string).trim();
    const email = (formData.get("email") as string).trim();

    if (!name || !email) {
      setError("Name and email are required.");
      return;
    }

    try {
      const user = await createUser({ name, email });
      setUsers((prev) => [...prev, user]);
      setSelectedUserId(user.id);
      event.currentTarget.reset();
      setError(null);
    } catch (err) {
      console.error(err);
      setError("Unable to create user. Email may already be taken.");
    }
  };

  const refreshTransactions = async () => {
    if (!selectedUserId) return;
    const data = await fetchTransactions(selectedUserId, {
      category: filters.category || undefined,
      type: filters.type || undefined,
      start_date: filters.start_date || undefined,
      end_date: filters.end_date || undefined,
    });
    setTransactions(data);
  };

  const refreshSummary = async () => {
    if (!selectedUserId) return;
    try {
      const data = await fetchMonthlySummary(selectedUserId, month, year);
      setSummary(data);
    } catch (err) {
      console.error(err);
      setSummary(null);
    }
  };

  const handleCreateTransaction = async (input: TransactionInput) => {
    if (!selectedUserId) return;
    try {
      await createTransaction(selectedUserId, input);
      await Promise.all([refreshTransactions(), refreshSummary()]);
      setError(null);
    } catch (err) {
      console.error(err);
      setError("Unable to create transaction.");
    }
  };

  const handleUpdateTransaction = async (input: TransactionInput) => {
    if (!selectedUserId || !editingTransaction) return;
    try {
      await updateTransaction(selectedUserId, editingTransaction.id, input);
      setEditingTransaction(null);
      await Promise.all([refreshTransactions(), refreshSummary()]);
      setError(null);
    } catch (err) {
      console.error(err);
      setError("Unable to update transaction.");
    }
  };

  const handleDeleteTransaction = async (transaction: Transaction) => {
    if (!selectedUserId) return;
    if (!window.confirm("Delete this transaction?")) return;
    try {
      await deleteTransaction(selectedUserId, transaction.id);
      await Promise.all([refreshTransactions(), refreshSummary()]);
      setError(null);
    } catch (err) {
      console.error(err);
      setError("Unable to delete transaction.");
    }
  };

  const selectedUser = useMemo(
    () => users.find((user) => user.id === selectedUserId) ?? null,
    [users, selectedUserId]
  );

  return (
    <div className="container">
      <header className="header">
        <div>
          <h1 className="header-title">Expense Tracker</h1>
          <p style={{ margin: 0, color: "#475569" }}>
            Manage your income, expenses, and download monthly budget reports.
          </p>
        </div>
        {selectedUser && (
          <div>
            <strong>{selectedUser.name}</strong>
            <p style={{ margin: 0, color: "#64748b" }}>{selectedUser.email}</p>
          </div>
        )}
      </header>

      {error && (
        <div className="card" style={{ borderLeft: "4px solid #f87171" }}>
          <strong>Error:</strong> {error}
        </div>
      )}

      <div className="card">
        <h2 style={{ marginTop: 0 }}>Select User</h2>
        <div style={{ display: "flex", gap: "1rem", alignItems: "center", marginBottom: "1rem" }}>
          <select
            value={selectedUserId ?? ""}
            onChange={(event) => {
              const value = event.target.value;
              setSelectedUserId(value ? Number(value) : null);
            }}
            style={{ minWidth: "220px" }}
          >
            <option value="" disabled>
              Choose a user
            </option>
            {users.map((user) => (
              <option key={user.id} value={user.id}>
                {user.name}
              </option>
            ))}
          </select>
          <span style={{ color: "#94a3b8" }}>or create a new user below</span>
        </div>
        <form onSubmit={handleCreateUser} style={{ display: "flex", gap: "1rem", flexWrap: "wrap" }}>
          <input type="text" name="name" placeholder="Full name" required />
          <input type="email" name="email" placeholder="Email address" required />
          <button className="btn" type="submit">
            Add User
          </button>
        </form>
      </div>

      <div className="card">
        <h2 style={{ marginTop: 0 }}>Monthly Summary</h2>
        <div style={{ display: "flex", gap: "1rem", flexWrap: "wrap", marginBottom: "1rem" }}>
          <label>
            Month
            <input
              type="number"
              min={1}
              max={12}
              value={month}
              onChange={(event) => setMonth(Number(event.target.value))}
            />
          </label>
          <label>
            Year
            <input
              type="number"
              min={2000}
              max={2100}
              value={year}
              onChange={(event) => setYear(Number(event.target.value))}
            />
          </label>
          {selectedUserId && summary && <ReportExport userId={selectedUserId} month={month} year={year} />}
        </div>
        <SummaryCards summary={summary} />
      </div>

      <div className="card">
        <div className="header">
          <h2 style={{ marginTop: 0 }}>{editingTransaction ? "Edit Transaction" : "Add Transaction"}</h2>
        </div>
        <TransactionForm
          onSubmit={editingTransaction ? handleUpdateTransaction : handleCreateTransaction}
          loading={loading}
          defaultValues={
            editingTransaction
              ? {
                  amount: editingTransaction.amount,
                  date: editingTransaction.date.substring(0, 10),
                  category: editingTransaction.category,
                  type: editingTransaction.type,
                  description: editingTransaction.description ?? "",
                }
              : undefined
          }
          onCancel={() => setEditingTransaction(null)}
        />
      </div>

      <div className="card">
        <h2 style={{ marginTop: 0 }}>Filters</h2>
        <div className="grid two">
          <label>
            Category
            <input
              type="text"
              value={filters.category}
              onChange={(event) => setFilters((prev) => ({ ...prev, category: event.target.value }))}
            />
          </label>
          <label>
            Type
            <select
              value={filters.type}
              onChange={(event) => setFilters((prev) => ({ ...prev, type: event.target.value as TransactionType | "" }))}
            >
              <option value="">All</option>
              <option value="income">Income</option>
              <option value="expense">Expense</option>
            </select>
          </label>
        </div>
        <div className="grid two">
          <label>
            Start Date
            <input
              type="date"
              value={filters.start_date}
              onChange={(event) => setFilters((prev) => ({ ...prev, start_date: event.target.value }))}
            />
          </label>
          <label>
            End Date
            <input
              type="date"
              value={filters.end_date}
              onChange={(event) => setFilters((prev) => ({ ...prev, end_date: event.target.value }))}
            />
          </label>
        </div>
        <div style={{ display: "flex", gap: "1rem" }}>
          <button className="btn" onClick={() => setFilters(initialFilters)}>
            Clear Filters
          </button>
          <button className="btn secondary" onClick={() => void refreshTransactions()}>
            Refresh
          </button>
        </div>
      </div>

      <TransactionTable
        transactions={transactions}
        onEdit={(transaction) => setEditingTransaction(transaction)}
        onDelete={async (transaction) => {
          await handleDeleteTransaction(transaction);
        }}
      />

      {loading && <p>Loading transactions...</p>}
    </div>
  );
}
