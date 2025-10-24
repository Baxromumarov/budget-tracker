import { useCallback, useEffect, useMemo, useState } from "react";
import { createTransaction, deleteTransaction, fetchTransactions, updateTransaction } from "../api/transactions";
import { fetchMonthlySummary } from "../api/reports";
import TransactionForm from "../components/TransactionForm";
import TransactionTable from "../components/TransactionTable";
import SummaryCards from "../components/SummaryCards";
import ReportExport from "../components/ReportExport";
import ProfileCard from "../components/ProfileCard";
import { useAuth } from "../context/AuthContext";
import { MonthlySummary, Transaction, TransactionInput, TransactionType } from "../types";

const today = new Date();
const initialMonth = today.getMonth() + 1;
const initialYear = today.getFullYear();

interface Filters {
  category: string;
  type: TransactionType | "";
  start_date: string;
  end_date: string;
}

const emptyFilters: Filters = {
  category: "",
  type: "",
  start_date: "",
  end_date: "",
};

export default function DashboardPage() {
  const { user, logout } = useAuth();
  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [summary, setSummary] = useState<MonthlySummary | null>(null);
  const [month, setMonth] = useState<number>(initialMonth);
  const [year, setYear] = useState<number>(initialYear);
  const [filters, setFilters] = useState<Filters>(emptyFilters);
  const [editingTransaction, setEditingTransaction] = useState<Transaction | null>(null);
  const [listLoading, setListLoading] = useState(false);
  const [formLoading, setFormLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [toast, setToast] = useState<string | null>(null);

  const filterPayload = useMemo(
    () => ({
      category: filters.category || undefined,
      type: filters.type || undefined,
      start_date: filters.start_date || undefined,
      end_date: filters.end_date || undefined,
    }),
    [filters]
  );

  const loadTransactions = useCallback(async () => {
    if (!user) return;
    setListLoading(true);
    try {
      const data = await fetchTransactions(filterPayload);
      setTransactions(data);
      setError(null);
    } catch (err) {
      console.error(err);
      setError("Unable to fetch transactions.");
      setToast(null);
    } finally {
      setListLoading(false);
    }
  }, [user, filterPayload]);

  const loadSummary = useCallback(async () => {
    if (!user) return;
    try {
      const data = await fetchMonthlySummary(month, year);
      setSummary(data);
    } catch (err) {
      console.error(err);
      setSummary(null);
    }
  }, [user, month, year]);

  useEffect(() => {
    if (!user) return;
    void loadTransactions();
    void loadSummary();
  }, [user, loadTransactions, loadSummary]);

  useEffect(() => {
    if (!toast) return;
    const timer = setTimeout(() => setToast(null), 3500);
    return () => clearTimeout(timer);
  }, [toast]);

  const refreshData = async () => {
    await Promise.all([loadTransactions(), loadSummary()]);
  };

  const handleCreateTransaction = async (input: TransactionInput) => {
    setFormLoading(true);
    try {
      await createTransaction(input);
      setToast("Transaction added.");
      setError(null);
      await refreshData();
    } catch (err) {
      console.error(err);
      setError("Unable to create transaction.");
      setToast(null);
    } finally {
      setFormLoading(false);
    }
  };

  const handleUpdateTransaction = async (input: TransactionInput) => {
    if (!editingTransaction) return;
    setFormLoading(true);
    try {
      await updateTransaction(editingTransaction.id, input);
      setToast("Transaction updated.");
      setEditingTransaction(null);
      setError(null);
      await refreshData();
    } catch (err) {
      console.error(err);
      setError("Unable to update transaction.");
      setToast(null);
    } finally {
      setFormLoading(false);
    }
  };

  const handleDeleteTransaction = async (transaction: Transaction) => {
    const confirmed = window.confirm("Delete this transaction?");
    if (!confirmed) return;
    try {
      await deleteTransaction(transaction.id);
      setToast("Transaction deleted.");
      setError(null);
      await refreshData();
    } catch (err) {
      console.error(err);
      setError("Unable to delete transaction.");
      setToast(null);
    }
  };

  if (!user) {
    return null;
  }

  return (
    <div className="dashboard-wrapper">
      <header className="hero">
        <div className="hero-content container">
          <div>
            <p className="eyebrow">Budget Manager</p>
            <h1>Balance your money with confidence</h1>
            <p className="muted">
              Log spending, stay on top of income, and export polished reports for your personal finances.
            </p>
          </div>
          <div className="hero-badge">
            <span>Active</span>
            <strong>{user.username}</strong>
          </div>
        </div>
      </header>

      <main className="container dashboard-content">
        <ProfileCard user={user} onLogout={logout} />

        {(error || toast) && (
          <div className={`alert ${error ? "danger" : "success"}`}>
            {error ?? toast}
          </div>
        )}

        <section className="card">
          <div className="section-header">
            <div>
              <h2>Monthly Overview</h2>
              <p className="muted">Select a period to review your earnings, spending, and savings.</p>
            </div>
            <div className="section-controls">
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
              {summary && <ReportExport month={month} year={year} />}
            </div>
          </div>
          <SummaryCards summary={summary} />
        </section>

        <section className="card">
          <div className="section-header">
            <h2>{editingTransaction ? "Edit Transaction" : "Add Transaction"}</h2>
            {editingTransaction && (
              <button className="btn secondary" onClick={() => setEditingTransaction(null)}>
                Cancel edit
              </button>
            )}
          </div>
          <TransactionForm
            onSubmit={editingTransaction ? handleUpdateTransaction : handleCreateTransaction}
            loading={formLoading}
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
          />
        </section>

        <section className="card">
          <div className="section-header">
            <h2>Filters</h2>
            <div className="section-controls">
              <button className="btn secondary" onClick={() => setFilters(emptyFilters)}>
                Clear all
              </button>
              <button className="btn secondary" onClick={() => void loadTransactions()}>
                Refresh list
              </button>
            </div>
          </div>
          <div className="grid two">
            <label>
              Category
              <input
                value={filters.category}
                onChange={(event) => setFilters((prev) => ({ ...prev, category: event.target.value }))}
                placeholder="e.g. Groceries"
              />
            </label>
            <label>
              Type
              <select
                value={filters.type}
                onChange={(event) =>
                  setFilters((prev) => ({ ...prev, type: event.target.value as TransactionType | "" }))
                }
              >
                <option value="">All</option>
                <option value="income">Income</option>
                <option value="expense">Expense</option>
              </select>
            </label>
          </div>
          <div className="grid two">
            <label>
              Start date
              <input
                type="date"
                value={filters.start_date}
                onChange={(event) => setFilters((prev) => ({ ...prev, start_date: event.target.value }))}
              />
            </label>
            <label>
              End date
              <input
                type="date"
                value={filters.end_date}
                onChange={(event) => setFilters((prev) => ({ ...prev, end_date: event.target.value }))}
              />
            </label>
          </div>
        </section>

        <section className="card">
          <div className="section-header">
            <h2>Transactions</h2>
            {listLoading && <span className="muted">Loading...</span>}
          </div>
          <TransactionTable
            transactions={transactions}
            onEdit={(transaction) => setEditingTransaction(transaction)}
            onDelete={(transaction) => handleDeleteTransaction(transaction)}
          />
        </section>
      </main>
    </div>
  );
}
