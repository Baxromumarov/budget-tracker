import { useEffect, useState } from "react";
import { TransactionInput, TransactionType } from "../types";

export interface TransactionFormProps {
  onSubmit: (input: TransactionInput) => Promise<void>;
  loading?: boolean;
  defaultValues?: TransactionInput;
  onCancel?: () => void;
}

const initialForm: TransactionInput = {
  amount: 0,
  date: new Date().toISOString().substring(0, 10),
  category: "",
  type: "expense",
  description: "",
};

const categories: Record<TransactionType, string[]> = {
  income: ["Salary", "Freelance", "Investments", "Gift"],
  expense: ["Food", "Rent", "Utilities", "Transport", "Entertainment", "Healthcare"],
};

export default function TransactionForm({ onSubmit, loading, defaultValues, onCancel }: TransactionFormProps) {
  const [form, setForm] = useState<TransactionInput>(defaultValues ?? initialForm);
  const [errors, setErrors] = useState<string | null>(null);

  useEffect(() => {
    if (defaultValues) {
      setForm(defaultValues);
    }
  }, [defaultValues]);

  const handleChange = (event: React.ChangeEvent<HTMLInputElement | HTMLSelectElement | HTMLTextAreaElement>) => {
    const { name, value } = event.target;
    setForm((prev) => ({
      ...prev,
      [name]: name === "amount" ? parseFloat(value || "0") : value,
    }));
  };

  const handleSubmit = async (event: React.FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (!form.category || !form.amount || !form.date) {
      setErrors("Please fill in all required fields.");
      return;
    }
    setErrors(null);
    await onSubmit({
      amount: form.amount,
      date: form.date,
      category: form.category,
      type: form.type,
      description: form.description ?? "",
    });
    if (!defaultValues) {
      setForm(initialForm);
    }
  };

  const suggestedCategories = categories[form.type];

  return (
    <form onSubmit={handleSubmit}>
      <div className="grid two">
        <label>
          Amount
          <input
            type="number"
            name="amount"
            min="0"
            step="0.01"
            value={form.amount}
            onChange={handleChange}
            required
          />
        </label>
        <label>
          Date
          <input type="date" name="date" value={form.date} onChange={handleChange} required />
        </label>
      </div>

      <div className="grid two">
        <label>
          Category
          <input type="text" name="category" value={form.category} onChange={handleChange} required />
        </label>
        <label>
          Type
          <select name="type" value={form.type} onChange={handleChange}>
            <option value="expense">Expense</option>
            <option value="income">Income</option>
          </select>
        </label>
      </div>

      {suggestedCategories.length > 0 && (
        <div className="chips">
          {suggestedCategories.map((category) => (
            <button
              key={category}
              type="button"
              className="chip"
              onClick={() => setForm((prev) => ({ ...prev, category }))}
            >
              {category}
            </button>
          ))}
        </div>
      )}

      <label>
        Description
        <textarea name="description" rows={3} value={form.description ?? ""} onChange={handleChange} />
      </label>

      {errors && <p style={{ color: "#dc2626", margin: 0 }}>{errors}</p>}

      <div style={{ display: "flex", gap: "0.75rem" }}>
        <button className="btn" type="submit" disabled={loading}>
          {defaultValues ? "Update Transaction" : "Add Transaction"}
        </button>
        {defaultValues && onCancel && (
          <button className="btn secondary" type="button" onClick={onCancel}>
            Cancel
          </button>
        )}
      </div>
    </form>
  );
}
