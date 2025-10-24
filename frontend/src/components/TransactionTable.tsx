import { useMemo } from "react";
import { format } from "date-fns";
import { Transaction } from "../types";

interface TransactionTableProps {
  transactions: Transaction[];
  onEdit: (transaction: Transaction) => void;
  onDelete: (transaction: Transaction) => Promise<void>;
}

export default function TransactionTable({ transactions, onEdit, onDelete }: TransactionTableProps) {
  const sorted = useMemo(
    () =>
      [...transactions].sort((a, b) => {
        const dateCompare = new Date(b.date).getTime() - new Date(a.date).getTime();
        return dateCompare !== 0 ? dateCompare : b.id - a.id;
      }),
    [transactions]
  );

  if (!sorted.length) {
    return <p>No transactions recorded yet. Start by adding one above.</p>;
  }

  return (
    <div className="card">
      <h2 style={{ marginTop: 0 }}>Transactions</h2>
      <table>
        <thead>
          <tr>
            <th>Date</th>
            <th>Type</th>
            <th>Category</th>
            <th>Amount</th>
            <th>Description</th>
            <th style={{ width: "140px" }}>Actions</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((transaction) => (
            <tr key={transaction.id}>
              <td>{format(new Date(transaction.date), "PP")}</td>
              <td>{transaction.type}</td>
              <td>{transaction.category}</td>
              <td style={{ fontWeight: 600, color: transaction.type === "income" ? "#16a34a" : "#dc2626" }}>
                {transaction.type === "income" ? "+" : "-"}${transaction.amount.toFixed(2)}
              </td>
              <td>{transaction.description ?? "—"}</td>
              <td style={{ display: "flex", gap: "0.5rem" }}>
                <button className="btn secondary" onClick={() => onEdit(transaction)}>
                  Edit
                </button>
                <button className="btn secondary" onClick={() => void onDelete(transaction)}>
                  Delete
                </button>
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </div>
  );
}
