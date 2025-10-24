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
    return <p className="muted">No transactions recorded yet. Start by adding one above.</p>;
  }

  return (
    <div className="table-wrapper">
      <table className="data-table">
        <thead>
          <tr>
            <th>Date</th>
            <th>Type</th>
            <th>Category</th>
            <th>Amount</th>
            <th>Description</th>
            <th style={{ width: "180px" }}>Actions</th>
          </tr>
        </thead>
        <tbody>
          {sorted.map((transaction) => (
            <tr key={transaction.id}>
              <td>{format(new Date(transaction.date), "PP")}</td>
              <td>
                <span className={`badge ${transaction.type}`}>{transaction.type}</span>
              </td>
              <td>
                <span className="badge soft">{transaction.category}</span>
              </td>
              <td className={transaction.type === "income" ? "amount income" : "amount expense"}>
                {transaction.type === "income" ? "+" : "-"}${transaction.amount.toFixed(2)}
              </td>
              <td>{transaction.description ?? "—"}</td>
              <td className="table-actions">
                <button className="btn secondary" onClick={() => onEdit(transaction)}>
                  Edit
                </button>
                <button className="btn ghost" onClick={() => void onDelete(transaction)}>
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
