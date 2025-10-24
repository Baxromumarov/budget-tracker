import { MonthlySummary } from "../types";

interface SummaryCardsProps {
  summary: MonthlySummary | null;
}

export default function SummaryCards({ summary }: SummaryCardsProps) {
  if (!summary) {
    return (
      <div className="card">
        <p>Select a month to view your summary.</p>
      </div>
    );
  }

  const statCard = (label: string, value: string, accent: string) => (
    <div style={{ background: accent, color: "#fff", borderRadius: "12px", padding: "1.5rem" }}>
      <p style={{ margin: 0, fontSize: "0.9rem", opacity: 0.8 }}>{label}</p>
      <h3 style={{ margin: "0.5rem 0 0", fontSize: "1.8rem" }}>{value}</h3>
    </div>
  );

  return (
    <div className="grid two">
      {statCard("Income", `$${summary.total_income.toFixed(2)}`, "linear-gradient(135deg, #4ade80, #16a34a)")}
      {statCard("Expenses", `$${summary.total_expenses.toFixed(2)}`, "linear-gradient(135deg, #fca5a5, #f43f5e)")}
      {statCard("Balance", `$${summary.balance.toFixed(2)}`, "linear-gradient(135deg, #60a5fa, #2563eb)")}
      {statCard("Top Category", summary.top_category ?? "—", "linear-gradient(135deg, #c084fc, #9333ea)")}
    </div>
  );
}
