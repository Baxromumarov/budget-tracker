export type TransactionType = "income" | "expense";

export interface Transaction {
  id: number;
  user_id: number;
  amount: number;
  date: string;
  category: string;
  type: TransactionType;
  description?: string | null;
}

export interface TransactionInput {
  amount: number;
  date: string;
  category: string;
  type: TransactionType;
  description?: string;
}

export interface MonthlySummary {
  month: number;
  year: number;
  total_income: number;
  total_expenses: number;
  balance: number;
  top_category: string | null;
}

export interface User {
  id: number;
  name: string;
  email: string;
}
