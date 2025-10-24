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
  username: string;
  email?: string | null;
  created_at: string;
}

export interface RegisterInput {
  name: string;
  username: string;
  email?: string;
  password: string;
}

export interface LoginInput {
  username: string;
  password: string;
}

export interface AuthResponse {
  access_token: string;
  token_type: string;
  user: User;
}
