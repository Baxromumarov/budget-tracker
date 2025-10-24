import api from "./client";
import { Transaction, TransactionInput, TransactionType } from "../types";

interface Filters {
  category?: string;
  type?: TransactionType;
  start_date?: string;
  end_date?: string;
}

export async function fetchTransactions(filters: Filters = {}): Promise<Transaction[]> {
  const response = await api.get<Transaction[]>(`/me/transactions`, {
    params: filters,
  });
  return response.data;
}

export async function createTransaction(input: TransactionInput): Promise<Transaction> {
  const response = await api.post<Transaction>(`/me/transactions`, input);
  return response.data;
}

export async function updateTransaction(
  transactionId: number,
  input: Partial<TransactionInput>
): Promise<Transaction> {
  const response = await api.put<Transaction>(`/me/transactions/${transactionId}`, input);
  return response.data;
}

export async function deleteTransaction(transactionId: number): Promise<void> {
  await api.delete(`/me/transactions/${transactionId}`);
}
