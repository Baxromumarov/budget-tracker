import api from "./client";
import { Transaction, TransactionInput, TransactionType } from "../types";

interface Filters {
  category?: string;
  type?: TransactionType;
  start_date?: string;
  end_date?: string;
}

export async function fetchTransactions(userId: number, filters: Filters = {}): Promise<Transaction[]> {
  const response = await api.get<Transaction[]>(`/users/${userId}/transactions`, {
    params: filters,
  });
  return response.data;
}

export async function createTransaction(userId: number, input: TransactionInput): Promise<Transaction> {
  const response = await api.post<Transaction>(`/users/${userId}/transactions`, input);
  return response.data;
}

export async function updateTransaction(
  userId: number,
  transactionId: number,
  input: Partial<TransactionInput>
): Promise<Transaction> {
  const response = await api.put<Transaction>(`/users/${userId}/transactions/${transactionId}`, input);
  return response.data;
}

export async function deleteTransaction(userId: number, transactionId: number): Promise<void> {
  await api.delete(`/users/${userId}/transactions/${transactionId}`);
}
