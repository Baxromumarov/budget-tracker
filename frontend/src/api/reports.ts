import api from "./client";
import { MonthlySummary } from "../types";

export async function fetchMonthlySummary(userId: number, month: number, year: number): Promise<MonthlySummary> {
  const response = await api.get<MonthlySummary>(`/users/${userId}/summary`, {
    params: { month, year },
  });
  return response.data;
}

export async function downloadReport(
  userId: number,
  month: number,
  year: number,
  format: "csv" | "json"
): Promise<Blob> {
  const response = await api.get(`/users/${userId}/reports/${format}`, {
    params: { month, year },
    responseType: "blob",
  });
  return response.data;
}
