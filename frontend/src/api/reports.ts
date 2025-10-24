import api from "./client";
import { MonthlySummary } from "../types";

export async function fetchMonthlySummary(month: number, year: number): Promise<MonthlySummary> {
  const response = await api.get<MonthlySummary>(`/me/summary`, {
    params: { month, year },
  });
  return response.data;
}

export async function downloadReport(month: number, year: number, format: "csv" | "json"): Promise<Blob> {
  const response = await api.get(`/me/reports/${format}`, {
    params: { month, year },
    responseType: "blob",
  });
  return response.data;
}
