import api from "./client";
import { AuthResponse, LoginInput, RegisterInput } from "../types";

export async function register(input: RegisterInput): Promise<AuthResponse> {
  const response = await api.post<AuthResponse>("/auth/register", input);
  return response.data;
}

export async function login(input: LoginInput): Promise<AuthResponse> {
  const response = await api.post<AuthResponse>("/auth/login", input);
  return response.data;
}
