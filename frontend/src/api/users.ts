import api from "./client";
import { User } from "../types";

export async function fetchUsers(): Promise<User[]> {
  const response = await api.get<User[]>("/users");
  return response.data;
}

export async function createUser(input: { name: string; email: string }): Promise<User> {
  const response = await api.post<User>("/users", input);
  return response.data;
}
