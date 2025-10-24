import api from "./client";
import { User } from "../types";

export async function fetchProfile(): Promise<User> {
  const response = await api.get<User>("/users/me");
  return response.data;
}
