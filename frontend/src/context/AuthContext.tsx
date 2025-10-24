import { createContext, useCallback, useContext, useEffect, useMemo, useState } from "react";
import { login as loginRequest, register as registerRequest } from "../api/auth";
import { fetchProfile } from "../api/users";
import { setAuthToken } from "../api/client";
import { AuthResponse, LoginInput, RegisterInput, User } from "../types";

interface AuthContextValue {
  user: User | null;
  token: string | null;
  loading: boolean;
  login: (input: LoginInput) => Promise<void>;
  register: (input: RegisterInput) => Promise<void>;
  logout: () => void;
  refreshProfile: () => Promise<void>;
}

const AuthContext = createContext<AuthContextValue | undefined>(undefined);
const TOKEN_STORAGE_KEY = "budget_tracker_token";

const storeSession = (auth: AuthResponse) => {
  localStorage.setItem(TOKEN_STORAGE_KEY, auth.access_token);
  setAuthToken(auth.access_token);
};

export function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = useState<User | null>(null);
  const [token, setToken] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const bootstrap = useCallback(async () => {
    const savedToken = localStorage.getItem(TOKEN_STORAGE_KEY);
    if (!savedToken) {
      setAuthToken(null);
      setLoading(false);
      return;
    }

    setAuthToken(savedToken);
    setToken(savedToken);
    try {
      const profile = await fetchProfile();
      setUser(profile);
    } catch (error) {
      console.error("Failed to restore session", error);
      localStorage.removeItem(TOKEN_STORAGE_KEY);
      setAuthToken(null);
      setToken(null);
      setUser(null);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void bootstrap();
  }, [bootstrap]);

  const handleAuthSuccess = (auth: AuthResponse) => {
    storeSession(auth);
    setToken(auth.access_token);
    setUser(auth.user);
  };

  const login = useCallback(async (input: LoginInput) => {
    const auth = await loginRequest(input);
    handleAuthSuccess(auth);
  }, []);

  const register = useCallback(async (input: RegisterInput) => {
    const auth = await registerRequest(input);
    handleAuthSuccess(auth);
  }, []);

  const logout = useCallback(() => {
    localStorage.removeItem(TOKEN_STORAGE_KEY);
    setAuthToken(null);
    setToken(null);
    setUser(null);
  }, []);

  const refreshProfile = useCallback(async () => {
    if (!token) {
      return;
    }
    const profile = await fetchProfile();
    setUser(profile);
  }, [token]);

  const value = useMemo(
    () => ({
      user,
      token,
      loading,
      login,
      register,
      logout,
      refreshProfile,
    }),
    [user, token, loading, login, register, logout, refreshProfile]
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  const context = useContext(AuthContext);
  if (!context) {
    throw new Error("useAuth must be used within an AuthProvider");
  }
  return context;
}
