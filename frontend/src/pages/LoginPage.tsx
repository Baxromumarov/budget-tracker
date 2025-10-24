import { FormEvent, useState } from "react";
import { Link, useLocation, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export default function LoginPage() {
  const { login } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [showPassword, setShowPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setLoading(true);
    setError(null);
    try {
      await login({ username, password });
      const redirectTo = (location.state as { from?: string } | null)?.from ?? "/";
      navigate(redirectTo, { replace: true });
    } catch (err) {
      console.error(err);
      setError("Invalid username or password.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-wrapper">
      <section className="auth-hero">
        <h1>Welcome back</h1>
        <p>
          Track every dollar with clarity. Sign in to access your personalised budget dashboard,
          insights, and reports.
        </p>
      </section>

      <section className="auth-card">
        <h2>Login</h2>
        <form onSubmit={handleSubmit} className="form-grid">
          <label>
            Username
            <input value={username} onChange={(event) => setUsername(event.target.value)} required />
          </label>
          <label>
            Password
            <div className="input-with-toggle">
              <input
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(event) => setPassword(event.target.value)}
                required
              />
              <button
                type="button"
                className="toggle-visibility"
                onClick={() => setShowPassword((prev) => !prev)}
                aria-label={showPassword ? "Hide password" : "Show password"}
              >
                {showPassword ? "Hide" : "Show"}
              </button>
            </div>
          </label>
          {error && <p className="form-error">{error}</p>}
          <button className="btn" type="submit" disabled={loading}>
            {loading ? "Signing in..." : "Sign in"}
          </button>
        </form>
        <p className="muted">
          Don't have an account? <Link to="/register">Create one</Link>
        </p>
      </section>
    </div>
  );
}
