import { FormEvent, useState } from "react";
import { Link, useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext";

export default function RegisterPage() {
  const { register } = useAuth();
  const navigate = useNavigate();
  const [form, setForm] = useState({
    name: "",
    username: "",
    email: "",
    password: "",
    confirmPassword: "",
  });
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (event: React.ChangeEvent<HTMLInputElement>) => {
    const { name, value } = event.target;
    setForm((prev) => ({ ...prev, [name]: value }));
  };

  const handleSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    if (form.password !== form.confirmPassword) {
      setError("Passwords do not match.");
      return;
    }
    setError(null);
    setLoading(true);
    try {
      await register({
        name: form.name.trim(),
        username: form.username.trim(),
        email: form.email.trim() || undefined,
        password: form.password,
      });
      navigate("/", { replace: true });
    } catch (err) {
      console.error(err);
      setError("Unable to register. Username or email may already be in use.");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="auth-wrapper">
      <section className="auth-hero">
        <h1>Create your account</h1>
        <p>
          Plan smarter budgets with tailored insights, monthly summaries, and exportable reports.
          Join now to get started.
        </p>
      </section>

      <section className="auth-card">
        <h2>Register</h2>
        <form onSubmit={handleSubmit} className="form-grid">
          <label>
            Name
            <input name="name" value={form.name} onChange={handleChange} required />
          </label>
          <label>
            Username
            <input name="username" value={form.username} onChange={handleChange} required />
          </label>
          <label>
            Email (optional)
            <input type="email" name="email" value={form.email} onChange={handleChange} />
          </label>
          <label>
            Password
            <div className="input-with-toggle">
              <input
                type={showPassword ? "text" : "password"}
                name="password"
                value={form.password}
                onChange={handleChange}
                minLength={8}
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
          <label>
            Confirm Password
            <div className="input-with-toggle">
              <input
                type={showConfirmPassword ? "text" : "password"}
                name="confirmPassword"
                value={form.confirmPassword}
                onChange={handleChange}
                minLength={8}
                required
              />
              <button
                type="button"
                className="toggle-visibility"
                onClick={() => setShowConfirmPassword((prev) => !prev)}
                aria-label={showConfirmPassword ? "Hide confirm password" : "Show confirm password"}
              >
                {showConfirmPassword ? "Hide" : "Show"}
              </button>
            </div>
          </label>
          {error && <p className="form-error">{error}</p>}
          <button className="btn" type="submit" disabled={loading}>
            {loading ? "Creating account..." : "Create account"}
          </button>
        </form>
        <p className="muted">
          Already have an account? <Link to="/login">Sign in</Link>
        </p>
      </section>
    </div>
  );
}
