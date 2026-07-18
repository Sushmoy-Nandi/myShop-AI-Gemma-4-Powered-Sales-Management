import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Store } from 'lucide-react';
import api from '../services/api';

export default function Register() {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [businessName, setBusinessName] = useState('');
  const [role, setRole] = useState('owner');
  const [error, setError] = useState('');
  const [loading, setLoading] = useState(false);
  const navigate = useNavigate();

  async function handleRegister(e) {
    e.preventDefault();
    setError('');
    setLoading(true);
    try {
      await api.auth.register(email, password, businessName, role);
      navigate('/login');
    } catch (err) {
      setError(err.message);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="auth-container">
      <div className="auth-card">
        {/* Brand icon */}
        <div style={{ display: 'flex', justifyContent: 'center', marginBottom: '1.5rem' }}>
          <div
            style={{
              background: 'var(--primary-gradient)',
              borderRadius: 16,
              padding: '1rem',
              display: 'inline-flex',
              boxShadow: 'var(--accent-glow)',
            }}
          >
            <Store size={32} color="white" />
          </div>
        </div>

        <h2>Create Account</h2>
        <p style={{ color: 'var(--text-secondary)', marginTop: '0.4rem' }}>
          Start automating your business insights
        </p>

        {error && (
          <div className="error-message" style={{ marginTop: '1.25rem' }}>{error}</div>
        )}

        <form onSubmit={handleRegister} className="auth-form">
          <input
            type="text"
            placeholder="Business Name"
            value={businessName}
            onChange={(e) => setBusinessName(e.target.value)}
            required
            autoComplete="organization"
          />
          <input
            type="email"
            placeholder="Email Address"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            required
            autoComplete="email"
          />
          <input
            type="password"
            placeholder="Password"
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            required
            autoComplete="new-password"
          />

          {/* Role select — styled to match auth-form inputs */}
          <select
            value={role}
            onChange={(e) => setRole(e.target.value)}
            style={{
              padding: '1rem 1.25rem',
              borderRadius: 12,
              border: '1px solid rgba(255,255,255,0.1)',
              background: 'rgba(0,0,0,0.2)',
              color: 'white',
              fontFamily: 'inherit',
              fontSize: '1rem',
              cursor: 'pointer',
              transition: 'all 0.3s ease',
              appearance: 'none',
              backgroundImage: `url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='16' height='16' viewBox='0 0 24 24' fill='none' stroke='%2394a3b8' stroke-width='2'%3E%3Cpath d='M6 9l6 6 6-6'/%3E%3C/svg%3E")`,
              backgroundRepeat: 'no-repeat',
              backgroundPosition: 'right 1rem center',
              paddingRight: '2.5rem',
            }}
          >
            <option value="owner" style={{ background: '#0f172a' }}>Owner</option>
            <option value="admin" style={{ background: '#0f172a' }}>Admin</option>
            <option value="staff" style={{ background: '#0f172a' }}>Staff</option>
          </select>

          <button type="submit" className="auth-btn" disabled={loading}>
            {loading ? 'Creating account…' : 'Register'}
          </button>
        </form>

        <p className="auth-footer">
          Already have an account? <Link to="/login">Login here</Link>
        </p>
      </div>
    </div>
  );
}
