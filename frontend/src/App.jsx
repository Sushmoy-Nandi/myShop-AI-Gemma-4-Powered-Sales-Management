import { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route, Navigate } from 'react-router-dom';
import Login from './pages/Login';
import Register from './pages/Register';
import Dashboard from './pages/Dashboard';
import Sales from './pages/Sales';
import Investments from './pages/Investments';
import Products from './pages/Products';
import Customers from './pages/Customers';
import Insights from './pages/Insights';
import Compare from './pages/Compare';
import DailyInsights from './pages/DailyInsights';
import Settings from './pages/Settings';
import StaffManagement from './pages/StaffManagement';
import AuditLog from './pages/AuditLog';
import AdminDashboard from './pages/AdminDashboard';
import Forecast from './pages/Forecast';
import MonthlyReport from './pages/MonthlyReport';
import Accounting from './pages/Accounting';
import api from './services/api';
import './index.css';

// ─── Private Route ─────────────────────────────────────────────────────────────
function PrivateRoute({ children }) {
  const token = localStorage.getItem('token');
  if (!token) return <Navigate to="/login" replace />;
  return children;
}

// ─── App ───────────────────────────────────────────────────────────────────────
export default function App() {
  const [user, setUser] = useState(null);
  const [authLoading, setAuthLoading] = useState(true);

  useEffect(() => {
    const token = localStorage.getItem('token');
    if (!token) {
      setAuthLoading(false);
      return;
    }

    api.auth
      .me()
      .then((data) => setUser(data))
      .catch(() => {
        // Token is invalid or expired — clear storage
        localStorage.removeItem('token');
        localStorage.removeItem('business_name');
      })
      .finally(() => setAuthLoading(false));
  }, []);

  function handleLogout() {
    localStorage.removeItem('token');
    localStorage.removeItem('business_name');
    setUser(null);
    window.location.href = '/login';
  }

  // Show a minimal loading screen while verifying the stored token
  if (authLoading) {
    return (
      <div
        style={{
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          height: '100vh',
          color: 'var(--text-secondary)',
          fontSize: '1rem',
        }}
      >
        <span className="loading-dots">
          Loading<span>.</span><span>.</span><span>.</span>
        </span>
      </div>
    );
  }

  const isLoggedIn = Boolean(localStorage.getItem('token'));

  return (
    <Router>
      <Routes>
        {/* Public routes */}
        <Route path="/login"    element={<Login onLogin={setUser} />} />
        <Route path="/register" element={<Register />} />

        {/* Protected routes */}
        <Route
          path="/dashboard"
          element={
            <PrivateRoute>
              <Dashboard user={user} onLogout={handleLogout} />
            </PrivateRoute>
          }
        />
        <Route
          path="/sales"
          element={
            <PrivateRoute>
              <Sales user={user} onLogout={handleLogout} />
            </PrivateRoute>
          }
        />
        <Route
          path="/investments"
          element={
            <PrivateRoute>
              <Investments user={user} onLogout={handleLogout} />
            </PrivateRoute>
          }
        />
        <Route
          path="/products"
          element={
            <PrivateRoute>
              <Products user={user} onLogout={handleLogout} />
            </PrivateRoute>
          }
        />
        <Route
          path="/customers"
          element={
            <PrivateRoute>
              <Customers user={user} onLogout={handleLogout} />
            </PrivateRoute>
          }
        />
        <Route
          path="/daily-insights"
          element={
            <PrivateRoute>
              <DailyInsights user={user} onLogout={handleLogout} />
            </PrivateRoute>
          }
        />
        <Route
          path="/insights"
          element={
            <PrivateRoute>
              <Insights user={user} onLogout={handleLogout} />
            </PrivateRoute>
          }
        />
        <Route
          path="/compare"
          element={
            <PrivateRoute>
              <Compare user={user} onLogout={handleLogout} />
            </PrivateRoute>
          }
        />
        <Route
          path="/forecast"
          element={
            <PrivateRoute>
              <Forecast user={user} onLogout={handleLogout} />
            </PrivateRoute>
          }
        />
        <Route
          path="/reports"
          element={
            <PrivateRoute>
              <MonthlyReport user={user} onLogout={handleLogout} />
            </PrivateRoute>
          }
        />
        <Route
          path="/accounting"
          element={
            <PrivateRoute>
              <Accounting user={user} onLogout={handleLogout} />
            </PrivateRoute>
          }
        />
        <Route
          path="/settings"
          element={
            <PrivateRoute>
              <Settings user={user} onLogout={handleLogout} />
            </PrivateRoute>
          }
        />
        <Route
          path="/staff"
          element={
            <PrivateRoute>
              <StaffManagement user={user} onLogout={handleLogout} />
            </PrivateRoute>
          }
        />
        <Route
          path="/audit-log"
          element={
            <PrivateRoute>
              <AuditLog user={user} onLogout={handleLogout} />
            </PrivateRoute>
          }
        />

        <Route
          path="/admin"
          element={
            <PrivateRoute>
              <AdminDashboard user={user} onLogout={handleLogout} />
            </PrivateRoute>
          }
        />

        {/* Fallback */}
        <Route
          path="*"
          element={<Navigate to={isLoggedIn ? '/dashboard' : '/login'} replace />}
        />
      </Routes>
    </Router>
  );
}
