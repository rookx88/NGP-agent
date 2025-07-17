import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import '../App.css';

const RegisterScreen: React.FC = () => {
  const [username, setUsername] = useState('');
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  const handleRegister = (e: React.FormEvent) => {
    e.preventDefault();
    // TODO: Add real registration logic
    if (!username || !email || !password) {
      setError('Please fill in all fields.');
      return;
    }
    setError(null);
    navigate('/login');
  };

  return (
    <div className="login-screen"> {/* Reuse login-screen styles for consistency */}
      <div className="container">
        {/* User Silhouette Placeholder */}
        <div className="image-placeholder" style={{ width: 120, height: 120, borderRadius: '50%', background: 'rgba(255,255,255,0.15)', marginBottom: 24, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          {/* TODO: Replace with actual user silhouette image */}
          <span style={{ color: '#fff', fontSize: 48, opacity: 0.5 }}>👤</span>
        </div>
        <h1 className="synch-title" style={{ marginBottom: 24 }}>REGISTER NOW!</h1>
        <form className="login-form" onSubmit={handleRegister}>
          <input
            className="neon-input"
            type="text"
            placeholder="Username"
            value={username}
            onChange={e => setUsername(e.target.value)}
          />
          <input
            className="neon-input"
            type="email"
            placeholder="Email"
            value={email}
            onChange={e => setEmail(e.target.value)}
          />
          <input
            className="neon-input"
            type="password"
            placeholder="Password"
            value={password}
            onChange={e => setPassword(e.target.value)}
          />
          {error && <div className="error-message">{error}</div>}
          <button type="submit" className="neon-button">GO!</button>
        </form>
        <button
          className="learn-more"
          style={{ marginTop: 24 }}
          onClick={() => navigate('/login')}
        >
          Already have an account? Login
        </button>
      </div>
    </div>
  );
};

export default RegisterScreen; 