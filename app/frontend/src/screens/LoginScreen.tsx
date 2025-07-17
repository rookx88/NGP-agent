import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import '../App.css';

const LoginScreen: React.FC = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState<string | null>(null);
  const navigate = useNavigate();

  const handleLogin = (e: React.FormEvent) => {
    e.preventDefault();
    // TODO: Add real authentication logic
    if (!username || !password) {
      setError('Please enter both username and password.');
      return;
    }
    setError(null);
    navigate('/your-world');
  };

  return (
    <div className="login-screen">
      <div className="container">
        {/* User Silhouette Placeholder */}
        <div className="image-placeholder" style={{ width: 120, height: 120, borderRadius: '50%', background: 'rgba(255,255,255,0.15)', marginBottom: 24, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          {/* TODO: Replace with actual user silhouette image */}
          <span style={{ color: '#fff', fontSize: 48, opacity: 0.5 }}>👤</span>
        </div>
        <h1 className="synch-title" style={{ marginBottom: 24 }}>LOGIN</h1>
        <form className="login-form" onSubmit={handleLogin}>
          <input
            className="neon-input"
            type="text"
            placeholder="Username or Email"
            value={username}
            onChange={e => setUsername(e.target.value)}
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
          onClick={() => navigate('/register')}
        >
          Register now
        </button>
      </div>
    </div>
  );
};

export default LoginScreen; 