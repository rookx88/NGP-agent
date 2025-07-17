import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import '../App.css';

interface Trend {
  name: string;
  score: number;
}

const LatestTrendingScreen: React.FC = () => {
  const [loading, setLoading] = useState(false);
  const [trends, setTrends] = useState<Trend[]>([]);
  const navigate = useNavigate();

  const handleDiscoverTrends = () => {
    setLoading(true);
    setTimeout(() => {
      setTrends([
        { name: 'AI Revolution', score: 0.92 },
        { name: 'Neon Sports', score: 0.85 },
        { name: 'Tech Takeover', score: 0.81 },
      ]);
      setLoading(false);
    }, 1500);
  };

  return (
    <div className="latesttrending-screen">
      <div className="container">
        {/* Logo Placeholder */}
        <div className="image-placeholder" style={{ width: 120, height: 120, borderRadius: '50%', background: 'rgba(255,255,255,0.15)', marginBottom: 24, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          {/* TODO: Replace with actual logo image */}
          <span style={{ color: '#fff', fontSize: 48, opacity: 0.5 }}>🌟</span>
        </div>
        <h1 className="synch-title" style={{ marginBottom: 24 }}>Latest Trending</h1>
        <button
          className="circle-button"
          onClick={handleDiscoverTrends}
          disabled={loading}
          style={{ marginBottom: 32 }}
        >
          {loading ? 'Loading...' : 'Discover Global Trends'}
        </button>
        <div className="output-label" style={{ color: '#fff', fontSize: 16, marginBottom: 8, fontWeight: 'bold', letterSpacing: 1 }}>
          Assessor.py outputs
        </div>
        <div className="output-area" style={{ width: '100%', minHeight: 160, background: '#0a1a2a', borderRadius: 18, padding: 16, alignItems: 'center', justifyContent: 'center', border: '2px solid #39ff14' }}>
          {loading ? (
            <div style={{ color: '#39ff14', fontSize: 18 }}>Loading...</div>
          ) : trends.length === 0 ? (
            <div style={{ color: '#fff', opacity: 0.7, fontSize: 16, textAlign: 'center' }}>No trends yet. Tap the button above!</div>
          ) : (
            trends.map((trend, idx) => (
              <div key={idx} className="trend-card" style={{ background: '#2a003f', borderRadius: 12, padding: 12, margin: '6px 0', width: '100%', alignItems: 'center', border: '1px solid #ff00cc', textAlign: 'center' }}>
                <div style={{ color: '#fff', fontSize: 18, fontWeight: 'bold' }}>{trend.name}</div>
                <div style={{ color: '#39ff14', fontSize: 14, marginTop: 4 }}>Score: {trend.score}</div>
              </div>
            ))
          )}
        </div>
        <div style={{ display: 'flex', justifyContent: 'center', gap: 24, marginTop: 32 }}>
          <button className="neon-button" onClick={() => navigate('/your-world')}>Your World</button>
          <button className="neon-button" onClick={() => navigate('/peeps-of-interest')}>Peeps of Interest</button>
        </div>
      </div>
    </div>
  );
};

export default LatestTrendingScreen; 