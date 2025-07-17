import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import '../App.css';

interface KOL {
  name: string;
  tweet: string;
  score: number;
}

interface EngagementPlan {
  angles: string[];
  response: string;
}

const mockKOLs: KOL[] = [
  { name: 'KOL #1', tweet: 'This is a viral tweet from KOL #1!', score: 0.88 },
  { name: 'KOL #2', tweet: 'Another hot take from KOL #2!', score: 0.81 },
];

const PeepsOfInterestScreen: React.FC = () => {
  const [selectedIdx, setSelectedIdx] = useState<number | null>(null);
  const [engagementPlan, setEngagementPlan] = useState<EngagementPlan | null>(null);
  const navigate = useNavigate();

  const handleKOLClick = (idx: number) => {
    setSelectedIdx(idx === selectedIdx ? null : idx);
    setEngagementPlan(null);
  };

  const handleGeneratePlan = () => {
    setEngagementPlan({
      angles: ['Challenge the status quo', 'Ask a bold question'],
      response: 'Here is a sample engagement response!',
    });
  };

  return (
    <div className="peeps-screen">
      <div className="container">
        {/* Logo Placeholder */}
        <div className="image-placeholder" style={{ width: 120, height: 120, borderRadius: '50%', background: 'rgba(255,255,255,0.15)', marginBottom: 24, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          {/* TODO: Replace with actual logo image */}
          <span style={{ color: '#fff', fontSize: 48, opacity: 0.5 }}>🌟</span>
        </div>
        <h1 className="synch-title" style={{ marginBottom: 24 }}>Peeps of Interest</h1>
        <div className="kol-row" style={{ display: 'flex', justifyContent: 'center', gap: 24, marginBottom: 24 }}>
          {mockKOLs.map((kol, idx) => (
            <button
              key={idx}
              className="kol-card"
              style={{ flex: 1, background: '#1a002a', borderRadius: 18, margin: '0 8px', padding: 16, alignItems: 'center', border: '2px solid #39ff14', boxShadow: '0 4px 12px #ff00cc88', color: '#fff', cursor: 'pointer' }}
              onClick={() => handleKOLClick(idx)}
              aria-label={`KOL ${idx + 1} Tweet`}
            >
              <div className="kol-label" style={{ fontWeight: 'bold', fontSize: 16, marginBottom: 8 }}>KOL #{idx + 1} TWEET</div>
              <div className="kol-tweet" style={{ fontSize: 15, opacity: 0.8, textAlign: 'center' }}>{kol.tweet}</div>
            </button>
          ))}
        </div>
        {selectedIdx !== null && (
          <div className="reveal-area" style={{ width: '100%', background: '#0a1a2a', borderRadius: 18, padding: 20, marginTop: 16, alignItems: 'center', border: '2px solid #39ff14', color: '#fff', textAlign: 'center' }}>
            <div className="reveal-title" style={{ fontSize: 18, fontWeight: 'bold', marginBottom: 8 }}>Tweet by {mockKOLs[selectedIdx].name}</div>
            <div className="reveal-tweet" style={{ fontSize: 16, marginBottom: 12 }}>{mockKOLs[selectedIdx].tweet}</div>
            <div className="virality-label" style={{ fontSize: 15, marginTop: 8 }}>Virality Score:</div>
            <div className="virality-score" style={{ color: '#39ff14', fontSize: 20, fontWeight: 'bold', marginBottom: 12 }}>{mockKOLs[selectedIdx].score}</div>
            <button className="neon-button" onClick={handleGeneratePlan} style={{ margin: '12px 0' }}>Generate Engagement Plan</button>
            {engagementPlan && (
              <div className="plan-area" style={{ marginTop: 16, width: '100%', alignItems: 'center' }}>
                <div className="plan-title" style={{ fontWeight: 'bold', fontSize: 16, marginTop: 8 }}>Engagement Angles:</div>
                {engagementPlan.angles.map((angle, i) => (
                  <div key={i} className="plan-angle" style={{ color: '#ff00cc', fontSize: 15, margin: '2px 0' }}>{angle}</div>
                ))}
                <div className="plan-title" style={{ fontWeight: 'bold', fontSize: 16, marginTop: 8 }}>Drafted Response:</div>
                <div className="plan-response" style={{ fontSize: 15, marginTop: 4 }}>{engagementPlan.response}</div>
              </div>
            )}
          </div>
        )}
        <div style={{ display: 'flex', justifyContent: 'center', gap: 24, marginTop: 32 }}>
          <button className="neon-button" onClick={() => navigate('/your-world')}>Your World</button>
          <button className="neon-button" onClick={() => navigate('/latest-trending')}>Latest Trending</button>
        </div>
      </div>
    </div>
  );
};

export default PeepsOfInterestScreen; 