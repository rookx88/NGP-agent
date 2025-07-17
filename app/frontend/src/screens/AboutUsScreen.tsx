import React from 'react';
import { useNavigate } from 'react-router-dom';
import '../App.css';

const AboutUsScreen: React.FC = () => {
  const navigate = useNavigate();
  return (
    <div className="about-screen">
      <div className="container">
        {/* Logo Placeholder */}
        <div className="image-placeholder" style={{ width: 120, height: 120, borderRadius: '50%', background: 'rgba(255,255,255,0.15)', marginBottom: 24, display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          {/* TODO: Replace with actual logo image */}
          <span style={{ color: '#fff', fontSize: 48, opacity: 0.5 }}>🌟</span>
        </div>
        <h1 className="synch-title" style={{ marginBottom: 24 }}>About Synch</h1>
        <p className="about-description" style={{ color: '#fff', fontSize: 18, textAlign: 'center', marginBottom: 40, lineHeight: '28px' }}>
          Synch is your gateway to viral trends and authentic engagement. Discover what's trending, analyze influencers, and stay ahead of the curve with a bold, modern social experience.
        </p>
        <button className="neon-button" onClick={() => navigate(-1)}>Back</button>
      </div>
    </div>
  );
};

export default AboutUsScreen; 