import React from 'react';
import { useNavigate } from 'react-router-dom';
import SynchHeader from '../assets/images/icons/SynchHeader300x150.svg';
import SynchLogo from '../assets/images/icons/SynchLogo300x300.svg';
import '../App.css';

const LandingScreen: React.FC = () => {
  const navigate = useNavigate();
  
  return (
    <div className="landing-page">
      {/* Black Header */}
      <header className="landing-header">
        <div className="header-content">
          {/* Header content without logo */}
        </div>
      </header>

      {/* Main Content Area with Gradient Background */}
      <main className="landing-main">
        <div className="landing-content">
          {/* Central Visual Area */}
          <div className="hero-section">
            {/* Synch Logo and Header Container */}
            <div className="synch-visual-container">
              {/* Synch Logo - Bigger and positioned behind */}
              <div className="synch-logo-container">
                <img src={SynchLogo} alt="Synch Logo" className="synch-logo" />
              </div>
              
              {/* Synch Header - Positioned to overlap the logo */}
              <div className="synch-header-container">
                <img src={SynchHeader} alt="Synch" className="synch-header" />
              </div>
            </div>
            
            {/* LOGIN! Button - Positioned at a slant */}
            <div className="login-button-container">
              <button 
                className="login-button-primary"
                onClick={() => navigate('/login')}
              >
                <span className="login-bracket">[</span>
                <span className="login-text">LOGIN!</span>
                <span className="login-bracket">]</span>
              </button>
            </div>
          </div>

          {/* Lower Section */}
          <div className="landing-lower">
            {/* Slogan */}
            <h2 className="landing-slogan">Go Viral. Stay Authentic.</h2>
            
            {/* Description */}
            <p className="landing-description">
              Stop Chasing Trends. Start Setting Them. Synch's AI helps you discover what's next and create authentic viral content.
            </p>
            
            {/* Call-to-Action Buttons */}
            <div className="cta-buttons">
              <button 
                className="register-button"
                onClick={() => navigate('/register')}
              >
                REGISTER
              </button>
              <button 
                className="read-more-button"
                onClick={() => navigate('/about')}
              >
                READ MORE
              </button>
              <button 
                className="register-button"
                onClick={() => navigate('/your-world')}
                style={{ marginTop: '1rem' }}
              >
                VIEW YOUR WORLD
              </button>
            </div>
          </div>
        </div>
      </main>

      {/* Black Footer */}
      <footer className="landing-footer">
        <div className="footer-content">
          <p>&copy; 2024 Synch. All rights reserved.</p>
        </div>
      </footer>
    </div>
  );
};

export default LandingScreen; 