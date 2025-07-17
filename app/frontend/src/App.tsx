// Main entry point for Synch React Web app
import React from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import LandingScreen from './screens/LandingScreen';
import LoginScreen from './screens/LoginScreen';
import RegisterScreen from './screens/RegisterScreen';
import AboutUsScreen from './screens/AboutUsScreen';
import YourWorldScreen from './screens/YourWorldScreen';
import LatestTrendingScreen from './screens/LatestTrendingScreen';
import PeepsOfInterestScreen from './screens/PeepsOfInterestScreen';
import './App.css';

const App: React.FC = () => {
  return (
    <Router>
      <Routes>
        <Route path="/" element={<LandingScreen />} />
        <Route path="/login" element={<LoginScreen />} />
        <Route path="/register" element={<RegisterScreen />} />
        <Route path="/about" element={<AboutUsScreen />} />
        <Route path="/your-world" element={<YourWorldScreen />} />
        <Route path="/latest-trending" element={<LatestTrendingScreen />} />
        <Route path="/peeps-of-interest" element={<PeepsOfInterestScreen />} />
      </Routes>
    </Router>
  );
};

export default App; 