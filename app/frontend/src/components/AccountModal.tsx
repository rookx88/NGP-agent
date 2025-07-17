import React, { useState } from 'react';

const AccountModal: React.FC = () => {
  const [accountSettings, setAccountSettings] = useState({
    email: '',
    notifications: true,
    theme: 'dark',
    language: 'en',
    autoSync: true
  });

  return (
    <div className="modal-body">
      <div className="modal-description">
        Manage your account settings, preferences, and Synch configuration options.
      </div>
      
      <div className="modal-form">
        <div className="form-group">
          <label className="form-label">Email Address</label>
          <input 
            type="email" 
            className="modal-input"
            value={accountSettings.email}
            onChange={(e) => setAccountSettings({...accountSettings, email: e.target.value})}
            placeholder="your.email@example.com"
          />
        </div>
        
        <div className="form-group">
          <label className="form-label">Theme</label>
          <select 
            className="modal-select"
            value={accountSettings.theme}
            onChange={(e) => setAccountSettings({...accountSettings, theme: e.target.value})}
          >
            <option value="dark">Dark Mode</option>
            <option value="light">Light Mode</option>
            <option value="auto">Auto</option>
          </select>
        </div>
        
        <div className="form-group">
          <label className="form-label">Language</label>
          <select 
            className="modal-select"
            value={accountSettings.language}
            onChange={(e) => setAccountSettings({...accountSettings, language: e.target.value})}
          >
            <option value="en">English</option>
            <option value="es">Español</option>
            <option value="fr">Français</option>
            <option value="de">Deutsch</option>
          </select>
        </div>
        
        <div className="form-group">
          <div className="checkbox-group">
            <label className="checkbox-label">
              <input 
                type="checkbox" 
                className="modal-checkbox"
                checked={accountSettings.notifications}
                onChange={(e) => setAccountSettings({...accountSettings, notifications: e.target.checked})}
              />
              <span className="checkmark"></span>
              Enable Notifications
            </label>
          </div>
        </div>
        
        <div className="form-group">
          <div className="checkbox-group">
            <label className="checkbox-label">
              <input 
                type="checkbox" 
                className="modal-checkbox"
                checked={accountSettings.autoSync}
                onChange={(e) => setAccountSettings({...accountSettings, autoSync: e.target.checked})}
              />
              <span className="checkmark"></span>
              Auto-sync Data
            </label>
          </div>
        </div>
        
        <div className="modal-actions">
          <button className="modal-btn primary">Save Changes</button>
          <button className="modal-btn secondary">Cancel</button>
        </div>
        
        <div className="modal-footer">
          <button className="modal-btn danger">Delete Account</button>
          <button className="modal-btn warning">Export Data</button>
        </div>
      </div>
    </div>
  );
};

export default AccountModal; 