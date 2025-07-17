import React, { useState } from 'react';

const YouModal: React.FC = () => {
  const [profileData, setProfileData] = useState({
    username: '',
    bio: '',
    interests: '',
    privacy: 'public'
  });

  return (
    <div className="modal-body">
      <div className="modal-description">
        Personalize your Synch experience by configuring your profile settings and preferences.
      </div>
      
      <div className="modal-form">
        <div className="form-group">
          <label className="form-label">Username</label>
          <input 
            type="text" 
            className="modal-input"
            value={profileData.username}
            onChange={(e) => setProfileData({...profileData, username: e.target.value})}
            placeholder="Enter your username"
          />
        </div>
        
        <div className="form-group">
          <label className="form-label">Bio</label>
          <textarea 
            className="modal-textarea"
            value={profileData.bio}
            onChange={(e) => setProfileData({...profileData, bio: e.target.value})}
            placeholder="Tell us about yourself..."
            rows={3}
          />
        </div>
        
        <div className="form-group">
          <label className="form-label">Interests</label>
          <input 
            type="text" 
            className="modal-input"
            value={profileData.interests}
            onChange={(e) => setProfileData({...profileData, interests: e.target.value})}
            placeholder="e.g., tech, fashion, gaming"
          />
        </div>
        
        <div className="form-group">
          <label className="form-label">Privacy</label>
          <select 
            className="modal-select"
            value={profileData.privacy}
            onChange={(e) => setProfileData({...profileData, privacy: e.target.value})}
          >
            <option value="public">Public</option>
            <option value="private">Private</option>
            <option value="friends">Friends Only</option>
          </select>
        </div>
        
        <div className="modal-actions">
          <button className="modal-btn primary">Save Changes</button>
          <button className="modal-btn secondary">Cancel</button>
        </div>
      </div>
    </div>
  );
};

export default YouModal; 