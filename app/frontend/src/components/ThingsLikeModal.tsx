import React, { useState } from 'react';

const ThingsLikeModal: React.FC = () => {
  const [interests, setInterests] = useState<string[]>([]);
  const [newInterest, setNewInterest] = useState('');

  const predefinedInterests = [
    'Technology', 'Fashion', 'Gaming', 'Music', 'Sports', 
    'Food', 'Travel', 'Art', 'Fitness', 'Movies', 
    'Books', 'Photography', 'Cars', 'Pets', 'Science'
  ];

  const handleAddInterest = () => {
    if (newInterest.trim() && !interests.includes(newInterest)) {
      setInterests([...interests, newInterest]);
      setNewInterest('');
    }
  };

  const handleRemoveInterest = (interest: string) => {
    setInterests(interests.filter(i => i !== interest));
  };

  const handlePredefinedInterest = (interest: string) => {
    if (!interests.includes(interest)) {
      setInterests([...interests, interest]);
    }
  };

  return (
    <div className="modal-body">
      <div className="modal-description">
        Configure your interests and preferences to help Synch curate personalized content and trending topics that matter to you.
      </div>
      
      <div className="modal-form">
        <div className="form-group">
          <label className="form-label">Add Custom Interest</label>
          <div className="input-group">
            <input 
              type="text" 
              className="modal-input"
              value={newInterest}
              onChange={(e) => setNewInterest(e.target.value)}
              placeholder="Enter your interest"
            />
            <button 
              className="modal-btn add-btn"
              onClick={handleAddInterest}
            >
              +
            </button>
          </div>
        </div>
        
        <div className="form-group">
          <label className="form-label">Quick Add Interests</label>
          <div className="interests-grid">
            {predefinedInterests.map((interest) => (
              <button
                key={interest}
                className={`interest-chip ${interests.includes(interest) ? 'selected' : ''}`}
                onClick={() => handlePredefinedInterest(interest)}
              >
                {interest}
              </button>
            ))}
          </div>
        </div>
        
        {interests.length > 0 && (
          <div className="form-group">
            <label className="form-label">Your Interests</label>
            <div className="selected-interests">
              {interests.map((interest) => (
                <div key={interest} className="selected-interest">
                  <span>{interest}</span>
                  <button 
                    className="remove-interest"
                    onClick={() => handleRemoveInterest(interest)}
                  >
                    ×
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}
        
        <div className="modal-actions">
          <button className="modal-btn primary">Save Changes</button>
          <button className="modal-btn secondary">Cancel</button>
        </div>
      </div>
    </div>
  );
};

export default ThingsLikeModal; 