import React, { useState } from 'react';

const PeopleFollowModal: React.FC = () => {
  const [followedAccounts, setFollowedAccounts] = useState(['', '', '']);
  const [newAccount, setNewAccount] = useState('');

  const handleAddAccount = (index: number) => {
    if (newAccount.trim()) {
      const updatedAccounts = [...followedAccounts];
      updatedAccounts[index] = newAccount;
      setFollowedAccounts(updatedAccounts);
      setNewAccount('');
    }
  };

  const handleRemoveAccount = (index: number) => {
    const updatedAccounts = [...followedAccounts];
    updatedAccounts[index] = '';
    setFollowedAccounts(updatedAccounts);
  };

  return (
    <div className="modal-body">
      <div className="modal-description">
        Adding people you follow lets Synch track virality from recent posts. Synch will scan these accounts and guide you to your best angle at the 'Peeps' page.
      </div>
      
      <div className="modal-form">
        <div className="form-group">
          <label className="form-label">Add New Account</label>
          <div className="input-group">
            <input 
              type="text" 
              className="modal-input"
              value={newAccount}
              onChange={(e) => setNewAccount(e.target.value)}
              placeholder="Enter username or handle"
            />
            <button 
              className="modal-btn add-btn"
              onClick={() => {
                const emptyIndex = followedAccounts.findIndex(account => !account);
                if (emptyIndex !== -1) {
                  handleAddAccount(emptyIndex);
                }
              }}
            >
              +
            </button>
          </div>
        </div>
        
        <div className="accounts-list">
          {followedAccounts.map((account, index) => (
            <div key={index} className="account-item">
              <span className="account-number">{index + 1}</span>
              <div className="account-input-line"></div>
              <button 
                className={`account-btn ${account ? 'remove' : 'add'}`}
                onClick={() => account ? handleRemoveAccount(index) : handleAddAccount(index)}
              >
                {account ? '−' : '+'}
              </button>
              {account && <span className="account-name">{account}</span>}
            </div>
          ))}
        </div>
        
        <div className="modal-actions">
          <button className="modal-btn primary">Save Changes</button>
          <button className="modal-btn secondary">Cancel</button>
        </div>
      </div>
    </div>
  );
};

export default PeopleFollowModal; 