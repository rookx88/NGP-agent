import React, { useState } from 'react';
import { useNavigate } from 'react-router-dom';
import SynchLogo from '../assets/images/icons/SynchLogo300x300.svg';
import SynchHeader from '../assets/images/icons/SynchHeader300x150.svg';
import YouButton from '../assets/images/icons/You_Settings_Button150x150.svg';
import PeopleFollowButton from '../assets/images/icons/People_Follow_Button.svg';
import ThingsLikeButton from '../assets/images/icons/Things_Like_Button.svg';
import AccountButton from '../assets/images/icons/Account_Button.svg';
import Modal from '../components/Modal';
import YouModal from '../components/YouModal';
import PeopleFollowModal from '../components/PeopleFollowModal';
import ThingsLikeModal from '../components/ThingsLikeModal';
import AccountModal from '../components/AccountModal';
import '../App.css';

const YourWorldScreen: React.FC = () => {
  const navigate = useNavigate();
  const [activeModal, setActiveModal] = useState<string | null>(null);

  const openModal = (modalType: string) => {
    setActiveModal(modalType);
  };

  const closeModal = () => {
    setActiveModal(null);
  };
  
  return (
    <div className="yourworld-page">
      {/* Header with Synch Title and Logo */}
      <header className="yourworld-header">
        <div className="header-content yourworld-header-flex">
          <div className="synch-header-container">
            <img src={SynchHeader} alt="Synch" className="synch-header" />
          </div>
          <div className="synch-logo-container yourworld-logo-overlap">
            <img src={SynchLogo} alt="Synch Logo" className="synch-logo" />
          </div>
        </div>
      </header>

      {/* Main Content Area */}
      <main className="yourworld-main">
        <div className="yourworld-content">
          {/* "YOUR WORLD" Button */}
          <div className="yourworld-button-container">
            <button className="yourworld-button">
              YOUR WORLD
            </button>
          </div>

          {/* 2x2 Grid for Floating Image Buttons */}
          <div className="yourworld-grid big-grid">
            {/* Top Left - YOU */}
            <div className="yourworld-item top-left">
              <img 
                src={YouButton} 
                alt="YOU" 
                className="yourworld-image big-image"
                onClick={() => openModal('you')}
              />
            </div>

            {/* Top Right - PEOPLE YOU FOLLOW */}
            <div className="yourworld-item top-right">
              <img 
                src={PeopleFollowButton} 
                alt="PEOPLE YOU FOLLOW" 
                className="yourworld-image big-image"
                onClick={() => openModal('people-follow')}
              />
            </div>

            {/* Bottom Left - THINGS YOU LIKE */}
            <div className="yourworld-item bottom-left">
              <img 
                src={ThingsLikeButton} 
                alt="THINGS YOU LIKE" 
                className="yourworld-image big-image"
                onClick={() => openModal('things-like')}
              />
            </div>

            {/* Bottom Right - ACCOUNT */}
            <div className="yourworld-item bottom-right">
              <img 
                src={AccountButton} 
                alt="ACCOUNT" 
                className="yourworld-image big-image"
                onClick={() => openModal('account')}
              />
            </div>
          </div>
        </div>
      </main>

      {/* Modals */}
      <Modal 
        isOpen={activeModal === 'you'} 
        onClose={closeModal} 
        title="YOU"
      >
        <YouModal />
      </Modal>

      <Modal 
        isOpen={activeModal === 'people-follow'} 
        onClose={closeModal} 
        title="PEOPLE YOU FOLLOW"
      >
        <PeopleFollowModal />
      </Modal>

      <Modal 
        isOpen={activeModal === 'things-like'} 
        onClose={closeModal} 
        title="THINGS YOU LIKE"
      >
        <ThingsLikeModal />
      </Modal>

      <Modal 
        isOpen={activeModal === 'account'} 
        onClose={closeModal} 
        title="ACCOUNT"
      >
        <AccountModal />
      </Modal>
    </div>
  );
};

export default YourWorldScreen; 