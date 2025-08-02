import { useState, useEffect } from 'react'
import 'bootstrap/dist/css/bootstrap.min.css';
import './App.css'
import Header from './components/header/header.jsx'
import Homebody from './components/home-body/home-body.jsx'
import LoginRegisterTabs from './components/LoginRegisterTabs.jsx';
import UserProfile from './components/UserProfile.jsx';
import { Modal } from 'react-bootstrap';
import axios from 'axios';

const BACKEND_URL = "https://orbital25-cv.onrender.com"; // Connecting to Render-hosted backend


function App() {
  // Authentication states
  const [isLoggedIn, setIsLoggedIn] = useState(false);
  const [currentUser, setCurrentUser] = useState(null);
  const [authError, setAuthError] = useState('');
  const [showAuthModal, setShowAuthModal] = useState(false);
  const [authInitialTab, setAuthInitialTab] = useState('login');

  // Initial check for login status on component mount
  useEffect(() => {
    const checkLoginStatus = async () => {
      try {
        const response = await axios.get(`${BACKEND_URL}/user_stats`, { withCredentials: true });
        setIsLoggedIn(true);
        setCurrentUser(response.data);
        setAuthError('');
      } catch (err) {
        console.error("Error checking login status:", err);
        setIsLoggedIn(false);
        setCurrentUser(null);
        // No specific error message for initial check, just means not logged in
      }
    };
    checkLoginStatus();
  }, []);

  // Callback for successful login from LoginRegisterTabs
  const handleLoginSuccess = (userData) => {
    setIsLoggedIn(true);
    setCurrentUser(userData);
    setAuthError('');
    setShowAuthModal(false);
  };

  // Callback for successful registration (no direct user data needed here)
  const handleRegisterSuccess = () => {
    setAuthError('');
  };

  // Handle logout
  const handleLogout = async () => {
    try {
      await axios.post(`${BACKEND_URL}/logout`, {}, { withCredentials: true });
      setIsLoggedIn(false);
      setCurrentUser(null);
      setAuthError('');
    } catch (err) {
      console.error("Logout error:", err);
      setAuthError(err.response?.data?.message || 'Logout failed.');
    }
  };

  // Function to open the auth modal to a specific tab
  const openAuthModal = (tab) => {
    setAuthInitialTab(tab);
    setShowAuthModal(true);
    setAuthError('');
  };

  // Function to update user stats from Home-body
  const handleUpdateUserStats = async (differencesFound, gameWon) => {
    if (!isLoggedIn || !currentUser?.user_id) return; 

    try {
        const response = await axios.post(`${BACKEND_URL}/update_stats`, {
            differencesFound,
            gameWon
        }, { withCredentials: true });
        // Fetch updated user stats to reflect changes in UI
        const updatedUserResponse = await axios.get(`${BACKEND_URL}/user_stats`, { withCredentials: true });
        setCurrentUser(updatedUserResponse.data);
    } catch (err) {
        console.error("Failed to update user stats:", err);
    }
  };

  return (
    <>
      <Header
          isLoggedIn={isLoggedIn}
          currentUser={currentUser}
          onLoginClick={() => openAuthModal('login')}
          onRegisterClick={() => openAuthModal('register')}
          onLogout={handleLogout}
        />
      <Homebody
        isLoggedIn={isLoggedIn}
        currentUser={currentUser}
        onUpdateUserStats={handleUpdateUserStats}
        onLogout={handleLogout}
      />

      {/* Authentication Modal */}
      <Modal show={showAuthModal} onHide={() => setShowAuthModal(false)} centered>
        <Modal.Header closeButton>
          <Modal.Title>{authInitialTab === 'login' ? 'Login' : 'Register'}</Modal.Title>
        </Modal.Header>
        <Modal.Body>
          <LoginRegisterTabs
            onLoginSuccess={handleLoginSuccess}
            onRegisterSuccess={handleRegisterSuccess}
            authError={authError}
            setAuthError={setAuthError}
            initialTab={authInitialTab}
            onTabChange={setAuthInitialTab}
          />
        </Modal.Body>
      </Modal>
    </>
  );
}

export default App;