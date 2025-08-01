import React, { useState } from 'react';
import './header.css';
import {Container, Nav, Navbar, NavDropdown} from 'react-bootstrap';
import { LogIn, UserPlus, User, LogOut, Award } from 'lucide-react';
import HowToPlayModal from '../HowToPlayModal';

const Header = ({ isLoggedIn, currentUser, onLoginClick, onRegisterClick, onLogout }) => {
  const [showHowToPlayModal, setShowHowToPlayModal] = useState(false);

  const handleShowHowToPlay = () => setShowHowToPlayModal(true);
  const handleCloseHowToPlay = () => setShowHowToPlayModal(false);

    return (
      <>
        <Navbar expand="lg" className="bg-body-tertiary">
          <Container>
            <Navbar.Brand href="/">spotthedifference</Navbar.Brand>
            <Navbar.Toggle aria-controls="basic-navbar-nav" />
            <Navbar.Collapse id="basic-navbar-nav">
              <Nav className="ms-auto">
                <Nav.Link onClick={handleShowHowToPlay}>How To Play</Nav.Link>
                <NavDropdown title="Account" id="basic-nav-dropdown" align="end">
                  {!isLoggedIn ? (
                    <>
                      <NavDropdown.Item onClick={onLoginClick}>
                          <LogIn size={16} className="me-2" />Login
                      </NavDropdown.Item>
                      <NavDropdown.Item onClick={onRegisterClick}>
                          <UserPlus size={16} className="me-2" />Register
                      </NavDropdown.Item>
                    </>
                  ) : (
                    <>
                      <NavDropdown.Item disabled>
                          <User size={16} className="me-2" />Logged in as: {currentUser?.username}
                      </NavDropdown.Item>
                      <NavDropdown.Divider />
                      <NavDropdown.Item onClick={onLogout}>
                          <LogOut size={16} className="me-2" />Logout
                      </NavDropdown.Item>
                    </>
                  )}
                </NavDropdown>
              </Nav>
            </Navbar.Collapse>
          </Container>
        </Navbar>
        <HowToPlayModal show={showHowToPlayModal} onClose={handleCloseHowToPlay} />
      </>
    );
    }

export default Header;