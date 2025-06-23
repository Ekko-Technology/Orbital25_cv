import React from 'react';
import './header.css';
import {Container, Nav, Navbar, NavDropdown} from 'react-bootstrap';

const Header = () => {
    return (
    <Navbar expand="lg" className="bg-body-tertiary">
      <Container>
        <Navbar.Brand href="#home">spotthedifference</Navbar.Brand>
        <Navbar.Toggle aria-controls="basic-navbar-nav" />
        <Navbar.Collapse id="basic-navbar-nav">
          <Nav className="ms-auto">
            <Nav.Link href="#home">How To Play</Nav.Link>
            {/* <Nav.Link href="#link">Link</Nav.Link> */}
            <NavDropdown title="Account" id="basic-nav-dropdown">
              <NavDropdown.Item href="#home">Login</NavDropdown.Item>
              <NavDropdown.Item href="#home">Register</NavDropdown.Item>
            </NavDropdown>
          </Nav>
        </Navbar.Collapse>
      </Container>
    </Navbar>
    );
    }

export default Header
