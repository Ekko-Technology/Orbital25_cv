import React, { useState, useEffect } from 'react';
import { Form, Button, Tabs, Tab, Alert, Spinner } from 'react-bootstrap';
import { User as UserIcon, LogIn } from 'lucide-react';
import axios from 'axios';

const BACKEND_URL = "https://orbital-2025-backend.onrender.com";

function LoginRegisterTabs({ onLoginSuccess, onRegisterSuccess, authError, setAuthError, initialTab, onTabChange }) {
    const [activeTab, setActiveTab] = useState('login');
    const [regUsername, setRegUsername] = useState('');
    const [regPassword, setRegPassword] = useState('');
    const [logUsername, setLogUsername] = useState('');
    const [logPassword, setLogPassword] = useState('');
    const [loading, setLoading] = useState(false);
    const [message, setMessage] = useState('');

    useEffect(() => {
        setActiveTab(initialTab || 'login');
        setMessage('');
        setAuthError('');
    }, [initialTab, setAuthError]);

    const handleTabSelect = (k) => {
        setActiveTab(k);
        if (onTabChange) {
            onTabChange(k);
        }
        setAuthError('');
        setMessage('');
    }

    const handleRegister = async (event) => {
        event.preventDefault();
        setAuthError('');
        setMessage('');
        setLoading(true);

        if (!regUsername || !regPassword) {
            setAuthError('Username and password are required for registration.');
            setLoading(false);
            return;
        }

        try {
            const response = await axios.post(`${BACKEND_URL}/register`, { 
                username: regUsername, 
                password: regPassword
            }, { withCredentials: true }); // Send cookies
            
            setMessage(response.data.message + " You can now log in.");
            setActiveTab('login'); // Switch to login tab after successful registration
            if (onTabChange) {
                onTabChange('login');
            }
            setRegUsername('');
            setRegPassword('');
            if (onRegisterSuccess) onRegisterSuccess(); // Callback to parent
        } catch (err) {
            console.error("Registration error:", err);
            setAuthError(err.response?.data?.error || 'Registration error. Please try again.');
        } finally {
            setLoading(false);
        }
    };

    const handleLogin = async (event) => {
        event.preventDefault();
        setAuthError('');
        setMessage('');
        setLoading(true);

        if (!logUsername || !logPassword) {
            setAuthError('Username and password are required for login.');
            setLoading(false);
            return;
        }

        try {
            const response = await axios.post(`${BACKEND_URL}/login`, { 
                username: logUsername, 
                password: logPassword  
            }, { withCredentials: true }); // Send cookies

            // onLoginSuccess will be called with user data including id, username, and stats
            if (onLoginSuccess) onLoginSuccess(response.data); 
            setMessage(response.data.message);
            setLogUsername('');
            setLogPassword('');
        } catch (err) {
            console.error("Login error:", err);
            setAuthError(err.response?.data?.error || "Login failed. Please check your credentials.");
        } finally {
            setLoading(false);
        }
    };

    return (
        <>
            {message && <Alert variant="info" className="mt-3">{message}</Alert>}
            {authError && <Alert variant="danger" className="mt-3">{authError}</Alert>}

            <Tabs activeKey={activeTab} onSelect={handleTabSelect} className="mb-3">
                <Tab eventKey="login" title={<span><LogIn size={16} className="me-2" />Login</span>}>
                    <Form onSubmit={handleLogin} className="mt-3">
                        <Form.Group className="mb-3" controlId="loginUsername">
                            <Form.Label>Username</Form.Label>
                            <Form.Control type="text" placeholder="Enter username" required value={logUsername}
                                onChange={(e) => setLogUsername(e.target.value)} disabled={loading} />
                        </Form.Group>
                        <Form.Group className="mb-3" controlId="loginPassword">
                            <Form.Label>Password</Form.Label>
                            <Form.Control type="password" placeholder="Password" required value={logPassword}
                                onChange={(e) => setLogPassword(e.target.value)} disabled={loading} />
                        </Form.Group>
                        <Button variant="primary" type="submit" disabled={loading}>
                            {loading ? <Spinner animation="border" size="sm" /> : 'Login'}
                        </Button>
                    </Form>
                </Tab>
                <Tab eventKey="register" title={<span><UserIcon size={16} className="me-2" />Register</span>}>
                    <Form onSubmit={handleRegister} className="mt-3">
                        <Form.Group className="mb-3" controlId="registerUsername">
                            <Form.Label>Username</Form.Label>
                            <Form.Control type="text" placeholder="Choose a username" required value={regUsername} 
                                onChange={(e) => setRegUsername(e.target.value)} disabled={loading} />
                        </Form.Group>
                        <Form.Group className="mb-3" controlId="registerPassword">
                            <Form.Label>Password</Form.Label>
                            <Form.Control type="password" placeholder="Password" required value={regPassword}
                                onChange={(e) => setRegPassword(e.target.value)} disabled={loading} />
                            <Form.Text className="text-muted">
                                Your password must be at least 6 characters long.
                            </Form.Text>
                        </Form.Group>
                        <Button variant="primary" type="submit" disabled={loading}>
                            {loading ? <Spinner animation="border" size="sm" /> : 'Register'}
                        </Button>
                    </Form>
                </Tab>
            </Tabs>
        </>
    );
}

export default LoginRegisterTabs;