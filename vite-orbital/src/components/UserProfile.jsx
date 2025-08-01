import React from 'react';
import { Button, Card } from 'react-bootstrap';
import { LogOut, Award, History } from 'lucide-react';

function UserProfile({ currentUser, onLogout, onShowGameHistory }) {
    if (!currentUser) {
        return null;
    }

    const { username, games_played, games_won, total_differences_found } = currentUser;

    return (
        <Card className="p-4 mb-4 shadow-sm rounded-4">
            <div className="d-flex justify-content-between align-items-center mb-3">
                <h5>Hello, {username || 'Guest'}!</h5>
                <div className="d-flex gap-2">
                    <Button variant="outline-primary" size="sm" onClick={onShowGameHistory}>
                        <History size={16} className="me-2" />Game History
                    </Button>
                    <Button variant="outline-danger" size="sm" onClick={onLogout}>
                        <LogOut size={16} className="me-2" />Logout
                    </Button>
                </div>
            </div>
            <Card className="p-4 mb-4">
                <h6><Award size={20} className="me-2" />Your Stats:</h6>
                <p className="mb-0">Games Played: {games_played}</p>
                <p className="mb-0">Games Won: {games_won}</p>
                <p>Total Differences Found: {total_differences_found}</p>
            </Card>
        </Card>
    );
}

export default UserProfile;