import React, { useState, useEffect } from "react";
import axios from "axios";
import { Container, Table, Spinner, Alert, Button, Row, Col} from "react-bootstrap";
import ImagePreviewModal from "./ImagePreviewModal";
import "./GameHistory.css"; 

const BACKEND_URL = "https://orbital-2025-backend.onrender.com";

function GameHistory({ currentUser, onBackToGame }) {
  const [historyData, setHistoryData] = useState(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState("");

  const [showPreviewModal, setShowPreviewModal] = useState(false);
  const [previewImageUrl, setPreviewImageUrl] = useState("");

  useEffect(() => {
    const fetchGameHistory = async () => {
      if (!currentUser || !currentUser.user_id) {
        setError("User not logged in or user ID not available.");
        setLoading(false);
        return;
      }

      setLoading(true);
      setError("");
      try {
        const response = await axios.get(
          `${BACKEND_URL}/user/${currentUser.user_id}/history`,
          { withCredentials: true }
        );
        setHistoryData(response.data);
        console.log("Game history fetched:", response.data);
      } catch (err) {
        console.error("Error fetching game history:", err);
        setError(
          err.response?.data?.error || "Failed to fetch game history."
        );
      } finally {
        setLoading(false);
      }
    };

    fetchGameHistory();
  }, [currentUser]); // Re-fetch when currentUser changes

  const handleImageClick = (imageUrl) => {
    setPreviewImageUrl(imageUrl);
    setShowPreviewModal(true);
  };

  const handleClosePreviewModal = () => {
    setShowPreviewModal(false);
    setPreviewImageUrl(""); // Clear the image URL when closing
  };

  if (loading) {
    return (
      <Container className="my-5 text-center">
        <Spinner animation="border" role="status">
          <span className="visually-hidden">Loading history...</span>
        </Spinner>
        <p className="mt-2">Loading game history...</p>
      </Container>
    );
  }

  if (error) {
    return (
      <Container className="my-5">
        <Alert variant="danger">
          <Alert.Heading>Error!</Alert.Heading>
          <p>{error}</p>
          <Button variant="outline-danger" onClick={onBackToGame}>
            Back to Game
          </Button>
        </Alert>
      </Container>
    );
  }

  if (!historyData || !historyData.games || historyData.games.length === 0) {
    return (
      <Container className="my-5 text-center">
        <Alert variant="info">
          <Alert.Heading>No Game History Found</Alert.Heading>
          <p>You haven't played any games yet. Start a new game!</p>
          <Button variant="primary" onClick={onBackToGame}>
            Back to Game
          </Button>
        </Alert>
      </Container>
    );
  }

  return (
    <Container className="my-5 game-history-container">
    <Row className="d-flex align-items-center mb-0">
        <Col className="d-flex justify-content-center flex-grow-1">
            <h2 className="text-center mb-4">Game History for {historyData.username}</h2>
        </Col>
        <Col xs="auto" className="ms-auto">
            <Button variant="primary" onClick={onBackToGame}>
                Back to Game
            </Button>
        </Col>
    </Row>
      <Table striped bordered hover responsive className="game-history-table">
        <thead>
          <tr>
            <th>Date Played</th>
            <th>Original Image</th>
            <th>Modified Image</th>
            <th>Score</th>
            <th>Total Differences</th>
            <th>Time Taken (s)</th>
          </tr>
        </thead>
        <tbody>
          {historyData.games.map((game, index) => (
            <tr key={index}>
              <td>{new Date(game.played_at).toLocaleString()}</td>
              <td>
                <img
                  src={game.original_image}
                  alt="Original"
                  className="history-thumbnail"
                  onClick={() => handleImageClick(game.original_image)}
                  style={{ cursor: 'pointer' }}
                />
              </td>
              <td>
                <img
                  src={game.modified_image}
                  alt="Modified"
                  className="history-thumbnail"
                  onClick={() => handleImageClick(game.modified_image)}
                  style={{ cursor: 'pointer' }}
                />
              </td>
              <td>{game.score}</td>
              <td>{game.total}</td>
              <td>{game.time_taken > 0 ? game.time_taken.toFixed(2) : "N/A"}</td>
            </tr>
          ))}
        </tbody>
      </Table>
      <div className="text-center mt-4">
        <Button variant="primary" onClick={onBackToGame}>
          Back to Game
        </Button>
      </div>

      <ImagePreviewModal
        show={showPreviewModal}
        imageUrl={previewImageUrl}
        onClose={handleClosePreviewModal}
      />

    </Container>
  );
}

export default GameHistory;