import React from 'react';
import { Modal, Button } from 'react-bootstrap';
import './HowToPlayModal.css'; 

function HowToPlayModal({ show, onClose }) {
  return (
    <Modal show={show} onHide={onClose} centered size="lg">
      <Modal.Header closeButton className="pb-2">
        <Modal.Title>How To Play</Modal.Title>
      </Modal.Header>
      <Modal.Body className="py-0">
        <h5>Welcome to Spot The Difference!</h5>
        <p>Your goal is simple: find all the differences between two seemingly identical images before you run out of attempts or time (in Time Attack mode).</p>

        <h6>Game Modes:</h6>
        <ul>
          <li><strong>Classic:</strong> Take your time! You have a limited number of wrong clicks (7 tries). The game ends when you find all differences or exceed the wrong click limit.</li>
          <li><strong>Time Attack:</strong> Challenge yourself against the clock! You have a set time limit (e.g., 30 seconds) to find all the differences. The game ends when you find all differences, run out of wrong clicks, or time expires, whichever comes first.</li>
        </ul>

        <h6>How to Play:</h6>
        <ol>
          <li><strong>Choose Your Mode:</strong> Select "Classic" or "Time Attack" from the panel above the images.</li>
          <li><strong>Upload or Generate an Image:</strong>
            <ul>
              <li>Click "Upload Image" to select a picture from your device.</li>
              <li>Or, click "Generate AI Image" and enter a prompt to have an AI create a unique image for you.</li>
            </ul>
          </li>
          <li><strong>Spot the Differences:</strong> Once the modified image appears on the right, carefully compare it to the original on the left. Click on any area in the **modified image** where you spot a difference.</li>
          <li><strong>Feedback:</strong>
            <ul>
              <li>A <span style={{ color: 'green', fontWeight: 'bold' }}>green circle</span> will appear if you find a correct difference.</li>
              <li>A <span style={{ color: 'red', fontWeight: 'bold' }}>red 'X'</span> will appear briefly if you click on a wrong spot (Classic mode only).</li>
            </ul>
          </li>
          <li><strong>Game End:</strong>
            <ul>
              <li>The game ends when you find all differences (you win!).</li>
              <li>In Classic mode, the game also ends if you make too many wrong clicks.</li>
              <li>In Time Attack mode, the game also ends if the timer runs out.</li>
            </ul>
          </li>
          <li><strong>Game History:</strong> Logged-in users can view their past game attempts, scores, and times (for Time Attack) on the "Game History" page.</li>
        </ol>

        <p>Have fun and sharpen your observation skills!</p>
      </Modal.Body>
      <Modal.Footer className="py-1">
        <Button variant="primary" onClick={onClose}>
          Got It!
        </Button>
      </Modal.Footer>
    </Modal>
  );
}

export default HowToPlayModal;