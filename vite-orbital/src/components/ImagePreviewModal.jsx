import React from 'react';
import { Modal, Button } from 'react-bootstrap';
import './ImagePreviewModal.css';

function ImagePreviewModal({ show, imageUrl, onClose }) {
  return (
    <Modal show={show} onHide={onClose} centered size="lg">
      <Modal.Header closeButton>
        <Modal.Title>Image Preview</Modal.Title>
      </Modal.Header>
      <Modal.Body className="text-center">
        {imageUrl ? (
          <img src={imageUrl} alt="Preview" className="img-fluid preview-image" />
        ) : (
          <p>No image to display.</p>
        )}
      </Modal.Body>
      <Modal.Footer>
        <Button variant="secondary" onClick={onClose}>
          Close
        </Button>
      </Modal.Footer>
    </Modal>
  );
}

export default ImagePreviewModal;