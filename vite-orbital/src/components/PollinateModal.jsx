import React, { useState } from "react";
import { Modal, Form, Button, Spinner, Alert } from "react-bootstrap";
import axios from "axios";

export default function PollinateModal({
  show,
  onClose,
  onImageReady,   // callback(pollinateImageUrl)
  backendUrl
}) {
  const [prompt, setPrompt] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError]   = useState("");
  const [generateImage, setGenerateImage] = useState("");

  const handleGenerate = async () => {
    setLoading(true);
    setError("");
    setGenerateImage("");
    try {
      const { data } = await axios.post(
        `${backendUrl}/ai-generate-image`,
        { prompt },
        { 
            headers: { "Content-Type": "application/json" },
            timeout: 30000 // 30 seconds
        }
      );
      setGenerateImage(data.imageUrl);
    } catch (err) {
        if (err.code === "ECONNABORTED") {
            setError("Request timed out. Try a simpler prompt.");
        } else {
            setError("Failed to generate image on Pollinate.ai");
        }
    } finally {
      setLoading(false);
    }
  };

  // When user confirms, fetch the image as a blob and pass it up
  const handleUseImage = async () => {
    try {
      const res = await fetch(generateImage);
      const blob = await res.blob();
      const file = new File([blob], "pollinate_image.png", { type: blob.type });
      onImageReady(URL.createObjectURL(file)); // For preview in LeftUploadCard
      // Optionally, you can also pass the File object up if you want to auto-upload
      onClose();
    } catch (err) {
      setError("Failed to use generated image.");
    }
  };

  return (
    <Modal show={show} onHide={onClose} centered>
      <Modal.Header closeButton>
        <Modal.Title>Generate Image with Pollinate AI</Modal.Title>
      </Modal.Header>

      <Modal.Body>
        <Form.Group>
          <Form.Label>Describe the image:</Form.Label>
          <Form.Control
            type="text"
            value={prompt}
            onChange={(e) => setPrompt(e.target.value)}
            placeholder="A Tom and Jerry Cartoon Image"
            disabled={loading}
          />
        </Form.Group>

        {error && <Alert variant="danger" className="mt-2">{error}</Alert>}

        {loading && <div className="text-center my-3"><Spinner animation="border" /></div>}

        {generateImage && (
          <div className="text-center my-3">
            <img
              src={generateImage}
              alt="Pollinate AI"
              className="img-fluid"
              style={{ maxHeight: 300, objectFit: "contain" }}
            />
          </div>
        )}
      </Modal.Body>

      <Modal.Footer className="d-flex justify-content-between w-100">
        <Button variant="secondary" onClick={onClose}>
            Cancel
        </Button>

        {!generateImage ? (
          <div>
            <Button
              variant="success"
              disabled={!prompt || loading}
              onClick={handleGenerate}
            >
              {loading ? <Spinner animation="border" size="sm" /> : "Generate"}
            </Button>
          </div>
        ) : (
          <div>
            {/* Regenerate button */}
            <Button variant="success" disabled={loading} onClick={handleGenerate} className="me-2">
                Regenerate
            </Button>
            <Button variant="primary" onClick={handleUseImage}>
              Use this Image
            </Button>
          </div>
        )}
      </Modal.Footer>
    </Modal>
  );
}