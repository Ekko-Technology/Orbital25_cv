import React, { useState, useRef, useEffect, useCallback } from "react";
import "./home-body.css";
import {
  Container,
  Row,
  Col,
  Form,
  Button,
  Card,
  Spinner,
  Alert,
} from "react-bootstrap";
import MessageAlert from "../MessageAlert";
import LeftUploadCard from "../LeftUploadCards";
import RightUploadCard from "../RightUploadCards";
import RestartButton from "../RestartButton";
import axios from "axios";
import { v4 as uuidv4 } from "uuid";


const BACKEND_URL = "https://orbital25-cv.onrender.com"; // Connecting to Render-hosted backend

// const BACKEND_URL = "http://localhost:5000"; // Connecting to Flask backend

function Homebody() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [originalImageUrl, setOriginalImageUrl] = useState("");
  const [modifiedImageUrl, setModifiedImageUrl] = useState("");
  const [differences, setDifferences] = useState([]);
  const [foundDifferences, setFoundDifferences] = useState(new Set());
  const [clickAttempts, setClickAttempts] = useState([]);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [gameStarted, setGameStarted] = useState(false);
  const [gameEnded, setGameEnded] = useState(true);
  const [showConfirm, setShowConfirm] = useState(false);

  // References for the canvas and images to get dimensions
  const modifiedImageRef = useRef(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const MAX_WRONG_CLICKS = 10;


  // Function to draw circles on the canvas
  const drawCircles = useCallback(() => {
    const canvas = canvasRef.current;
    const ctx = canvas.getContext("2d");
    const img = modifiedImageRef.current;

    if (!img || !canvas || !ctx) return;

    const naturalWidth = img.naturalWidth;
    const naturalHeight = img.naturalHeight;

    const displayedWidth = img.offsetWidth;
    const displayedHeight = img.offsetHeight;

    canvas.width = displayedWidth;
    canvas.height = displayedHeight;

    // erases canvas 
    ctx.clearRect(0, 0, canvas.width, canvas.height);

    const now = Date.now();
    const X_DURATION = 1000;

    // Draw circles for correct and incorrect clicks
    clickAttempts.forEach((attempt) => {
      const {x, y, type, timestamp} = attempt;
      if (type === "wrong" && (now - timestamp) < X_DURATION) {
        ctx.save()
        ctx.strokeStyle = "red";
        ctx.lineWidth = 4;
        ctx.beginPath();
        ctx.moveTo(x - 10, y - 10);
        ctx.lineTo(x + 10, y + 10);
        ctx.moveTo(x + 10, y - 10);
        ctx.lineTo(x - 10, y + 10);
        ctx.stroke();
        ctx.restore();
      }
    });

    // Draw model answer circles for found differences
    differences.forEach((diff) => {
      if (foundDifferences.has(diff.id)) {
        const [x1_natural, y1_natural, x2_natural, y2_natural] = diff.coords;
        const scaleX = displayedWidth / naturalWidth;
        const scaleY = displayedHeight / naturalHeight;
        const x1_display = x1_natural * scaleX;
        const y1_display = y1_natural * scaleY;
        const x2_display = x2_natural * scaleX;
        const y2_display = y2_natural * scaleY;
        const centerX_display = (x1_display + x2_display) / 2;
        const centerY_display = (y1_display + y2_display) / 2;
        const radius = Math.max(
          (x2_display - x1_display) / 2,
          (y2_display - y1_display) / 2,
          20
        );
        ctx.beginPath();
        ctx.arc(centerX_display, centerY_display, radius, 0, Math.PI * 2);
        ctx.lineWidth = 3;
        ctx.strokeStyle = "green";
        ctx.stroke();
      }
    });

    
  //   clickAttempts.forEach((attempt) => {
  //     const {x, y, type, timestamp} = attempt;
  //     if (type === "correct")
  //     {
  //       ctx.beginPath();
  //       ctx.arc(x, y, 20, 0, Math.PI * 2);
  //       ctx.lineWidth = 3;
  //       ctx.strokeStyle = "green";
  //       ctx.stroke();
  //     }
  //     else if (type === "wrong" && (now - timestamp) < X_DURATION) {
  //       ctx.save();
  //       ctx.strokeStyle = "red";
  //       ctx.lineWidth = 4;
  //       ctx.beginPath();
  //       ctx.moveTo(x - 10, y - 10);
  //     ctx.lineTo(x + 10, y + 10);
  //     ctx.moveTo(x + 10, y - 10);
  //     ctx.lineTo(x - 10, y + 10);
  //     ctx.stroke();
  //     ctx.restore();
  //     }
  //   });


    // If game is over (all found or too many wrong clicks), reveal all differences
    if (foundDifferences.size === differences.length && differences.length > 0) {
      differences.forEach((diff) => {
        const [x1_natural, y1_natural, x2_natural, y2_natural] = diff.coords;

        // Scale natural coordinates to displayed coordinates for drawing
        const scaleX = displayedWidth / naturalWidth;
        const scaleY = displayedHeight / naturalHeight;

        const x1_display = x1_natural * scaleX;
        const y1_display = y1_natural * scaleY;
        const x2_display = x2_natural * scaleX;
        const y2_display = y2_natural * scaleY;

        const centerX_display = (x1_display + x2_display) / 2;
        const centerY_display = (y1_display + y2_display) / 2;

        const radius = Math.max(
          (x2_display - x1_display) / 2,
          (y2_display - y1_display) / 2,
          20
        );

        ctx.beginPath();
        ctx.arc(centerX_display, centerY_display, radius, 0, Math.PI * 2);
        ctx.lineWidth = 3;
        ctx.strokeStyle = "lime";
        ctx.stroke();
      });
    }
  }, [clickAttempts, foundDifferences, differences]); // Redraw when dependencies change



  // Redraw when modifiedImageUrl changes or window resizes
  useEffect(() => {
    const img = modifiedImageRef.current;
    if (img && img.complete) {
      // Ensure image is fully loaded before drawing
      drawCircles();
    }
  }, [modifiedImageUrl, drawCircles]); // Run when modified image URL changes

  useEffect(() => {
    window.addEventListener("resize", drawCircles);
    // Add an event listener to the image itself in case it loads AFTER component mounts
    const img = modifiedImageRef.current;
    if (img) {
      img.addEventListener("load", drawCircles);
    }

    return () => {
      window.removeEventListener("resize", drawCircles);
      if (img) {
        img.removeEventListener("load", drawCircles);
      }
    };
  }, [drawCircles]);

  useEffect(() => {
    return () => {
      if (originalImageUrl && originalImageUrl.startsWith("blob:")) {
        URL.revokeObjectURL(originalImageUrl);
      }
    };
  }, [originalImageUrl]);


  useEffect(() => {
    const now = Date.now();
    const X_DURATION = 1000;
    const wrongTimestamps = clickAttempts
    .filter(a => a.type === "wrong" && now - a.timestamp < X_DURATION)
    .map(a => a.timestamp);
    if (wrongTimestamps.length > 0) {
      const soonest = Math.min(...wrongTimestamps);
      const timeout = X_DURATION - (now - soonest);
      const timer = setTimeout(drawCircles, timeout + 10);
      return () => clearTimeout(timer);
    }
  }, [clickAttempts, drawCircles]);



  const handleFileChange = (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);

    // Reset game state when a new file is selected
    setModifiedImageUrl("");
    setDifferences([]);
    setFoundDifferences(new Set());
    setClickAttempts([]);
    setMessage("");
    setError("");
    setGameStarted(false);

    // Create a temporary URL for the selected file to display it immediately
    if (file) {
      const objectUrl = URL.createObjectURL(file);
      setOriginalImageUrl(objectUrl);
    } else {
      setOriginalImageUrl("");
    }
  };

  const handleUpload = async () => {
    if (!selectedFile) {
      setError("Please select an image file first.");
      return;
    }

    setLoading(true);
    setError("");
    setMessage("");


    //sends response to flask backend for "upload-and-process"
    const formData = new FormData();
    formData.append("image", selectedFile);

    try {
      const response = await axios.post(
        `${BACKEND_URL}/upload-and-process`,
        formData,
        {
          headers: {
            "Content-Type": "multipart/form-data",
          },
        }
      );
      
      // response handling from backend 
      const {
        originalImageUrl: backendOriginalUrl,
        modifiedImageUrl,
        rawDifferencesForFrontendDemo,
      } = response.data;

      // Revoke the temporary blob URL for the original image if it exists
      if (originalImageUrl.startsWith("blob:")) {
        URL.revokeObjectURL(originalImageUrl);
      }

      
      setOriginalImageUrl(backendOriginalUrl);
      setModifiedImageUrl(modifiedImageUrl);
      // setOriginalImageUrl(`${BACKEND_URL}${backendOriginalUrl}`);
      // setModifiedImageUrl(`${BACKEND_URL}${modifiedImageUrl}`);

      // Assign a unique ID to each difference for tracking found differences
      const differencesWithIds = rawDifferencesForFrontendDemo.map(
        (coords) => ({
          id: uuidv4(), // Generate a unique ID for each difference
          coords: coords,
        })
      );
      setDifferences(differencesWithIds);

      setGameStarted(true);
      setGameEnded(false);
      setMessage("Images loaded! Find the differences.");
      setClickAttempts([]); // Reset click attempts for new game
      setFoundDifferences(new Set()); // Reset found differences for new game
    } catch (err) {
      console.error("Error uploading or processing image:", err);
      if (err.response && err.response.data && err.response.data.error) {
        setError(
          `Failed to upload or process image: ${err.response.data.error}`
        );
      } else {
        setError("Failed to upload or process image. Please try again.");
      }
      setGameStarted(false);
      setGameEnded(true);
    } finally {
      setLoading(false);
    }
  };

  // Handle click on the modified image
  const handleImageClick = (event) => {
    if (
      !gameStarted ||
      foundDifferences.size === differences.length ||
      clickAttempts.filter((a) => a.type === "wrong").length >= MAX_WRONG_CLICKS
    ) {
      // Don't allow clicks if game not started, finished, or too many wrong clicks
      return;
    }

    const img = modifiedImageRef.current;
    if (!img) return;

    // Get click coordinates relative to the image element
    const rect = img.getBoundingClientRect();
    const clickX_display = event.clientX - rect.left;
    const clickY_display = event.clientY - rect.top;

    // Scale click coordinates to the natural (backend) dimensions of the image
    const scaleX = img.naturalWidth / img.offsetWidth;
    const scaleY = img.naturalHeight / img.offsetHeight;

    const clickX_natural = clickX_display * scaleX;
    const clickY_natural = clickY_display * scaleY;

    // Check if the click is within any unfound difference area
    let isCorrectClick = false;
    let foundDiffId = null;

    for (const diff of differences) {
      if (!foundDifferences.has(diff.id)) {
        const [x1, y1, x2, y2] = diff.coords;

        const tolerance = 10;
        if (
          clickX_natural >= x1 - tolerance &&
          clickX_natural <= x2 + tolerance &&
          clickY_natural >= y1 - tolerance &&
          clickY_natural <= y2 + tolerance
        ) {
          isCorrectClick = true;
          foundDiffId = diff.id;
          break; // Found a difference, no need to check others
        }
      }
    }

    // --- Game Logic ---
    if (isCorrectClick) {
      if (foundDiffId) {
        setFoundDifferences((prev) => new Set(prev).add(foundDiffId));
        setMessage("Difference found! Keep going!");

        // Check if all differences are found
        if (foundDifferences.size + 1 === differences.length) {
          setMessage("Congratulations! You found all the differences!");
          setGameStarted(false); // End game
          setGameEnded(true);

          setTimeout(() => drawCircles(), 50);
        }
      }
    } else {
      const wrongClicks = clickAttempts.filter(
        (attempt) => attempt.type === "wrong"
      ).length;
      if (wrongClicks < MAX_WRONG_CLICKS - 1) {
        setClickAttempts((prev) => [
          ...prev,
          {
            x: clickX_display,
            y: clickY_display,
            type: "wrong",
            timestamp: Date.now()
          },
        ]);
        setMessage(
          `Oops! Wrong spot. You have ${
            MAX_WRONG_CLICKS - (wrongClicks + 1)
          } tries left.`
        );
      } else {
        // This is the last wrong click
        setClickAttempts((prev) => [
          ...prev,
          {
            x: clickX_display,
            y: clickY_display,
            type: "wrong",
          },
        ]);
        setMessage(
          `Game Over! You made too many wrong clicks. The differences are now revealed.`
        );
        setGameStarted(false); // End game
        setGameEnded(true);

        setFoundDifferences(new Set(differences.map((d) => d.id))); // Reveal all differences

        setTimeout(() => drawCircles(), 50); // Trigger redraw to show all highlights immediately
      }
    }
  };

  // Function to trigger the hidden file input
  const triggerFileInput = () => {
    fileInputRef.current.click();
  };

  return (
    <Container className="my-5">
      {/* Main Control Card (for file selection and messages) */}
      <Row className="mb-3 justify-content-center">
        <Col md={12}>
          {/* If error occurs, insert error div to inform user*/}
          {error && (
            <MessageAlert
              type="danger"
              text={error}
              onClose={() => setError(null)}
            />
          )}
          {message && (
            <MessageAlert
              type="info"
              text={message}
              onClose={() => setMessage("")}
            />
          )}
        </Col>
      </Row>

      <Row className="justify-content-center">
        {/* Left Image Card: Original Image */}
        <Col md={6} className="mb-3">
          <LeftUploadCard
            HeaderText="Original Image"
            onFileSelect={handleFileChange}
            onUpload={handleUpload}
            loading={loading}
            selectedFile={selectedFile}
            fileInputRef={fileInputRef}
            triggerFileInput={triggerFileInput}
            originalImageUrl={originalImageUrl}
            modifiedImageUrl={modifiedImageUrl}
          />
        </Col>

        {/* Right Image Card: Modified Image */}
        <Col md={6} className="mb-3">
          <RightUploadCard
            onUpload={handleUpload}
            loading={loading}
            selectedFile={selectedFile}
            modifiedImageUrl={modifiedImageUrl}
            modifiedImageRef={modifiedImageRef}
            canvasRef={canvasRef}
            handleImageClick={handleImageClick}
            gameStarted={gameStarted}
            foundDifferences={foundDifferences}
            differences={differences}
            clickAttempts={clickAttempts}
            MAX_WRONG_CLICKS={MAX_WRONG_CLICKS}
          />
        </Col>
      </Row>
      
      {/* restart game with new image button */}
      <Row className="justify-content-center mt-4">
        <Col md={4} className="d-flex justify-content-center">
          <RestartButton
            gameEnded={gameEnded}
            loading={loading}
            onRestart={triggerFileInput}
            showConfirm={showConfirm}
            setShowConfirm={setShowConfirm}
          />
        </Col>
      </Row>
    </Container>
  );
}

export default Homebody;
