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
  Badge
} from "react-bootstrap";
import MessageAlert from "../MessageAlert";
import LeftUploadCard from "../LeftUploadCards";
import RightUploadCard from "../RightUploadCards";
import RestartButton from "../RestartButton";
import PollinateModal from "../PollinateModal";
import UserProfile from "../UserProfile";
import GameHistory from "../GameHistory"; 
import axios from "axios";
import { v4 as uuidv4 } from "uuid";

const BACKEND_URL = "https://orbital25-cv.onrender.com"; // Connecting to Render-hosted backend

const TIME_LIMIT_SECONDS = 30;

function Homebody({ isLoggedIn, currentUser, onUpdateUserStats, onLogout }) {
  const [selectedFile, setSelectedFile] = useState(null);
  const [originalImageUrl, setOriginalImageUrl] = useState("");
  const [modifiedImageUrl, setModifiedImageUrl] = useState("");

  const [originalImageCloudinaryUrl, setOriginalImageCloudinaryUrl] = useState(""); 
  const [modifiedImageCloudinaryUrl, setModifiedImageCloudinaryUrl] = useState("");
 
  const [originalImagePublicId, setOriginalImagePublicId] = useState("");
  const [modifiedImagePublicId, setModifiedImagePublicId] = useState("");

  const currentPublicIdsRef = useRef({ original: "", modified: "" });

  const [differences, setDifferences] = useState([]);
  const [foundDifferences, setFoundDifferences] = useState(new Set());
  const [clickAttempts, setClickAttempts] = useState([]);
  const [message, setMessage] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [gameStarted, setGameStarted] = useState(false);
  const [gameEnded, setGameEnded] = useState(true);
  const [showConfirm, setShowConfirm] = useState(false);
  const [showGameHistory, setShowGameHistory] = useState(false);

  const [gameMode, setGameMode] = useState('classic');

  const [timeLeft, setTimeLeft] = useState(TIME_LIMIT_SECONDS);
  const timerIntervalRef = useRef(null);
  const gameStartTimeRef = useRef(null);

  // States for utilizing Pollinate AI
  const [showPollinateModal, setShowPollinateModal] = useState(false);
  const [pollinateImage, setPollinateImage] = useState(""); // url of generated image

  // References for the canvas and images to get dimensions
  const modifiedImageRef = useRef(null);
  const canvasRef = useRef(null);
  const fileInputRef = useRef(null);
  const MAX_WRONG_CLICKS = 7;

  // Helper function to extract public ID from Cloudinary URL
  const getPublicIdFromUrl = (url) => {
    if (!url) return null;
    const parts = url.split('/');

    const uploadIndex = parts.indexOf('upload');
    if (uploadIndex === -1 || uploadIndex + 2 >= parts.length) return null; // Ensure 'upload' and enough parts after it
    const publicIdWithExtension = parts.slice(uploadIndex + 2).join('/');
    return publicIdWithExtension.split('.')[0]; // Remove extension
  };

  // Function to send POST request to start deleting guest files (now Cloudnary assets)
  const cleanupGuestImages = async (publicIds = []) => {
    try {
      await axios.post(
        `${BACKEND_URL}/cleanup-guest-files`,
        { public_ids: publicIds },
        { withCredentials: true }
      );
      console.log("Temporary guest Cloudinary assets cleaned up.");
    } catch (err) {
      console.error("Failed to clean up guest Cloudinary assets", err);
    }
  };

  // New function to delete temporary Cloudinary assets for logged-in users
  const deleteUserTempImages = async (publicIds) => {
    if (!isLoggedIn || !publicIds || publicIds.length === 0) {
      return;
    }
    try {
      await axios.post(`${BACKEND_URL}/delete-user-temp-images`, { public_ids: publicIds }, { withCredentials: true });
      console.log("Logged-in user's previous temporary Cloudinary assets cleaned up.");
    } catch (err) {
      console.error("Failed to clean up logged-in user's temporary Cloudinary assets:", err);
    }
  };


  // Function to save game data and images (now Cloudinary URLs)
  const saveGameDataAndImages = async (scoreValue, totalValue, originalUrl, modifiedUrl, timeTaken = 0) => { 
    if (!isLoggedIn || !currentUser?.user_id) {
      console.warn("Attempted to save game data without being logged in.");
      return;
    }

    if (!originalUrl || !modifiedUrl) { 
      console.error("Missing Cloudinary URLs for saving.");
      return;
    }

    console.log("Saving game data. Payload:", { 
        original_image_cloudinary_url: originalUrl,
        modified_image_cloudinary_url: modifiedUrl,
        score: scoreValue,
        total: totalValue,
        time_taken: timeTaken,
    });

    try {
    const response = await axios.post(`${BACKEND_URL}/save-game`, {
    original_image_cloudinary_url: originalUrl, 
    modified_image_cloudinary_url: modifiedUrl, 
    score: scoreValue, 
    total: totalValue, 
    time_taken: timeTaken,
    }, { withCredentials: true });

    setMessage(response.data.message);
    console.log("Game saved to DB and images uploaded to Cloudinary:", response.data);


    } catch (err) {
      console.error("Failed to save game data or upload images to Cloudinary:", err);
      if (err.response && err.response.data && err.response.data.error) {
        setError(
        `Failed to save game data: ${err.response.data.error}`
      );
      } else {
       setError("Failed to save game data. Please try again.");
      }
    }
  };

  // Function to draw circles on the canvas
  const drawCircles = useCallback(() => {
    const canvas = canvasRef.current;
    if (!canvas) {
      console.warn("Canvas ref is null, cannot draw circles.");
      return;
    }
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
      const { x, y, type, timestamp } = attempt;
      if (type === "wrong" && now - timestamp < X_DURATION) {
        ctx.save();
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

    // If game is over (all found or too many wrong clicks), reveal all differences
    if (
      foundDifferences.size === differences.length &&
      differences.length > 0
    ) {
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
      .filter((a) => a.type === "wrong" && now - a.timestamp < X_DURATION)
      .map((a) => a.timestamp);
    if (wrongTimestamps.length > 0) {
      const soonest = Math.min(...wrongTimestamps);
      const timeout = X_DURATION - (now - soonest);
      const timer = setTimeout(drawCircles, timeout + 10);
      return () => clearTimeout(timer);
    }
  }, [clickAttempts, drawCircles]);


  useEffect(() => {
    currentPublicIdsRef.current = { 
      original: originalImagePublicId, 
      modified: modifiedImagePublicId 
    };
    console.log("DEBUG: currentPublicIdsRef updated by useEffect:", currentPublicIdsRef.current);
  }, [originalImagePublicId, modifiedImagePublicId]);

  useEffect(() => {
    if (gameStarted && gameMode === 'timeAttack') {
      timerIntervalRef.current = setInterval(() => {
        setTimeLeft((prevTime) => {
          if (prevTime <= 1) { // Use <= 1 to ensure it hits 0 and then stops
            clearInterval(timerIntervalRef.current);
            endGameDueToTime(); // End game when time runs out
            return 0;
          }
          return prevTime - 1;
        });
      }, 1000);
    }

    // Cleanup function for the interval
    return () => {
      if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
        timerIntervalRef.current = null;
      }
    };
  }, [gameStarted, gameMode]);

  const endGameDueToTime = useCallback(() => {
    console.log("Time ran out! Ending game.");
    setGameStarted(false);
    setGameEnded(true);

    // const finalScore = foundDifferences.size; 
    // const totalDifferences = differences.length;

    // setMessage(`Time's up! You found ${finalScore} out of ${totalDifferences} differences. The differences are now revealed.`);
    // setFoundDifferences(new Set(differences.map((d) => d.id))); // Reveal all differences

    // let finalTimeTaken = 0;
    // if (gameStartTimeRef.current) {
    //     finalTimeTaken = (Date.now() - gameStartTimeRef.current) / 1000;
    // }
    
    // Save game data for Time Attack mode
    // if (isLoggedIn) {
    //     saveGameDataAndImages(
    //         finalScore, 
    //         totalDifferences, 
    //         originalImageCloudinaryUrl, 
    //         modifiedImageCloudinaryUrl, 
    //         finalTimeTaken
    //     );
    //     onUpdateUserStats(finalScore, false); // Game not won by finding all diffs, but by time out
    // }
    setFoundDifferences(prevFoundDifferences => {
        const finalScore = prevFoundDifferences.size; 
        const totalDifferences = differences.length;

        console.log("DEBUG: finalScore captured in functional update:", finalScore);

        setMessage(`Time's up! You found ${finalScore} out of ${totalDifferences} differences. The differences are now revealed.`);
        
        let finalTimeTaken = 0;
        if (gameStartTimeRef.current) {
            finalTimeTaken = (Date.now() - gameStartTimeRef.current) / 1000;
        }
        
        if (isLoggedIn) {
            saveGameDataAndImages(
                finalScore, 
                totalDifferences, 
                originalImageCloudinaryUrl, 
                modifiedImageCloudinaryUrl, 
                finalTimeTaken
            );
            onUpdateUserStats(finalScore, false); 
        }

        // Return the new Set to reveal all differences visually
        return new Set(differences.map((d) => d.id));
    });
  }, [foundDifferences, differences, originalImageCloudinaryUrl, modifiedImageCloudinaryUrl, isLoggedIn, saveGameDataAndImages, onUpdateUserStats]);
  
  // Completely clear the board
  const resetGameState = async () => {
    console.log("DEBUG: resetGameState called."); 
    console.log("DEBUG: currentPublicIdsRef.current at start of resetGameState:", currentPublicIdsRef.current); // Crucial debug log
    console.log("DEBUG: isLoggedIn:", isLoggedIn, "gameEnded:", gameEnded); // Added debug for conditions

    // Stop any active timer
    if (timerIntervalRef.current) {
        clearInterval(timerIntervalRef.current);
        timerIntervalRef.current = null;
    }
    setTimeLeft(TIME_LIMIT_SECONDS); // Reset timer display
    gameStartTimeRef.current = null; // Reset game start time

    const publicIdsToClean = [];
    if (currentPublicIdsRef.current.original) publicIdsToClean.push(currentPublicIdsRef.current.original);
    if (currentPublicIdsRef.current.modified) publicIdsToClean.push(currentPublicIdsRef.current.modified);

    console.log("DEBUG: publicIdsToClean array:", publicIdsToClean);

    // CONDITIONAL CLEANUP LOGIC
    const shouldDelete = publicIdsToClean.length > 0 && (!isLoggedIn || !gameEnded);
    
    if (shouldDelete) {
      try {
        if (isLoggedIn) {
          console.log(`Frontend: Cleaning up user ${currentUser?.user_id}'s images on restart:`, publicIdsToClean);
          await deleteUserTempImages(publicIdsToClean);
        } else {
          console.log(`Frontend: Cleaning up guest images on restart:`, publicIdsToClean);
          await cleanupGuestImages(publicIdsToClean);
        }
      } catch (error) {
        console.error("Error during explicit image cleanup on restart:", error);
      }
    }
    else if (publicIdsToClean.length > 0 && isLoggedIn && gameEnded) {
        // If logged in and game ended, do NOT delete temporary images on restart
        console.log("Logged-in user finished game and is restarting. Skipping temporary image cleanup.");
    } 
    else {
      console.log("No public IDs to clean up, or no previous game was active.");
    }

    setSelectedFile(null);
    setOriginalImageUrl("");
    setModifiedImageUrl("");
    setOriginalImageCloudinaryUrl("");
    setModifiedImageCloudinaryUrl("");
    setOriginalImagePublicId(""); // Clear public IDs from state AFTER cleanup
    setModifiedImagePublicId("");
    currentPublicIdsRef.current = { original: "", modified: "" }; 
    setPollinateImage(""); // clear AI preview
    setDifferences([]);
    setFoundDifferences(new Set());
    setClickAttempts([]);
    setMessage("");
    setError("");
    setGameStarted(false);
    setGameEnded(true);

    if (canvasRef.current) {
      const ctx = canvasRef.current.getContext("2d");
      if (ctx)
        ctx.clearRect(0, 0, canvasRef.current.width,canvasRef.current.height);
    }
    console.log("Game state reset complete.");
  }; 


  const handleFileChange = async (event) => {
    const file = event.target.files[0];
    setSelectedFile(file);

    // Reset game state when a new file is selected
    setModifiedImageUrl("");
    setOriginalImageCloudinaryUrl(""); // Clear Cloudinary URLs
    setModifiedImageCloudinaryUrl(""); // Clear Cloudinary URLs
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
      // setError("Please select an image file first.");
      return;
    }

    console.log("--- Frontend: Attempting upload ---");
    console.log("Frontend: isLoggedIn =", isLoggedIn);
    console.log("Frontend: currentUser =", currentUser);
    console.log("Frontend: BACKEND_URL =", BACKEND_URL);

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
          withCredentials: true,
        }
      );

      // response handling from flask backend
      const {
        originalImageUrl: backendOriginalUrl,
        modifiedImageUrl,
        original_image_cloudinary_url, // Now expecting Cloudinary URLs
        modified_image_cloudinary_url, 
        original_public_id, // Public ID from backend
        modified_public_id, 
        rawDifferencesForFrontendDemo,
      } = response.data;

      console.log("Frontend received originalImageUrl:", backendOriginalUrl);
      console.log("Frontend received modifiedImageUrl:", modifiedImageUrl);
      console.log("Frontend received original_image_cloudinary_url (for save):", original_image_cloudinary_url);
      console.log("Frontend received modified_image_cloudinary_url (for save):", modified_image_cloudinary_url);
      console.log("Frontend received original_public_id (for cleanup):", original_public_id);
      console.log("Frontend received modified_public_id (for cleanup):", modified_public_id);

      // Revoke the temporary blob URL for the original image if it exists
      if (originalImageUrl && originalImageUrl.startsWith("blob:")) {
        URL.revokeObjectURL(originalImageUrl);
      }

      setOriginalImageUrl(backendOriginalUrl);
      setModifiedImageUrl(modifiedImageUrl);

      setOriginalImageCloudinaryUrl(original_image_cloudinary_url);
      setModifiedImageCloudinaryUrl(modified_image_cloudinary_url);
      setOriginalImagePublicId(original_public_id); // Store public ID
      setModifiedImagePublicId(modified_public_id);
      currentPublicIdsRef.current = { original: original_public_id, modified: modified_public_id };

      console.log("DEBUG: Public IDs set after upload. State:", { originalImagePublicId, modifiedImagePublicId }, "Ref:", currentPublicIdsRef.current);

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

      //start timer if game mode is Time Attack
      if (gameMode === 'timeAttack') {
          console.log("Starting Time Attack timer!");
          setTimeLeft(TIME_LIMIT_SECONDS);
          gameStartTimeRef.current = Date.now();
      }

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
      // Clear image states on error
      setOriginalImageUrl("");
      setModifiedImageUrl("");
      setOriginalImageCloudinaryUrl("");
      setModifiedImageCloudinaryUrl("");
      setOriginalImagePublicId("");
      setModifiedImagePublicId("");
      currentPublicIdsRef.current = { original: "", modified: "" };
    } finally {
      setLoading(false);
    }
  };

  // Handle click on the modified image
  const handleImageClick = (event) => {
    if (
      !gameStarted ||
      foundDifferences.size === differences.length ||
      clickAttempts.filter((a) => a.type === "wrong").length >= MAX_WRONG_CLICKS ||
      (gameMode === 'timeAttack' && timeLeft <= 0)
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
        const currentScore = foundDifferences.size + 1;
        const totalDifferences = differences.length;
        const currentOriginalUrl = originalImageCloudinaryUrl; // Use Cloudinary URL
        const currentModifiedUrl = modifiedImageCloudinaryUrl; 



        if (currentScore === totalDifferences) {
          setMessage("Congratulations! You found all the differences!");
          setGameStarted(false); // End game
          setGameEnded(true);

          // Stop timer if active
          if (timerIntervalRef.current) {
            clearInterval(timerIntervalRef.current);
            timerIntervalRef.current = null;
          }

          let finalTimeTaken = 0;
          if (gameMode === 'timeAttack' && gameStartTimeRef.current) {
              finalTimeTaken = (Date.now() - gameStartTimeRef.current) / 1000;
          }

          setTimeout(() => {
            // drawCircles(),
            if (isLoggedIn) {
                    saveGameDataAndImages(currentScore, totalDifferences, currentOriginalUrl, currentModifiedUrl, finalTimeTaken);
                    onUpdateUserStats(currentScore, true);
                } else {
                    // cleanupGuestImages(); // For guest users, cleanup is handled by the proactive cleanup on new upload
                    // or by explicit restart. No need for cleanup here if game completed.
                }
          }, 50);
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
            timestamp: Date.now(),
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

        // Stop timer if active
        if (timerIntervalRef.current) {
            clearInterval(timerIntervalRef.current);
            timerIntervalRef.current = null;
        }

        const finalScoreOnLoss = foundDifferences.size;
        const totalDifferences = differences.length;
        const currentOriginalUrl = originalImageCloudinaryUrl;  
        const currentModifiedUrl = modifiedImageCloudinaryUrl;

        let finalTimeTaken = 0;
        if (gameMode === 'timeAttack' && gameStartTimeRef.current) {
            finalTimeTaken = (Date.now() - gameStartTimeRef.current) / 1000;
        }

        setFoundDifferences(new Set(differences.map((d) => d.id))); // Reveal all differences

        setTimeout(() => {
          // drawCircles(),
          if (isLoggedIn) {
            saveGameDataAndImages(finalScoreOnLoss, totalDifferences, currentOriginalUrl, currentModifiedUrl);
            onUpdateUserStats(finalScoreOnLoss, false);
          } else {
            // cleanupGuestImages(); // For guest users, cleanup is handled by the proactive cleanup on new upload
            // or by explicit restart. No need for cleanup here if game completed.
          }// Update stats for loss, found diffs only
        }, 50); // Trigger redraw to show all highlights immediately
      }
    }
  };

  
  // Function to trigger the hidden file input
  const triggerFileInput = () => {
    resetGameState();               
    if (fileInputRef.current) {
      fileInputRef.current.value = "";
      fileInputRef.current.click();     
    }
  };

  const handleShowGameHistory = () => {
    setShowGameHistory(true);
  };

  const handleBackToGame = () => {
    setShowGameHistory(false);
  };

  const handleGameModeChange = (mode) => {
    if (gameStarted) {
      // Prevent changing mode mid-game
      setMessage("Please restart the game to change modes.");
      return;
    }
    setGameMode(mode);
    setMessage(`Game mode set to: ${mode === 'classic' ? 'Classic' : 'Time Attack'}`);
    resetGameState(); // Reset game state when mode changes
  };

  const handleTestSession = async () => {
    try {
        console.log("--- Frontend: Testing session ---");
        console.log("Frontend: isLoggedIn =", isLoggedIn);
        console.log("Frontend: currentUser =", currentUser);
        const response = await axios.get(`${BACKEND_URL}/test-session`, { withCredentials: true });
        console.log("Frontend: /test-session response:", response.data);
        setMessage(`Session test: ${response.data.message}. User ID: ${response.data.user_id}`);
    } catch (err) {
        console.error("Frontend: Error testing session:", err);
        setError(err.response?.data?.error || "Failed to test session.");
    }
  };

  return (
    // pollinate AI model
    <>
      <PollinateModal
        show={showPollinateModal}
        onClose={() => setShowPollinateModal(false)}
        backendUrl={BACKEND_URL}
        onImageReady={async (url) => {
          await resetGameState();
          setPollinateImage(url);
          const res = await fetch(url);
          const blob = await res.blob();
          const file = new File([blob], "pollinate_image.png", {
            type: blob.type,
          });
          setSelectedFile(file);
          setOriginalImageUrl(url);
          await handleUpload();
        }}
      />

      <Container className="my-5">
        <Row className="mb-3 justify-content-center">
          <Col md={12}>
            {/* User Profile display for logged-in users */}
            {isLoggedIn && currentUser && (
                <UserProfile 
                currentUser={currentUser} 
                onLogout={onLogout} 
                onShowGameHistory={handleShowGameHistory}/>
            )}

            {/*Messages & Alerts Card */}
            {error && <MessageAlert type="danger" text={error} onClose={() => setError(null)} />}
            {message && <MessageAlert type="info" text={message} onClose={() => setMessage("")} />}
          </Col>
        </Row>


        {/* Main Control Card with Conditional Rendering: Game History or Main Game*/}
        {showGameHistory ? (
          <GameHistory 
            currentUser={currentUser} 
            onBackToGame={handleBackToGame} 
          />
        ) : (
          <>
            {/* Game Mode Selection Panel */}
            <Row className="mb-3 justify-content-center">
                <Col md={8}>
                    <Card className="game-mode-panel p-0 shadow-sm rounded-3">
                        <Card.Body className="d-flex justify-content-center align-items-center flex-wrap">
                            <h5 className="mb-0 me-3">Game Mode:</h5>
                            <Button 
                                variant={gameMode === 'classic' ? 'primary' : 'outline-primary'} 
                                onClick={() => handleGameModeChange('classic')}
                                className="me-2 mb-2 mb-md-0 rounded-pill"
                                disabled={gameStarted}
                            >
                                Classic
                            </Button>
                            <Button 
                                variant={gameMode === 'timeAttack' ? 'danger' : 'outline-danger'} 
                                onClick={() => handleGameModeChange('timeAttack')}
                                className="rounded-pill"
                                disabled={gameStarted}
                            >
                                Time Attack
                            </Button>
                            {/* Timer Display */}
                            {gameMode === 'timeAttack' && gameStarted && (
                                <div className="ms-md-auto mt-2 mt-md-0 text-center">
                                    <h5 className="mb-0">Time Left: <Badge bg="warning" text="dark" className="fs-5">{timeLeft}s</Badge></h5>
                                </div>
                            )}
                        </Card.Body>
                    </Card>
                </Col>
            </Row>

            <Row className="mb-3 justify-content-center">
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
                      pollinateImage={pollinateImage}
                      setShowPollinateModal={setShowPollinateModal}
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
                  onRestart={resetGameState}
                  showConfirm={showConfirm}
                  setShowConfirm={setShowConfirm}
                />
              </Col>
            </Row>
          </>
        )}
      </Container>
    </>
  );
}

export default Homebody;