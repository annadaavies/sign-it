import React, { useRef, useEffect, useState, useCallback } from "react";
import Webcam from "react-webcam";
import { predictLetter, predictClothing } from "../../services/api";
import styles from "./CameraFeed.module.css";

function CameraFeed({
  predictedLetters = [],
  setPredictedLetters = () => {},
  translatedSentence = [],
  setTranslatedSentence = () => {},
  clothingMode = false,
}) {
  const webcamRef = useRef(null);
  const [isPredicting, setIsPredicting] = useState(true);

  //For letter predictions
  const [currentLetter, setCurrentLetter] = useState("");

  //For clothing predictions
  const [predictedClothing, setPredictedClothing] = useState("");
  const [showClothingMessage, setShowClothingMessage] = useState(false);

  //For feedback to user
  const [feedbackMessage, setFeedbackMessage] = useState("");
  const [showFeedback, setShowFeedback] = useState(false);

  //Track whether clothing capture has happened at least once
  const [clothingCaptured, setClothingCaptured] = useState(false);

  /**
   * Poll for letter predictions every 0.5s if:
   *   - isPredicting is true,
   *   - AND (we're not in clothingMode) OR (we are in clothingMode AND clothingCaptured===true).
   */
  useEffect(() => {
    let intervalId;
    if (isPredicting && (!clothingMode || clothingCaptured)) {
      intervalId = setInterval(async () => {
        await captureImageAndPredict();
      }, 500);
    }
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [isPredicting, clothingMode, clothingCaptured]);

  const captureImageAndPredict = useCallback(async () => {
    if (!webcamRef.current) return;
    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) return;

    try {
      const letter = await predictLetter(imageSrc);
      if (letter && letter.length === 1 && letter.match(/[A-Z]/)) {
        setCurrentLetter(letter);
        setPredictedLetters((prev) => [...prev, letter]);
      } else {
        setCurrentLetter(letter);
      }
    } catch (error) {
      console.error("Prediction error:", error);
    }
  }, [setPredictedLetters]);

  const handleDeleteLastLetter = () => {
    setPredictedLetters((prev) => prev.slice(0, -1));
  };

  const handleAddWord = () => {
    if (predictedLetters.length === 0) return;
    const newWord = predictedLetters.join("");
    setTranslatedSentence((prev) => [...prev, newWord]);
    setPredictedLetters([]);
  };

  const togglePredicting = () => {
    setIsPredicting((prev) => !prev);
  };

  /**
   * Capture clothing item:
   * - if a valid item is returned, set it as predictedClothing,
   * - show the overlay,
   * - and enable letter polling by setting clothingCaptured = true (and isPredicting = true).
   * If the user wants to retake, they can press this again and override the item.
   */
  const handleCaptureClothing = async () => {
    if (!webcamRef.current) return;
    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) return;

    try {
      const item = await predictClothing(imageSrc);
      console.log("Clothing item predicted:", item);
      setPredictedClothing(item);

      if (item && item !== "No Frame") {
        setShowClothingMessage(true);
        setClothingCaptured(true);
        setIsPredicting(true);

        setTimeout(() => {
          setShowClothingMessage(false);
        }, 3000);
      }
    } catch (error) {
      console.error("Clothing capture error:", error);
    }
  };

  const handleCheckClothingSign = () => {
    const spelledWord = predictedLetters.join("").toLowerCase();

    if (!predictedClothing) {
      setFeedbackMessage("No clothing item captured yet!");
      setShowFeedback(true);
      setTimeout(() => setShowFeedback(false), 3000);
      return;
    }

    if (spelledWord === predictedClothing.toLowerCase()) {
      setFeedbackMessage("Well done -- that's the right sign.");
    } else {
      setFeedbackMessage(
        `That's not the right sign. (You spelled "${spelledWord}" but item is "${predictedClothing}")`
      );
    }

    setShowFeedback(true);
    setTimeout(() => setShowFeedback(false), 3000);
    setPredictedLetters([]);
  };

  return (
    <div className={styles.container}>
      <div className={styles.webcamWrapper}>
        <Webcam
          audio={false}
          ref={webcamRef}
          screenshotFormat="image/jpeg"
          className={styles.webcam}
          videoConstraints={{
            facingMode: "user",
          }}
        />

        <div className={styles.dashedOverlay}></div>

        {clothingMode && (
          <button
            className={styles.captureButton}
            onClick={handleCaptureClothing}
          >
            Capture Clothing
          </button>
        )}
      </div>

      <div className={styles.bottomBar}>
        <div className={styles.predictedLettersBox}>
          {predictedLetters.length === 0 ? (
            <span className={styles.placeholderText}>
              Letters will appear...
            </span>
          ) : (
            <span className={styles.predictedLetters}>
              {predictedLetters.join("")}
            </span>
          )}
        </div>

        <div className={styles.buttonGroup}>
          <button
            className={styles.deleteButton}
            onClick={handleDeleteLastLetter}
            disabled={predictedLetters.length === 0}
          >
            Delete
          </button>

          {!clothingMode && (
            <button
              className={styles.addWordButton}
              onClick={handleAddWord}
              disabled={predictedLetters.length === 0}
            >
              Add Word
            </button>
          )}

          {clothingMode ? (
            <button
              className={styles.addWordButton}
              onClick={handleCheckClothingSign}
            >
              Enter
            </button>
          ) : (
            <button className={styles.addWordButton} onClick={togglePredicting}>
              {isPredicting ? "Stop" : "Start"} Predicting
            </button>
          )}
        </div>
      </div>

      <p>Detected: {currentLetter}</p>

      {showClothingMessage && (
        <div className={styles.clothingOverlay}>
          That is a {predictedClothing}. Now, sign it!
        </div>
      )}

      {showFeedback && (
        <div className={styles.feedbackOverlay}>{feedbackMessage}</div>
      )}
    </div>
  );
}

export default CameraFeed;
