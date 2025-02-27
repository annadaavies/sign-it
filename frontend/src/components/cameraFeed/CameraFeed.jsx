import React, { useRef, useEffect, useState, useCallback } from "react";
import Webcam from "react-webcam";
import { predictLetter } from "../../services/api";
import styles from "./CameraFeed.module.css";

function CameraFeed({
  predictedLetters = [],
  setPredictedLetters = () => {},
  translatedSentence = [],
  setTranslatedSentence = () => {},
}) {
  const webcamRef = useRef(null);
  const [isPredicting, setIsPredicting] = useState(true);
  const [currentLetter, setCurrentLetter] = useState("");

  useEffect(() => {
    let intervalId;
    if (isPredicting) {
      intervalId = setInterval(async () => {
        await captureImageAndPredict();
      }, 500); // Polls every 0.5 seconds
    }
    return () => {
      if (intervalId) clearInterval(intervalId);
    };
  }, [isPredicting]);

  const captureImageAndPredict = useCallback(async () => {
    if (!webcamRef.current) return;
    const imageSrc = webcamRef.current.getScreenshot();
    if (!imageSrc) return;

    try {
      const letter = await predictLetter(imageSrc);
      if (letter && letter.length === 1 && letter.match(/[A-Z]/)) {
        setCurrentLetter(letter);
        setPredictedLetters((prev) => [...prev, letter]);
      } else if (letter === "Stabilizing...") {
        setCurrentLetter(letter);
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
      </div>

      <div className={styles.bottomBar}>
        <div className={styles.predictedLettersBox}>
          {predictedLetters.length === 0 ? (
            <span className={styles.placeholderText}>
              Letters will appear here...
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
          <button
            className={styles.addWordButton}
            onClick={handleAddWord}
            disabled={predictedLetters.length === 0}
          >
            Add Word
          </button>
          <button className={styles.addWordButton} onClick={togglePredicting}>
            {isPredicting ? "Stop" : "Start"} Predicting
          </button>
        </div>
      </div>

      {<p>Detected: {currentLetter}</p>}
    </div>
  );
}

export default CameraFeed;
