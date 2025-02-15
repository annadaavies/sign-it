import React, { useRef, useCallback } from "react";
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

  const captureImage = useCallback(async () => {
    if (!webcamRef.current) return;
    const imageSrc = webcamRef.current.getScreenshot();
    if (imageSrc) {
      try {
        const letter = await predictLetter(imageSrc);
        if (letter) {
          setPredictedLetters((prev) => [...prev, letter]);
        }
      } catch (error) {
        console.error("Prediction error:", error);
      }
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

        <button className={styles.captureButton} onClick={captureImage}>
          Capture
        </button>
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
        </div>
      </div>
    </div>
  );
}

export default CameraFeed;
