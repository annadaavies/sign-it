import React, { useState, useEffect } from "react";
import styles from "./SignDisplay.module.css";

function SignDisplay({ signs = [] }) {
  const [currentIndex, setCurrentIndex] = useState(0);
  const [hasFinished, setHasFinished] = useState(false);

  useEffect(() => {
    if (!signs || signs.length === 0) {
      setHasFinished(false);
      setCurrentIndex(0);
      return;
    }

    if (currentIndex >= signs.length) {
      setHasFinished(true);
      return;
    }

    const currentSign = signs[currentIndex];

    if (!currentSign) {
      setCurrentIndex((prev) => prev + 1);
      return;
    }

    let duration = 2.0;

    if (currentSign.type === "video") {
      duration = currentSign.duration || 3.0;
    } else if (currentSign.type === "image") {
      duration = currentSign.duration || 2.0;
    } else if (currentSign.type === "pause") {
      duration = currentSign.duration || 0.5;
    }

    const timer = setTimeout(() => {
      setCurrentIndex((prev) => prev + 1);
    }, duration * 1000);

    return () => clearTimeout(timer);
  }, [signs, currentIndex]);

  const handleReplay = () => {
    setHasFinished(false);
    setCurrentIndex(0);
  };

  if (!signs || signs.length === 0) {
    return (
      <div className={styles.signDisplay}>
        <div className={styles.translationContainer}>
          <div className={styles.placeholder}>
            <p>Your translated signs will appear here.</p>
          </div>
        </div>
      </div>
    );
  }

  if (hasFinished) {
    return (
      <div className={styles.signDisplay}>
        <div className={styles.translationContainer}>
          <div className={styles.placeholder}>
            <p>All signs displayed.</p>
          </div>
          <button className={styles.replayButton} onClick={handleReplay}>
            Replay
          </button>
        </div>
      </div>
    );
  }

  const currentSign = signs[currentIndex];

  if (!currentSign) {
    return null;
  }

  return (
    <div className={styles.signDisplay}>
      <div className={styles.translationContainer}>
        <div className={styles.mediaWrapper}>
          {currentSign.type === "image" && (
            <img
              src={`/signs/${currentSign.value}`}
              alt={currentSign.signLabel}
              className={styles.signMedia}
            />
          )}
          {currentSign.type === "video" && (
            <video
              src={`/signs/${currentSign.value}`}
              className={styles.signMedia}
              autoPlay
              muted
            />
          )}
          {currentSign.type === "pause" && (
            <div className={styles.placeholder}>
              <p>Pause...</p>
            </div>
          )}
        </div>
        <div className={styles.signLabel}>
          {currentSign.signLabel || currentSign.label}
        </div>

        <button className={styles.replayButton} onClick={handleReplay}>
          Replay
        </button>
      </div>
    </div>
  );
}

export default SignDisplay;
