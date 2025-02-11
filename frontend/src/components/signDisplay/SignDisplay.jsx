import React, { useState, useEffect } from "react";
import styles from "./SignDisplay.module.css";

function SignDisplay({ signs }) {
  const [currentIndex, setCurrentIndex] = useState(0);

  useEffect(() => {
    if (!signs || signs.legnth === 0)
      return; /*If no signs or out of range, do nothing.*/
    if ((currentIndex < 0) | (currentIndex >= signs.length)) return;

    const currentSign = signs[currentIndex];
    if (!currentSign) return;

    let duration = 2.0;
    if (currentSign.type === "video") {
      duration = currentSign.duration || 3.0;
    } else if (currentSign.type === "image") {
      duration = currentSign.duration || 2.0;
    }

    const timer = setTimeout(() => {
      setCurrentIndex((prev) => prev + 1);
    }, duration * 1000);

    return () => clearTimeout(timer);
  }, [signs, currentIndex]);

  if (!signs || signs.length === 0) {
    return (
      <div className={styles.signDisplay}>
        <div className={styles.placeholder}>
          <p>Your translated signs will appear here.</p>
          <small>Try typing "hello"...</small>
        </div>
      </div>
    );
  }

  if (currentIndex >= signs.length) {
    return (
      <div className={styles.signDisplay}>
        <div className={styles.placeholder}>
          <p>All signs displayed.</p>
        </div>
      </div>
    );
  }

  const currentSign = signs[currentIndex];

  return (
    <div className={styles.signDisplay}>
      <div className={styles.singleSignContainer}>
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
      </div>
      <div className={styles.signLabel}>{currentSign.signLabel}</div>
    </div>
  );
}

export default SignDisplay;
