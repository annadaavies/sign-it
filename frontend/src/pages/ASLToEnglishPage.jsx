import React, { useState } from "react";
import CameraFeed from "../components/cameraFeed/CameraFeed";
import styles from "../assets/styles/ASLToEnglish.module.css";

function ASLToEnglishPage() {
  const [predictedLetters, setPredictedLetters] = useState([]);
  const [translatedSentence, setTranslatedSentence] = useState([]);

  const [clothingMode, setClothingMode] = useState(false);

  const toggleInteractiveMode = () => {
    setPredictedLetters([]);
    setTranslatedSentence([]);
    setClothingMode((prev) => !prev);
  };

  return (
    <div className={styles.pageContainer}>
      <h2 className={styles.sectionTitle}>ASL to English Translation</h2>
      <div className={styles.toggleContainer}>
        <button
          className={`${styles.toggleButton} ${
            clothingMode ? styles.active : ""
          }`}
          onClick={toggleInteractiveMode}
        >
          {clothingMode ? "Exit Interactive Mode" : "Enter Interactive Mode"}
        </button>
      </div>
      <div className={styles.contentArea}>
        <CameraFeed
          predictedLetters={predictedLetters}
          setPredictedLetters={setPredictedLetters}
          translatedSentence={translatedSentence}
          setTranslatedSentence={setTranslatedSentence}
          clothingMode={clothingMode}
        />
        <div className={styles.translationResult}>
          <h3>Translated Sentence:</h3>
          <p>
            {translatedSentence.length > 0
              ? translatedSentence.join(" ")
              : "No translation yet."}
          </p>
        </div>
      </div>
    </div>
  );
}

export default ASLToEnglishPage;
