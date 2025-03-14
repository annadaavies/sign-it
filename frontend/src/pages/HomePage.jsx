import React, { useState } from "react";
import CameraFeed from "../components/cameraFeed/CameraFeed";
import TranslationBox from "../components/translationBox/TranslationBox";
import SignDisplay from "../components/signDisplay/SignDisplay";
import styles from "../assets/styles/HomePage.module.css";

function HomePage({ mode }) {
  const [predictedLetters, setPredictedLetters] = useState([]);
  const [translatedSentence, setTranslatedSentence] = useState([]);
  const [signs, setSigns] = useState([]);

  const [clothingMode, setClothingMode] = useState(false);

  const toggleClothingMode = () => {
    setPredictedLetters([]);
    setTranslatedSentence([]);
    setSigns([]);
    setClothingMode((prev) => !prev);
  };

  if (mode === "aslToEnglish") {
    return (
      <div className={styles.homeContainer}>
        <button
          style={{
            background: clothingMode ? "#ffc107" : "#e8f0fe",
            color: clothingMode ? "#000" : "#1a73e8",
            padding: "0.75 rem 1.5rem",
            borderRadius: "8px",
            border: "none",
            marginBottom: "1rem",
            cursor: "pointer",
            fontWeight: "600",
          }}
          onClick={toggleClothingMode}
        >
          {clothingMode ? "Exit Interactive Mode" : "Interactive Mode"}
        </button>

        <CameraFeed
          predictedLetters={predictedLetters}
          setPredictedLetters={setPredictedLetters}
          translatedSentence={translatedSentence}
          setTranslatedSentence={setTranslatedSentence}
          clothingMode={clothingMode}
        />

        {!clothingMode && (
          <div className={styles.translatedSentenceBox}>
            <h3>Translated Sentence:</h3>
            <div className={styles.translatedSentence}>
              {translatedSentence.map((word, index) => (
                <span key={index} className={styles.translatedWord}>
                  {word}
                </span>
              ))}
            </div>
          </div>
        )}
      </div>
    );
  }

  return (
    <div className={styles.homeContainer}>
      <div className={styles.englishToAslContainer}>
        <TranslationBox mode={mode} onTranslate={setSigns} />
        <SignDisplay signs={signs} />
      </div>
    </div>
  );
}

export default HomePage;
