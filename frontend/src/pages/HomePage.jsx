import React, { useState } from "react";
import CameraFeed from "../components/cameraFeed/CameraFeed";
import TranslationBox from "../components/translationBox/TranslationBox";
import SignDisplay from "../components/signDisplay/SignDisplay";
import styles from "../assets/styles/HomePage.module.css";

function HomePage({ mode }) {
  const [predictedLetters, setPredictedLetters] = useState([]);
  const [translatedSentence, setTranslatedSentence] = useState([]);
  const [signs, setSigns] = useState([]);

  return (
    <div className={styles.homeContainer}>
      {mode === "aslToEnglish" ? (
        <div className={styles.aslToEnglishContainer}>
          <CameraFeed
            predictedLetters={predictedLetters}
            setPredictedLetters={setPredictedLetters}
            translatedSentence={translatedSentence}
            setTranslatedSentence={setTranslatedSentence}
          />

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
        </div>
      ) : (
        <div className={styles.englishToAslContainer}>
          <TranslationBox mode={mode} onTranslate={setSigns} />
          <SignDisplay signs={signs} />
        </div>
      )}
    </div>
  );
}

export default HomePage;
