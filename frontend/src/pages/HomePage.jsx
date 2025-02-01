import { useState } from "react";
import CameraFeed from "../components/cameraFeed/CameraFeed";
import TranslationBox from "../components/translationBox/TranslationBox";
import SignDisplay from "../components/signDisplay/SignDisplay";
import styles from "../assets/styles/HomePage.module.css";

const HomePage = ({ mode }) => {
  const [predictedLetters, setPredictedLetters] = useState([]);
  const [sentence, setSentence] = useState([]);
  const [signs, setSigns] = useState([]);

  const handleAddWord = () => {
    const word = predictedLetters.join("");
    setSentence([...sentence, word]);
    setPredictedLetters([]);
  };

  const handleDelete = () => {
    setPredictedLetters((prev) => prev.slice(0, -1));
  };

  return (
    <div className={styles.homeContainer}>
      {mode == "aslToEnglish" ? (
        <>
          <div className={styles.cameraSection}>
            <CameraFeed
              onPrediction={(letter) =>
                setPredictedLetters([...predictedLetters, letter])
              }
            />
          </div>

          <div className={styles.translationSection}>
            <TranslationBox
              mode={mode}
              predictedText={predictedLetters.join("")}
              onAddWord={handleAddWord}
              onDelete={handleDelete}
              sentence={sentence}
            />
          </div>
        </>
      ) : (
        <div className={styles.englishToAslContainer}>
          <TranslationBox mode={mode} on Translate={setSigns} />
          <SignDisplay signs={signs} />
        </div>
      )}
    </div>
  );
};

export default HomePage;
