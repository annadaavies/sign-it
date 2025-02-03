import { useState } from "react";
import { translateText } from "../../services/api";
import styles from "./TranslationBox.module.css";

const TranslationBox = ({
  mode,
  predictedText,
  onAddWord,
  onDelete,
  sentence,
  onTranslate,
}) => {
  const [inputText, setInputText] = useState("");

  const handleSubmit = async (e) => {
    e.preventDefault();
    if (inputText.trim()) {
      try {
        const translatedSigns = await translateText(inputText);

        onTranslate(translatedSigns);
      } catch (error) {
        console.error("Translation error:", error);
      }
    }
  };

  return mode == "aslToEnglish" ? (
    <div className={styles.translationBox}>
      <div className={styles.predictionArea}>
        <div className={styles.wordBox}>
          <span className={styles.predictedtext}>{predictedText}</span>
          <button
            className={styles.deleteButton}
            onClick={onDelete}
            disabled={!predictedText.length}
          >
            ⌫
          </button>
        </div>
        <button
          className={styles.addButton}
          onClick={onAddWord}
          disabled={!predictedText.length}
        >
          Add Word
        </button>
      </div>

      <div className={styles.sentenceArea}>
        <h3>Constructed Sentence:</h3>
        <div className={styles.sentence}>
          {sentence.map((word, index) => (
            <span key={index} className={styles.sentenceWord}>
              {word}
            </span>
          ))}
        </div>
      </div>
    </div>
  ) : (
    <div className={styles.englishInput}>
      <form onSubmit={handleSubmit}>
        <input
          type="text"
          value={inputText}
          onChange={(e) => setInputText(e.target.value)}
          placeholder="Enter English text to translate..."
          className={styles.textInput}
        />
        <button type="submit" className={styles.translateButton}>
          Translate to ASL
        </button>
      </form>
    </div>
  );
};

export default TranslationBox;
