import React, { useState } from "react";
import TranslationBox from "../components/translationBox/TranslationBox";
import SignDisplay from "../components/signDisplay/SignDisplay";
import styles from "../assets/styles/EnglishToASL.module.css";

function EnglishToASLPage() {
  const [signs, setSigns] = useState([]);

  return (
    <div className={styles.pageContainer}>
      <h2 className={styles.sectionTitle}>English to ASL Translation</h2>

      <div className={styles.displaySection}>
        <SignDisplay signs={signs} />
      </div>

      <div className={styles.inputSection}>
        <TranslationBox mode="englishToAsl" onTranslate={setSigns} />
      </div>
    </div>
  );
}

export default EnglishToASLPage;
