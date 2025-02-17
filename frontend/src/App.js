import React, { useState } from "react";
import HomePage from "./pages/HomePage";
import styles from "./assets/styles/App.module.css";

function App() {
  const [mode, setMode] = useState("aslToEnglish");

  return (
    <div className={styles.appContainer}>
      <header className={styles.appHeader}>
        <img src="/logo.png" alt="SignIt" className={styles.logo} />

        <h1 className={styles.title}>SignIt</h1>
        <div className={styles.modeSwitcher}>
          <button
            className={`${styles.modeButton} ${
              mode === "aslToEnglish" ? styles.active : ""
            }`}
            onClick={() => setMode("aslToEnglish")}
          >
            ASL → English
          </button>

          <button
            className={`${styles.modeButton} ${
              mode === "englishToAsl" ? styles.active : ""
            }`}
            onClick={() => setMode("englishToAsl")}
          >
            English → ASL
          </button>
        </div>
      </header>
      <HomePage mode={mode} />
    </div>
  );
}

export default App;
