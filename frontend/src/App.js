import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import HomePage from "./pages/HomePage";
import ASLToEnglishPage from "./pages/ASLToEnglishPage";
import EnglishToASLPage from "./pages/EnglishToASLPage";
import NavigationBar from "./components/navigationBar/NavigationBar";
import styles from "./assets/styles/App.module.css";

function App() {
  return (
    <Router>
      <div className={styles.appContainer}>
        <NavigationBar />
        <div className={styles.content}>
          <Routes>
            <Route path="/" element={<HomePage />} />
            <Route path="/asl-to-english" element={<ASLToEnglishPage />} />
            <Route path="/english-to-asl" element={<EnglishToASLPage />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;
