import React from "react";
import { Link } from "react-router-dom";
import styles from "../assets/styles/HomePage.module.css";

function HomePage() {
  return (
    <div className={styles.homeContainer}>
      <div className={styles.leftSection}>
        <h1 className={styles.title}>SignIt</h1>
        <p className={styles.description}>
          The bidirectional sign language translation app.
        </p>
        <div className={styles.buttonGroup}>
          <Link to="/asl-to-english">
            <button className={styles.selectionButton}>ASL → English</button>
          </Link>
          <Link to="/english-to-asl">
            <button className={styles.selectionButton}>English → ASL</button>
          </Link>
        </div>
      </div>
      <div className={styles.rightSection}>
        <img
          src="/curved-hand.png"
          alt="Curved Hand Artwork"
          className={styles.artwork}
        />
      </div>
    </div>
  );
}

export default HomePage;
