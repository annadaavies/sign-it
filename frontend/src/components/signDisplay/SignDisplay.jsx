import React from "react";
import styles from "./SignDisplay.module.css";

const SignDisplay = ({ signs }) => {
  return (
    <div className={styles.signDisplay}>
      <div className={styles.translationContainer}>
        {signs.length > 0 ? (
          <div className={styles.signGrid}>
            {signs.map((sign, index) => (
              <div key={index} className={styles.signItem}>
                {sign.type === "image" ? (
                  <img
                    src={`/signs/${sign.value}`}
                    alt={sign.label}
                    className={styles.signImage}
                  />
                ) : (
                  <video
                    src={`/signs/${sign.value}`}
                    className={styles.signVideo}
                    autoPlay
                    loop
                    muted
                  />
                )}
                <span className={styles.signLabel}>{sign.label}</span>
              </div>
            ))}
          </div>
        ) : (
          <div className={styles.placeholder}>
            <p>Your translated signs will appear here</p>
            <small>Example: Try typing "Hello world"</small>
          </div>
        )}
      </div>
    </div>
  );
};

export default SignDisplay;
