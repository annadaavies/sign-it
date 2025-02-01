import styles from "../components/cameraFeed/CameraFeed.module.css";

const SignDisplay = ({ signs }) => {
  return (
    <div className={styles.signDisplay}>
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
                  src={`/signs/${signs.value}`}
                  className={styles.signVideo}
                  autoPlay
                  loop
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
  );
};

export default SignDisplay;
