import Webcam from "react-webcam";
import { useRef, useEffect } from "react";
import { predictLetter } from "../services/api";
import styles from "../components/cameraFeed/CameraFeed.module.css";

const CameraFeed = ({ onPrediction }) => {
  const webcameRef = useRef(null);
  const intervalRef = useRef();

  const capture = async () => {
    const imageSrc = webcamRef.current?.getScreenshot();
    if (imageSrc) {
      try {
        const letter = await predictLetter(imageSrc);
        onPrediction(letter);
      } catch (error) {
        console.error("Prediction error:", error);
      }
    }
  };

  useEffect(() => {
    intervalRef.current = setInterval(capture, 2000);
    {
      /* Capture every two seconds */
    }

    return () => clearInterval(intervalRef.current);
  }, []);

  return (
    <div className={styles.cameraContainer}>
      <Webcam
        audio={false}
        ref={webcamRef}
        screenshotFormat="image/jpeg"
        className={styles.camera}
        videoConstraints={{
          facingMode: "user",
          width: 1280,
          height: 720,
        }}
      />
      <div className={styles.overlay}></div>
    </div>
  );
};
