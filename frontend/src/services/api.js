import axios from "axios";

const API_BASE = "http://localhost:5000/api";

export const predictLetter = async (imageData) => {
  try {
    const blob = await fetch(imageData).then((res) => res.blob());

    const formData = new FormData();
    formData.append("image", blob, "frame.jpg");

    const response = await axios.post(`${API_BASE}/predict`, formData, {
      headers: {
        "Content-Type": "multipart/form-data",
      },
    });

    return response.data.letter;
  } catch (error) {
    console.error("Prediction API error", error);
    return null;
  }
};

export const translateText = async (text) => {
  try {
    const response = await axios.post(`${API_BASE}/translate`, { text });
    return response.data.signs;
  } catch (error) {
    console.error("Translation API error:", error);
    return [];
  }
};
