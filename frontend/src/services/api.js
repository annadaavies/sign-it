import axios from "axios";

const API_BASE = "http://localhost:5000/api";

export const predictLetter = async (imageData) => {
  const formData = new FormData();
  const blob = await fetch(imageData).then((res) => res.blob());
  formData.append("image", blob, "frame.jpg");

  const response = await axios.post(`${API_BASE}/predict`, formData);
  return response.data.letter;
};

export const translateText = async (text) => {
  const response = await axios.post(`${API_BASE}/translate`, { text });
  return response.data.signs;
};
