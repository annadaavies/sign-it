import axios from "axios";

const API_BASE = "http://localhost:5000/api";

function parseDictionaryEntry(entryArray) {
  const parsedObject = {};
  entryArray.forEach((item) => {
    parsedObject[item.key] = item.value;
  });
  return parsedObject;
}

export const translateText = async (text) => {
  try {
    const response = await axios.post(`${API_BASE}/translateText`, { text });
    const parsedSigns = response.data.signs.map((item) => {
      return parseDictionaryEntry(item.value);
    });

    return parsedSigns; // e.g. [{type: "video", label: "HELLO", value: "signs/hello.mp4"}, ...]
  } catch (error) {
    console.error("Translation API error:", error);
    return [];
  }
};

export const predictLetter = async (base64Image) => {
  try {
    const response = await axios.post(`${API_BASE}/predictLetter`, {
      image: base64Image,
    });
    return response.data.letter;
  } catch (error) {
    console.error("Prediction API error:", error);
    return [];
  }
};
