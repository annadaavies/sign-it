import axios from "axios";

const API_BASE = "http://localhost:5000/api";

//This is defined as a standard function rather than an export function as it is only intended for internal use within the module.
function parseDictionaryEntry(entryArray) {
  const parsedObject = {};
  entryArray.forEach((item) => {
    parsedObject[item.key] = item.value;
  });
  return parsedObject;
}

//The export const functions are 'arrow' functions and designed to be exported immediately. They make up the module's public API for other parts of the application.
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
    console.error("Letter prediction API error:", error);
    return [];
  }
};

export const predictClothing = async (base64Image) => {
  try {
    const response = await axios.post(`${API_BASE}/predictClothing`, {
      image: base64Image,
    });
    return response.data.item;
  } catch (error) {
    console.error("Clothing prediction API error:", error);
    return [];
  }
};
