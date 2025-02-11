import axios from "axios";

/**
 * Helper function to convert a serialised "Dictionary" entry
 * into an object with { type, label, value, ... } keys.
 */
function parseDictionaryEntry(entryArray) {
  // entryArray is something like:
  // [
  //   { key: 'type', value: 'video' },
  //   { key: 'label', value: 'HELLO' },
  //   { key: 'value', value: 'signs/hello.mp4' },
  //   ...
  // ]
  const parsedObject = {};
  entryArray.forEach((item) => {
    parsedObject[item.key] = item.value;
  });
  return parsedObject;
}

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
    /**
     * response.data.signs is an array of objects:
     *   [{ key: '0', value: [...Dictionary entries...] },
     *    { key: '1', value: [...Dictionary entries...] },
     *    ... ]
     *
     * We need to map each "value" array into a normal { type, label, value, … } object
     */
    const parsedSigns = response.data.signs.map((item) => {
      return parseDictionaryEntry(item.value);
    });

    return parsedSigns; // e.g. [{type: "video", label: "HELLO", value: "signs/hello.mp4"}, ...]
  } catch (error) {
    console.error("Translation API error:", error);
    return [];
  }
};
