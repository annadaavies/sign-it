import os
import cv2
import numpy
import tensorflow
import mediapipe
import base64
from backend.config import MODEL_PATH, ASL_LABELS, BUFFER_SIZE, CONSENSUS_THRESHOLD, PROCESSING_SIZE

prediction_buffer = []

media_pipe_hands = mediapipe.solutions.hands

hands_detector = media_pipe_hands.Hands(static_image_mode = False, max_num_hands = 1, min_detection_confidence = 0.7)

model = tensorflow.keras.models.load_model(MODEL_PATH)

def preprocess_bgr_frame(frame_bgr: numpy.ndarray) -> numpy.ndarray:
    """
    Detect a single hand within a BGR frame, and preprocess it in the following manner: 
    - Extract and resize its bounding box
    - Centre hand on a black 200 x 200 background 
    - Convert final result to grayscale 
    
    Return None if no hand is found in the frame. 
    
    Args: 
    frame_bgr: Input frame in BGR colour format. 
    
    Returns: 
    numpy.ndarray | None: A 200 x 200 x 3 (3 channels) preprocessed image (if a hand is found in frame), otherwise None. 
    """ 
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    
    detection_results = hands_detector.process(frame_rgb)
    if not detection_results.multi_hand_landmarks: 
        return None
    
    frame_height, frame_width, _ = frame_bgr.shape
    
    hand_mask = numpy.zeros((frame_height, frame_width), dtype=numpy.uint8) #This creates an empty mask to fill with what will be the detected hand region. 
    
    for hand_landmarks in detection_results.multi_hand_landmarks: 
        landmark_points = []
        for landmark in hand_landmarks.landmark: 
            x_coord = int(landmark.x * frame_width) 
            y_coord = int(landmark.y * frame_height) 
            landmark_points.append((x_coord, y_coord)) 
        cv2.fillPoly(hand_mask, [numpy.array(landmark_points, dtype=numpy.int32)], 255) #This fills the empty mask with the polygon described by the detected hand landmarks. 
        
    dilation_kernel = numpy.ones((5,5), numpy.uint8)
    hand_mask_dilated = cv2.dilate(hand_mask, dilation_kernel, iterations=1)
    hand_mask_blurred = cv2.GaussianBlur(hand_mask_dilated, (21,21), 0) #This section uses cv2 to dilate and blur the hand frame to form a softer, more contiguous region  of the hand. 
    
    hand_only_image = numpy.where(hand_mask_blurred[..., None] > 0,
                                  frame_bgr,
                                  numpy.zeros_like(frame_bgr))
    
    contours, _ = cv2.findContours(hand_mask_blurred, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) 
    if not contours: 
        return None
    
    x_min, x_max, y_min, y_max = frame_width, 0, frame_height, 0
    for contour in contours: 
        x, y, rect_width, rect_height = cv2.boundingRect(contour) 
        x_min, y_min = min(x_min, x), min(y_min, y)
        x_max, y_max = max(x_max, x + rect_width), max(y_max, y + rect_height)
        
    hand_region= hand_only_image[y_min:y_max, x_min:x_max]
    hand_width, hand_height = x_max - x_min, y_max - y_min
    
    scale_factor = min(PROCESSING_SIZE / hand_width, PROCESSING_SIZE / hand_height) * 0.8
    scaled_width = int(hand_width * scale_factor) 
    scaled_height = int(hand_height * scale_factor) 
    resized_hand_region = cv2.resize(hand_region, (scaled_width, scaled_height), interpolation=cv2.INTER_AREA)
    
    centered_frame = numpy.zeros((PROCESSING_SIZE, PROCESSING_SIZE, 3), dtype=numpy.uint8)
    x_offset = (PROCESSING_SIZE - scaled_width) // 2
    y_offset = (PROCESSING_SIZE - scaled_height) // 2
    centered_frame[y_offset:y_offset + scaled_height, x_offset:x_offset + scaled_width] = resized_hand_region
    
    frame_grayscale = cv2.cvtColor(centered_frame, cv2.COLOR_BGR2GRAY) 
    expanded_frame = numpy.stack((frame_grayscale, frame_grayscale, frame_grayscale), axis=-1)
    
    return expanded_frame

def predict_letter_from_base64(image_b64: str) -> str: 
    """
    Decode a base64-encoded image of a ASL letter sign, preprocess with MediaPipe, feed it into Keras ASL model. 
    
    Implement rolling buffer of predictions to improve stability and transition time of hand. 
    
    Once stable, returns predicted letter. Until stable, predictions will continue to be added to buffer. 
    
    Returns "No Frame" if image frame can't be decoded. 
    
    Returns "No Hand" if no hand can be detected in frame.
    """
    
    global prediction_buffer
    
    try: 
        base64_data = image_b64.split(",")[1] #Strips the base64 prefix if present. 
    except IndexError: 
        return "No Frame"
    
    image_bytes = base64.b64decode(base64_data)
    
    image_array = numpy.frombuffer(image_bytes, numpy.uint8)
    frame_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR) 
    if frame_bgr is None: 
        return "No Frame"

    preprocessed_frame = preprocess_bgr_frame(frame_bgr) 
    if preprocessed_frame is None: 
        return "No Hand"
    
    preprocessed_frame = numpy.float32(preprocessed_frame) / 255.0
    preprocessed_frame = numpy.expand_dims(preprocessed_frame, axis=0)
    
    predictions = model.predict(preprocessed_frame)
    predicted_index = numpy.argmax(predictions) 
    
    prediction_buffer.append(predicted_index) 
    if len(prediction_buffer) > BUFFER_SIZE: 
        prediction_buffer.pop(0)
        
    if prediction_buffer.count(predicted_index) >= CONSENSUS_THRESHOLD:
        prediction_buffer = [] 
        return ASL_LABELS[predicted_index]
    else: 
        return "Stabilising..."




