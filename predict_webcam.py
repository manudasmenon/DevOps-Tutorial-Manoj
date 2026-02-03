"""
Glasses vs No Glasses - Real-Time Webcam Prediction
====================================================
Uses your trained CNN model to predict in real-time whether
the person in the webcam has glasses or not.

Make sure to run train_model.py first to create the model!
"""

import os
import cv2
import numpy as np
from tensorflow import keras

# ============ CONFIGURATION ============
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(SCRIPT_DIR, "glasses_model.keras")
CLASS_NAMES_PATH = os.path.join(SCRIPT_DIR, "class_names.txt")
IMAGE_SIZE = (128, 128)  # Must match training!
# ======================================


def load_model_and_classes():
    """Load the trained model and class names."""
    if not os.path.exists(MODEL_PATH):
        print("❌ Error: Model not found!")
        print(f"   Run train_model.py first to train the model.")
        print(f"   Expected path: {MODEL_PATH}")
        return None, None
    
    print("📂 Loading model...")
    model = keras.models.load_model(MODEL_PATH)
    
    # Load class names
    if os.path.exists(CLASS_NAMES_PATH):
        with open(CLASS_NAMES_PATH, "r") as f:
            class_names = [line.strip() for line in f.readlines()]
    else:
        class_names = ["no_glasses", "glasses"]  # Default fallback
    
    print(f"   Classes: {class_names}")
    return model, class_names


def load_face_detector():
    """Load OpenCV's face detector (comes with OpenCV)."""
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    return cv2.CascadeClassifier(cascade_path)


def detect_face(frame, face_cascade):
    """Detect face in frame. Returns face region or None."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(60, 60))
    if len(faces) > 0:
        x, y, w, h = faces[0]  # Use largest/first face
        # Add padding
        pad = int(0.2 * min(w, h))
        x1 = max(0, x - pad)
        y1 = max(0, y - pad)
        x2 = min(frame.shape[1], x + w + pad)
        y2 = min(frame.shape[0], y + h + pad)
        return frame[y1:y2, x1:x2], (x1, y1, x2, y2)
    return None, None


def preprocess_frame(face_img):
    """
    Prepare the face image for the model.
    Resize and normalize to match training.
    """
    # Resize to model's expected input size
    resized = cv2.resize(face_img, IMAGE_SIZE)
    # Convert BGR (OpenCV) to RGB (model expects)
    rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
    # Keep pixel values 0-255 - model's Rescaling layer normalizes (same as training!)
    rgb_float = rgb.astype(np.float32)
    # Add batch dimension: (128, 128, 3) -> (1, 128, 128, 3)
    batch = np.expand_dims(rgb_float, axis=0)
    return batch


def main():
    print("=" * 50)
    print("  Glasses vs No Glasses - Webcam Prediction")
    print("=" * 50)
    
    # Load model
    model, class_names = load_model_and_classes()
    if model is None:
        return
    
    # Open webcam (0 = default camera)
    print("\n📷 Opening webcam...")
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam!")
        return
    
    # Load face detector for better accuracy
    face_cascade = load_face_detector()
    
    print("\n🎯 Instructions:")
    print("   - Position your face in front of the camera")
    print("   - Face will be auto-detected for prediction")
    print("   - Press 'Q' to quit")
    print("=" * 50)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("Could not read from webcam")
            break
        
        # Detect face - use face crop for better prediction
        face_roi, face_box = detect_face(frame, face_cascade)
        
        if face_roi is not None:
            # Preprocess and predict on face
            processed = preprocess_frame(face_roi)
            predictions = model.predict(processed, verbose=1)
            predicted_class_idx = np.argmax(predictions[0])
            confidence = predictions[0][predicted_class_idx]
            label = class_names[predicted_class_idx]
            
            # Draw face rectangle
            x1, y1, x2, y2 = face_box
            color = (0, 255, 0) if label == "glasses" else (0, 165, 255)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        else:
            label = "No face detected"
            confidence = 0
        
        # Display result on frame
        display_text = f"{label}: {confidence*100:.1f}%" if face_roi is not None else label
        color = (0, 255, 0) if label == "glasses" else (0, 165, 255) if label == "no_glasses" else (200, 200, 200)
        
        # Draw text background for readability
        (text_width, text_height), _ = cv2.getTextSize(
            display_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2
        )
        cv2.rectangle(frame, (10, 10), (20 + text_width, 40 + text_height), (0, 0, 0), -1)
        cv2.putText(
            frame, display_text, (15, 40),
            cv2.FONT_HERSHEY_SIMPLEX, 1, color, 2
        )
        
        # Show frame
        cv2.imshow("Glasses vs No Glasses", frame)
        
        # Quit on 'Q' press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    print("\n👋 Webcam closed. Goodbye!")


if __name__ == "__main__":
    main()
