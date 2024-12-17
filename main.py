import os
import cv2
import numpy as np
from tensorflow.keras import layers, models
from sklearn.model_selection import train_test_split
import mediapipe as mp
import time
# Define dataset folder path

DATASET_FOLDER = r"C:\Users\bassim pc\Desktop\Eye diseases\datasets"  # Update with your dataset path
print(f"Dataset Directory: {DATASET_FOLDER}")
# Dynamically get disease names from folder names in the dataset directory
disease_names = sorted([d for d in os.listdir(DATASET_FOLDER) if os.path.isdir(os.path.join(DATASET_FOLDER, d))])
print(f"Diseases found: {disease_names}")
# Function to preprocess the dataset
def preprocess_dataset(dataset_dir, target_size=(128, 128)):
    data = []
    labels = []
    for disease_id, disease_folder in enumerate(disease_names):
        disease_folder_path = os.path.join(dataset_dir, disease_folder)
        if os.path.isdir(disease_folder_path):
            for img_file in os.listdir(disease_folder_path):
                if img_file.lower().endswith(('.png', '.jpg', '.jpeg')):
                    img_path = os.path.join(disease_folder_path, img_file)
                    image = cv2.imread(img_path)
                    if image is not None:
                        try:
                            resized_image = cv2.resize(image, target_size)
                            data.append(resized_image)
                            labels.append(disease_id)
                        except Exception as e:
                            print(f"Error resizing image {img_path}: {e}")
                    else:
                        print(f"Failed to load image: {img_path}")
    data = np.array(data) / 255.0  # Normalize pixel values
    labels = np.array(labels)
    return data, labels


# Load and preprocess the dataset
X, y = preprocess_dataset(DATASET_FOLDER)
print(f"Dataset loaded: {len(X)} images, {len(y)} labels.")

# Split dataset into train, validation, and test sets
X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
print(f"Train: {len(X_train)} images, Validation: {len(X_val)} images, Test: {len(X_test)} images.")
# Function to build the CNN model
def build_cnn_model(input_shape=(128, 128, 3), num_classes=len(disease_names)):
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    return model
# Build and summarize the model
model = build_cnn_model()
model.summary()

# Train the model
history = model.fit(X_train, y_train, epochs=5, batch_size=32, validation_data=(X_val, y_val))

# Evaluate the model on the test data
test_loss, test_accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Save the trained model
model_path = 'eye_disease_model.h5'
model.save(model_path)
print(f"Model saved at {model_path}.")

# Function to classify eye disease
def classify_eye_disease(image, model):
    input_image = cv2.resize(image, (128, 128)) / 255.0
    input_image = input_image.reshape(-1, 128, 128, 3)
    prediction = model.predict(input_image)
    disease_id = np.argmax(prediction)
    return disease_names[disease_id], prediction[0][disease_id]
# Real-time disease detection using webcam with Mediapipe
def real_time_detection(model):
    mp_face_mesh = mp.solutions.face_mesh
    mp_drawing = mp.solutions.drawing_utils
    mp_drawing_styles = mp.solutions.drawing_styles

    # Initialize FPS counter
    prev_frame_time = 0
    new_frame_time = 0
    
    # Constants for eye landmarks
    LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
    RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]

    cap = cv2.VideoCapture(0)
    print("Press 'q' to quit, 's' to save a screenshot")

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5) as face_mesh:

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture image from webcam.")
                break

            # Calculate FPS
            new_frame_time = time.time()
            fps = 1/(new_frame_time-prev_frame_time)
            prev_frame_time = new_frame_time
            fps = int(fps)

            # Convert the image to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rgb.flags.writeable = False
            results = face_mesh.process(frame_rgb)
            frame_rgb.flags.writeable = True
            
            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:
                    # Draw face mesh
                    mp_drawing.draw_landmarks(
                        image=frame,
                        landmark_list=face_landmarks,
                        connections=mp_face_mesh.FACEMESH_CONTOURS,
                        landmark_drawing_spec=None,
                        connection_drawing_spec=mp_drawing_styles.get_default_face_mesh_contours_style())

                    # Extract eye regions
                    h, w, _ = frame.shape
                    left_eye_points = [(int(landmark.x * w), int(landmark.y * h)) 
                                     for idx, landmark in enumerate(face_landmarks.landmark) 
                                     if idx in LEFT_EYE]
                    right_eye_points = [(int(landmark.x * w), int(landmark.y * h)) 
                                      for idx, landmark in enumerate(face_landmarks.landmark) 
                                      if idx in RIGHT_EYE]

                    # Get bounding boxes for eyes
                    left_eye_rect = cv2.boundingRect(np.array(left_eye_points))
                    right_eye_rect = cv2.boundingRect(np.array(right_eye_points))

                    # Extract and process each eye
                    for eye_rect, eye_label in [(left_eye_rect, "Left"), (right_eye_rect, "Right")]:
                        x, y, w, h = eye_rect
                        # Add padding to the eye region
                        padding = 10
                        x = max(0, x - padding)
                        y = max(0, y - padding)
                        w = min(frame.shape[1] - x, w + 2 * padding)
                        h = min(frame.shape[0] - y, h + 2 * padding)
                        
                        eye_region = frame[y:y+h, x:x+w]
                        
                        if eye_region.size > 0:
                            try:
                                disease, confidence = classify_eye_disease(eye_region, model)
                                # Draw rectangle around eye
                                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                                # Display results
                                cv2.putText(frame, f"{eye_label} Eye: {disease}", 
                                          (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 
                                          0.6, (0, 255, 0), 2)
                                cv2.putText(frame, f"Conf: {confidence*100:.1f}%", 
                                          (x, y + h + 20), cv2.FONT_HERSHEY_SIMPLEX, 
                                          0.6, (0, 255, 0), 2)
                            except Exception as e:
                                print(f"Error processing {eye_label} eye: {str(e)}")

            # Display FPS
            cv2.putText(frame, f"FPS: {fps}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                       1, (0, 255, 0), 2)

            cv2.imshow('Eye Disease Detection', frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                # Save screenshot
                timestamp = time.strftime("%Y%m%d-%H%M%S")
                cv2.imwrite(f"eye_detection_{timestamp}.jpg", frame)
                print(f"Screenshot saved as eye_detection_{timestamp}.jpg")

    cap.release()
    cv2.destroyAllWindows()


# Function to detect eye disease in a static image
def detect_from_image(image_path, model):
    mp_face_detection = mp.solutions.face_detection
    mp_drawing = mp.solutions.drawing_utils

    image = cv2.imread(image_path)
    if image is None:
        print("Error: Unable to load the image.")
        return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        results = face_detection.process(image_rgb)

        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                ih, iw, _ = image.shape
                x, y, w, h = int(bboxC.xmin * iw), int(bboxC.ymin * ih), int(bboxC.width * iw), int(bboxC.height * ih)

                # Crop the eye region
                eye_region = image[y:y + h, x:x + w]

                if eye_region.size > 0:
                    disease, confidence = classify_eye_disease(eye_region, model)

                    # Draw bounding box and results
                    mp_drawing.draw_detection(image, detection)
                    cv2.putText(image, f"{disease}: {confidence * 100:.2f}%", (x, y - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow('Static Image Detection', image)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

# Example usage for static image detection
# Replace 'example_image.jpg' with the path to your image
# Construct path dynamically
base_folder = "Eye diseases/datasets/cataract"
image_name = "image0.png"
image_path = os.path.join(base_folder, image_name)

# Use the constructed path
# detect_from_image(image_path, model)
# Run real-time detection
real_time_detection(model)
