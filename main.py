import numpy as np
import dlib
import joblib
import mediapipe as mp
import time
import os
from datetime import datetime

#memory, procesing power
# final one

# Create 'Proofs' folder if it doesn't exist
if not os.path.exists("Proofs"):
    os.makedirs("Proofs")

# Load models
knn = joblib.load('knn_model.pkl')
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')

# Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7, min_tracking_confidence=0.7)

# Background subtractor for blob detection
fgbg = cv2.createBackgroundSubtractorMOG2()

# Load video
# cap = cv2.VideoCapture("mno.mp4")     #OUT SIDE THE BIN
# cap = cv2.VideoCapture("mno2.mp4")      #INSIDE THE BIN
cap = cv2.VideoCapture("both.mp4")      #BOTH INSIDE AND OUTSIDE THE BIN


frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))

# Adjusted ground level
ground_level = int(frame_height * 0.75)

# Dictionary to track previous y-positions of detected objects
object_positions = {}

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    timestamp = time.time()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_frame)

    # Face detection
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = dlib.get_frontal_face_detector()(gray)
    detected_faces = {}

    for rect in faces:
        shape = shape_predictor(gray, rect)
        embedding = np.array(face_rec_model.compute_face_descriptor(frame, shape))
        prediction = knn.predict([embedding])
        label = prediction[0] if prediction else "Unknown"
        detected_faces[label] = (rect.left(), rect.top(), rect.right(), rect.bottom())

        cv2.rectangle(frame, (rect.left(), rect.top()), (rect.right(), rect.bottom()), (0, 255, 0), 2)
        cv2.putText(frame, label, (rect.left(), rect.top() - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 0, 0), 2)

    # Hand detection
    hand_positions = []
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            x_avg = int(np.mean([landmark.x * frame.shape[1] for landmark in hand_landmarks.landmark]))
            y_avg = int(np.mean([landmark.y * frame.shape[0] for landmark in hand_landmarks.landmark]))
            hand_positions.append((x_avg, y_avg))
            cv2.circle(frame, (x_avg, y_avg), 10, (255, 0, 0), -1)

    # Background subtraction for falling object detection
    fgmask = fgbg.apply(frame)
    contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    for i, contour in enumerate(contours):
        if cv2.contourArea(contour) < 1000:
            continue

        x, y, w, h = cv2.boundingRect(contour)
        centroid = (x + w // 2, y + h // 2)

        # Store and compare previous y-position
        prev_y = object_positions.get(i, None)
        object_positions[i] = y  # Update current position

        # Check if the object is falling
        if prev_y is not None and y > prev_y and (y + h) > ground_level:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Falling Object", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
            print("WARNING: Object has fallen!")

            # Take screenshot with person's name, date, and time
            person_name = "Unknown"
            if detected_faces:
                person_name = list(detected_faces.keys())[0]  # Take the first detected person

            current_time = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
            screenshot_filename = f"Proofs/{person_name}_{current_time}_fall_detected.png"

            cv2.imwrite(screenshot_filename, frame)
            print(f"Screenshot saved as {screenshot_filename}")

    # Draw ground level line
    cv2.line(frame, (0, ground_level), (frame_width, ground_level), (0, 0, 255), 2)

    cv2.imshow('TrashBot System', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
