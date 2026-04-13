import cv2
import numpy as np
import dlib
import joblib
import time
import os
import warnings
import threading
from datetime import datetime
import firebase_admin
from firebase_admin import credentials, db
from mailAutomation import send_mail

warnings.filterwarnings("ignore", category=UserWarning)

# ----- Firebase Setup -----
cred = credentials.Certificate("serviceAccountKey.json")
firebase_admin.initialize_app(cred, {
    "databaseURL": "YOUR DATA BASE URL"
})

# ----- Local Name → System ID Map -----
name_to_id = {
    "Aman Chand": "2022002974",
    "Elon Musk": "2022003852",
    "Shubham Jaswar": "2022445249"
}

# ----- File System -----
os.makedirs("Proofs", exist_ok=True)

#Load ML Models
knn = joblib.load('knn_model.pkl')
shape_predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')
face_rec_model = dlib.face_recognition_model_v1('dlib_face_recognition_resnet_model_v1.dat')
face_detector = dlib.get_frontal_face_detector()

#Motion Detector
fgbg = cv2.createBackgroundSubtractorMOG2()

#Video Source
cap = cv2.VideoCapture("both.mp4")
if not cap.isOpened():
    raise IOError("Error: Could not open video file.")

frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
ground_level = int(frame_height * 0.75)

object_positions = {}
FRAME_SKIP_FACE = 3
BG_RESET_INTERVAL = 600
FIREBASE_TIMEOUT_SEC = 2.5
MIN_CONTOUR_AREA = 1000
FETCH_COOLDOWN = 30  # seconds cooldown per person

frame_count = 0
last_labels = {}
last_fetch_time = {}  # per-person Firebase fetch timer

detection_count = {}  # person_name - count
MAIL_COOLDOWN = 10  # seconds
last_mail_time = {}  # per-person mail timer

# -----------------------------------------------
# Firebase
# -----------------------------------------------
def fetch_data_by_system_id(system_id):
    """Fetch data from Firebase under Students/<system_id>."""
    try:
        ref = db.reference(f"Students/{system_id}")
        data = ref.get()
        if data:
            data["system_id"] = system_id
            return data
        else:
            print(f"⚠️ No record found for System ID {system_id} in Firebase.")
    except Exception as e:
        print(f"⚠️ Firebase fetch error for {system_id}: {e}")
    return None


def fetch_person_data_blocking(person_name, out):
    """Fetch person details using system ID mapping."""
    system_id = name_to_id.get(person_name)
    if not system_id:
        out["error"] = f"No System ID found for {person_name}"
        return
    print(f"[DEBUG] Fetching Firebase data for {person_name} -> {system_id}")
    data = fetch_data_by_system_id(system_id)
    if data:
        out["data"] = data
    else:
        out["error"] = f"No data found for System ID {system_id}"


# -----------------------------------------------
print("System started. Press 'q' to quit.")
# -----------------------------------------------

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break
    frame_count += 1

    try:
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        detected_faces = {}

        # ---- Face Recognition ----
        if frame_count % FRAME_SKIP_FACE == 0:
            faces = face_detector(gray)
            for rect in faces:
                try:
                    shape = shape_predictor(gray, rect)
                    embedding = np.array(face_rec_model.compute_face_descriptor(frame, shape))
                    prediction = knn.predict([embedding]) if embedding is not None else []
                    label = prediction[0] if len(prediction) > 0 else "Unknown"
                    detected_faces[label] = (rect.left(), rect.top(), rect.right(), rect.bottom())
                    last_labels = detected_faces or last_labels
                except Exception:
                    pass
        else:
            detected_faces = last_labels

        # Draw face boxes
        for label, (lx1, ly1, lx2, ly2) in detected_faces.items():
            display_text = label
            if label in name_to_id:
                display_text = f"{label} | {name_to_id[label]}"
            cv2.rectangle(frame, (lx1, ly1), (lx2, ly2), (0, 255, 0), 2)
            cv2.putText(frame, display_text, (lx1, ly1 - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 0, 0), 2)

        # ---- Motion / Fall Detection ----
        fgmask = fgbg.apply(frame)
        contours, _ = cv2.findContours(fgmask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) < MIN_CONTOUR_AREA:
                continue
            x, y, w, h = cv2.boundingRect(contour)
            prev_y = object_positions.get(i)
            object_positions[i] = y

            if prev_y is not None and y > prev_y and (y + h) > ground_level:
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(frame, "Falling Object", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                print(f"⚠️Falling event @ frame {frame_count}")

                # Identify the person
                person_name = "Unknown"
                if detected_faces:
                    person_name = next(iter(detected_faces.keys()))

                # ---- Skip first detection per person ----
                if person_name != "Unknown":
                    detection_count[person_name] = detection_count.get(person_name, 0) + 1

                    if detection_count[person_name] == 1:
                        print(f"⏭️ Skipping first detection for {person_name}")
                        continue

                # Save screenshot (always)
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                screenshot_filename = f"Proofs/{person_name}_{timestamp}_fall_detected.png"
                cv2.imwrite(screenshot_filename, frame)
                print(f"Saved -> {screenshot_filename}")

                # ---- Fetch Firebase only once per 20 seconds per person ----
                if person_name != "Unknown":
                    now = time.time()
                    last_time = last_fetch_time.get(person_name, 0)
                    if now - last_time >= FETCH_COOLDOWN:
                        last_fetch_time[person_name] = now
                        user_data_holder = {}
                        t = threading.Thread(
                            target=fetch_person_data_blocking,
                            args=(person_name, user_data_holder),
                            daemon=True
                        )
                        t.start()
                        t.join(timeout=FIREBASE_TIMEOUT_SEC)
                        user_data = user_data_holder.get("data")

                        if user_data:
                            print("========== 🔍 PERSON DETAILS FROM FIREBASE ==========")
                            for key, value in user_data.items():
                                print(f"{key.capitalize():<12}: {value}")
                            print("====================================================\n")
                            name = user_data.get("name")
                            email = user_data.get("e-mail")

                            if name and email:
                                now = time.time()
                                last_sent = last_mail_time.get(name, 0)

                                if now - last_sent >= MAIL_COOLDOWN:
                                    success = send_mail(
                                        receiver_email=email,
                                        name=name,
                                        event="Improper waste disposal detected in University Campus",
                                        screenshot_path = screenshot_filename
                                    )

                                    if success:
                                        last_mail_time[name] = now
                                        print(f"📧 Mail sent to {name}")
                                else:
                                    print(f"⏳ Mail cooldown active for {name}")
                        else:
                            print(f"⚠️  No Firebase data found for {person_name}")
                    else:
                        print(f"Skipping fetch for {person_name} (within cooldown time)")
                else:
                    print("ℹ️  No recognized person for this fall")

        # ---- Maintenance ----
        if frame_count % BG_RESET_INTERVAL == 0:
            fgbg = cv2.createBackgroundSubtractorMOG2()
            object_positions.clear()
            print("[INFO] Reset background subtractor (memory hygiene)")

        if frame_count % 50 == 0:
            print(f"[INFO] Heartbeat: frame {frame_count}")

        # ---- Display ----
        cv2.line(frame, (0, ground_level), (frame_width, ground_level), (0, 0, 255), 2)
        cv2.imshow("TrashBot System", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    except Exception as loop_err:
        print("Loop error:", loop_err)
        continue

cap.release()
cv2.destroyAllWindows()
print("System terminated.")