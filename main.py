import cv2
import mediapipe as mp
import time

# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

import os
import sys

def resource_path(filename):
    if getattr(sys, "frozen", False):
        # PyInstaller temporary extraction directory
        print("frozen")
        base_path = sys._MEIPASS
    else:
        # Normal Python execution
        print("normal")
        base_path = os.path.dirname(os.path.abspath(__file__))

    return os.path.join(base_path, filename)


MODEL_PATH = resource_path("hand_landmarker.task")

# ---------------------------------------------------------
# MediaPipe setup
# ---------------------------------------------------------

BaseOptions = mp.tasks.BaseOptions
HandLandmarker = mp.tasks.vision.HandLandmarker
HandLandmarkerOptions = mp.tasks.vision.HandLandmarkerOptions
RunningMode = mp.tasks.vision.RunningMode

options = HandLandmarkerOptions(
    base_options=BaseOptions(
        model_asset_path=MODEL_PATH
    ),
    running_mode=RunningMode.VIDEO,
    num_hands=2,
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.5
)

# ---------------------------------------------------------
# Open camera
# ---------------------------------------------------------

cap = cv2.VideoCapture(0)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)

if not cap.isOpened():
    raise RuntimeError("Could not open camera.")

# Timestamp for MediaPipe video mode
start_time = time.monotonic()

# ---------------------------------------------------------
# Run hand tracking
# ---------------------------------------------------------

with HandLandmarker.create_from_options(options) as landmarker:

    while True:
        ret, frame = cap.read()

        if not ret:
            print("Could not read frame from camera.")
            break

        # Mirror effect
        frame = cv2.flip(frame, 1)

        # Get frame dimensions
        h, w, _ = frame.shape

        # OpenCV uses BGR, MediaPipe expects RGB
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Create MediaPipe Image
        mp_image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=rgb_frame
        )

        # MediaPipe VIDEO mode requires monotonically increasing
        # timestamps in milliseconds.
        timestamp_ms = int((time.monotonic() - start_time) * 1000)

        # Run hand detection
        result = landmarker.detect_for_video(
            mp_image,
            timestamp_ms
        )

        # -------------------------------------------------
        # Draw detected hands
        # -------------------------------------------------

        for hand_landmarks in result.hand_landmarks:

            # Draw connections
            connections = [
                (0, 1),
                (1, 2),
                (2, 3),
                (3, 4),

                (0, 5),
                (5, 6),
                (6, 7),
                (7, 8),

                (0, 9),
                (9, 10),
                (10, 11),
                (11, 12),

                (0, 13),
                (13, 14),
                (14, 15),
                (15, 16),

                (0, 17),
                (17, 18),
                (18, 19),
                (19, 20),

                (5, 9),
                (9, 13),
                (13, 17)
            ]

            # Draw skeleton
            for start, end in connections:
                x1 = int(hand_landmarks[start].x * w)
                y1 = int(hand_landmarks[start].y * h)

                x2 = int(hand_landmarks[end].x * w)
                y2 = int(hand_landmarks[end].y * h)

                cv2.line(
                    frame,
                    (x1, y1),
                    (x2, y2),
                    (255, 0, 0),
                    2
                )

            # Draw landmarks
            for idx, landmark in enumerate(hand_landmarks):

                x = int(landmark.x * w)
                y = int(landmark.y * h)

                # Fingertips
                if idx in [4, 8, 12, 16, 20]:

                    cv2.circle(
                        frame,
                        (x, y),
                        8,
                        (0, 0, 255),
                        -1
                    )

                else:

                    cv2.circle(
                        frame,
                        (x, y),
                        4,
                        (0, 255, 0),
                        -1
                    )

        # -------------------------------------------------
        # Display
        # -------------------------------------------------

        cv2.putText(
            frame,
            "Hand Tracking (Press Q to quit)",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8,
            (0, 255, 0),
            2
        )

        cv2.imshow("Hand Tracking", frame)

        # Quit with Q
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break

# ---------------------------------------------------------
# Cleanup
# ---------------------------------------------------------

cap.release()
cv2.destroyAllWindows()
