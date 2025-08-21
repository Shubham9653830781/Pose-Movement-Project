import cv2
import mediapipe as mp
import numpy as np

# Initialize mediapipe pose class.
mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

# Initialize webcam.
cap = cv2.VideoCapture(0)
cap.set(3, 768)
cap.set(4, 576)

# Distance calibration polynomial
X = [537, 464, 341, 285, 236, 188, 153, 141]
y = [50, 75, 100, 125, 150, 200, 250, 300]
coff = np.polyfit(X, y, 2)

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = pose.process(img_rgb)

    h, w, _ = img.shape
    vx = vy = vz = 0

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(img, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

        # Get nose landmark
        nose = results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE]
        nose_x = int(nose.x * w)
        nose_y = int(nose.y * h)
        cv2.circle(img, (nose_x, nose_y), 6, (0, 255, 0), -1)

        # Movement logic thresholds
        lr_thresh_min, lr_thresh_max = 340, 420
        ud_thresh_min, ud_thresh_max = 250, 310
        lrgap, lrvel = 50, 0.5
        udgap, udvel = 35, 0.5

        # LEFT/RIGHT direction
        if nose_x < lr_thresh_min:
            dx_m = (lr_thresh_min - nose_x) / 100  # approx 100 px = 1m
            vx = -1 * ((lr_thresh_min - nose_x) // lrgap + 1) * lrvel
            cv2.putText(img, f'Move Left | {dx_m:.2f} m', (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        elif nose_x > lr_thresh_max:
            dx_m = (nose_x - lr_thresh_max) / 100
            vx = ((nose_x - lr_thresh_max) // lrgap + 1) * lrvel
            cv2.putText(img, f'Move Right | {dx_m:.2f} m', (50, 80),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)

        # UP/DOWN direction
        if nose_y < ud_thresh_min:
            dy_m = (ud_thresh_min - nose_y) / 100
            vy = ((ud_thresh_min - nose_y) // udgap + 1) * udvel
            cv2.putText(img, f'Move Up | {dy_m:.2f} m', (50, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 150, 200), 2)

        elif nose_y > ud_thresh_max:
            dy_m = (nose_y - ud_thresh_max) / 100
            vy = -1 * ((nose_y - ud_thresh_max) // udgap + 1) * udvel
            cv2.putText(img, f'Move Down | {dy_m:.2f} m', (50, 140),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 150, 200), 2)

        # FORWARD/BACKWARD using shoulder distance
        left_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.LEFT_SHOULDER]
        right_shoulder = results.pose_landmarks.landmark[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        box_width = abs(left_shoulder.x - right_shoulder.x) * w

        A, B, C = coff
        distanceCM = A * box_width**2 + B * box_width + C
        distance_m = distanceCM / 100  # Convert to meters

        c, v = 25, 0.5
        if distanceCM < 125:
            dz_m = (125 - distanceCM) / 100
            vz = -1 * ((125 - distanceCM) // c + 1) * v
            cv2.putText(img, f'Move Backward | {dz_m:.2f} m', (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 100), 2)

        elif distanceCM > 175:
            dz_m = (distanceCM - 175) / 100
            vz = ((distanceCM - 175) // c + 1) * v
            cv2.putText(img, f'Move Forward | {dz_m:.2f} m', (50, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 100), 2)

        # Display current velocity vector
        cv2.putText(img, f'vx: {vx:.2f} m/s | vy: {vy:.2f} m/s | vz: {vz:.2f} m/s', (30, h - 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)

    cv2.imshow('MediaPipe Drone Follower', img)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
