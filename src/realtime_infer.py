# src/realtime_infer.py
"""
Realtime pipeline:
 - capture webcam frames
 - use MediaPipe FaceMesh to get landmarks
 - compute EAR & MAR via features.extract_features_from_face
 - maintain rolling window and counters; on threshold publish alert via mqtt_publisher
 - draw overlay on frame and show window
"""

import cv2
import time
import numpy as np
import collections
import argparse

from features import mp_face, extract_features_from_face
from iot.mqtt_publisher import publish_alert, init_client

# thresholds (tune for your dataset) - More realistic detection
EAR_THRESHOLD = 0.22
EAR_CONSEC_FRAMES = 60  # ~2.5 seconds at 24fps (eyes closed for 2.5+ seconds)
MAR_THRESHOLD = 0.6
MAR_CONSEC = 30  # ~1.25 seconds at 24fps (yawning for 1.25+ seconds)

# Additional thresholds for more robust detection
HEAD_DOWN_THRESHOLD = 0.3  # Head pose threshold for nodding
HEAD_DOWN_CONSEC = 60  # ~2.5 seconds of head down
YAWN_COUNT_THRESHOLD = 4  # Number of yawns before alert
YAWN_WINDOW = 300  # 12.5 seconds window to count yawns

def main(args):
    init_client()
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Unable to open webcam.")
        return
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    ear_counter = 0
    mar_counter = 0
    head_down_counter = 0
    rolling_ear = collections.deque(maxlen=10)
    
    # Yawn tracking
    yawn_timestamps = collections.deque(maxlen=10)  # Store timestamps of yawns
    last_yawn_time = 0
    
    # Alert cooldown to prevent spam
    last_alert_time = 0
    alert_cooldown = 5  # 5 seconds between alerts
    with mp_face.FaceMesh(static_image_mode=False, max_num_faces=1, refine_landmarks=True, min_detection_confidence=0.5, min_tracking_confidence=0.5) as face_mesh:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(img_rgb)
            alert = False
            features = {}
            current_time = time.time()
            
            if results.multi_face_landmarks:
                face_lm = results.multi_face_landmarks[0].landmark
                feats = extract_features_from_face(face_lm, frame.shape)
                features = feats
                ear = feats['ear']
                mar = feats['mar']
                head_pose = feats['head_pose']
                
                rolling_ear.append(ear)
                avg_ear = np.mean([max(0.0, v) for v in rolling_ear]) if rolling_ear else ear
                
                # 1. EYES CLOSED CHECK (5+ seconds)
                if avg_ear < EAR_THRESHOLD:
                    ear_counter += 1
                else:
                    ear_counter = 0
                
                # 2. HEAD DOWN CHECK (5+ seconds)
                if head_pose[0] is not None:  # rotation vector available
                    head_rotation_y = abs(head_pose[0][1])  # Y-axis rotation (nodding)
                    if head_rotation_y > HEAD_DOWN_THRESHOLD:
                        head_down_counter += 1
                    else:
                        head_down_counter = 0
                
                # 3. YAWNING CHECK (4-5 yawns in 12.5 seconds)
                if mar > MAR_THRESHOLD:
                    mar_counter += 1
                    # Record yawn if it's been going on for a while
                    if mar_counter >= MAR_CONSEC and (current_time - last_yawn_time) > 2:
                        yawn_timestamps.append(current_time)
                        last_yawn_time = current_time
                        mar_counter = 0  # Reset to avoid counting same yawn multiple times
                else:
                    mar_counter = 0
                
                # Count yawns in the last 12.5 seconds
                recent_yawns = [t for t in yawn_timestamps if (current_time - t) <= YAWN_WINDOW/fps]
                
                # ALERT CONDITIONS (all must be met for realistic drowsiness)
                eyes_closed_long = ear_counter >= EAR_CONSEC_FRAMES  # 5+ seconds eyes closed
                head_down_long = head_down_counter >= HEAD_DOWN_CONSEC  # 5+ seconds head down
                multiple_yawns = len(recent_yawns) >= YAWN_COUNT_THRESHOLD  # 4+ yawns in 12.5 seconds
                
                # Only alert if multiple conditions are met AND cooldown has passed
                if (eyes_closed_long or head_down_long or multiple_yawns) and (current_time - last_alert_time) > alert_cooldown:
                    alert = True
                    last_alert_time = current_time
            # Publish and local notify
            if alert:
                payload = {"alert":"drowsy", "ear": features.get('ear'), "mar": features.get('mar'), 
                          "yawns": len(recent_yawns), "eyes_closed": eyes_closed_long, "head_down": head_down_long}
                publish_alert(payload)
                # simple local beep via OpenCV window (visual)
                cv2.putText(frame, "DROWSINESS ALERT!", (30,60), cv2.FONT_HERSHEY_SIMPLEX, 1.4, (0,0,255), 3)
            
            # overlay detection values and status
            if features:
                # Basic values
                cv2.putText(frame, f"EAR:{features['ear']:.2f}", (10,30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                cv2.putText(frame, f"MAR:{features['mar']:.2f}", (10,60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255,255,0), 2)
                
                # Detection counters
                cv2.putText(frame, f"Eyes Closed: {ear_counter}/{EAR_CONSEC_FRAMES}", (10,90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                cv2.putText(frame, f"Head Down: {head_down_counter}/{HEAD_DOWN_CONSEC}", (10,110), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                cv2.putText(frame, f"Recent Yawns: {len(recent_yawns)}/{YAWN_COUNT_THRESHOLD}", (10,130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                
                # Status indicators
                if eyes_closed_long:
                    cv2.putText(frame, "EYES CLOSED LONG!", (10,160), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                if head_down_long:
                    cv2.putText(frame, "HEAD DOWN LONG!", (10,180), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
                if multiple_yawns:
                    cv2.putText(frame, "MULTIPLE YAWNS!", (10,200), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,0,255), 2)
            cv2.imshow("Drowsiness Detection", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC
                break
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--no-mqtt", action="store_true", help="Don't init MQTT (debug)")
    args = parser.parse_args()
    main(args)