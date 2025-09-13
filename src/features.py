# src/features.py
"""
Face/eye feature extraction using MediaPipe FaceMesh and OpenCV.
Computes:
 - Eye Aspect Ratio (EAR) for left & right eyes
 - Mouth Aspect Ratio (MAR) for yawning detection
 - Head pose estimation (solvePnP) returning rotation vector
"""

import cv2
import mediapipe as mp
import numpy as np
from math import hypot

mp_face = mp.solutions.face_mesh
_FACE_MESH = mp.solutions.face_mesh

# Indices for landmarks (MediaPipe)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
MOUTH = [61, 291, 13, 14]  # left corner, right corner, top center, bottom center
HP_INDICES = {"nose_tip":1, "chin":152, "left_eye":33, "right_eye":263, "left_mouth":61, "right_mouth":291}

def _dist(a, b):
    return hypot(a[0]-b[0], a[1]-b[1])

def eye_aspect_ratio(pts):
    # pts: list of 6 (x,y)
    A = _dist(pts[1], pts[5])
    B = _dist(pts[2], pts[4])
    C = _dist(pts[0], pts[3])
    return (A + B) / (2.0 * C + 1e-8)

def mouth_aspect_ratio(pts):
    # pts: [left_corner, right_corner, top_center, bottom_center]
    vertical = _dist(pts[2], pts[3])
    horizontal = _dist(pts[0], pts[1]) + 1e-8
    return vertical / horizontal

def get_landmark_coords(landmark, w, h):
    return (int(landmark.x * w), int(landmark.y * h))

def estimate_head_pose(landmarks, image_shape):
    img_h, img_w = image_shape[:2]
    image_points = np.array([
        get_landmark_coords(landmarks[HP_INDICES["nose_tip"]], img_w, img_h),
        get_landmark_coords(landmarks[HP_INDICES["chin"]], img_w, img_h),
        get_landmark_coords(landmarks[HP_INDICES["left_eye"]], img_w, img_h),
        get_landmark_coords(landmarks[HP_INDICES["right_eye"]], img_w, img_h),
        get_landmark_coords(landmarks[HP_INDICES["left_mouth"]], img_w, img_h),
        get_landmark_coords(landmarks[HP_INDICES["right_mouth"]], img_w, img_h)
    ], dtype='double')

    model_points = np.array([
        (0.0, 0.0, 0.0),
        (0.0, -330.0, -65.0),
        (-225.0, 170.0, -135.0),
        (225.0, 170.0, -135.0),
        (-150.0, -150.0, -125.0),
        (150.0, -150.0, -125.0)
    ])

    focal_length = img_w
    center = (img_w/2, img_h/2)
    camera_matrix = np.array([[focal_length, 0, center[0]],
                              [0, focal_length, center[1]],
                              [0, 0, 1]], dtype='double')
    dist_coeffs = np.zeros((4,1))
    success, rotation_vector, translation_vector = cv2.solvePnP(model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)
    return success, rotation_vector, translation_vector

# Helper that given mediapipe results returns EAR_left, EAR_right, MAR, head_pose
def extract_features_from_face(face_landmarks, image_shape):
    # face_landmarks: list-like of landmarks (MediaPipe)
    img_h, img_w = image_shape[:2]
    lm = {i: face_landmarks[i] for i in range(len(face_landmarks))}
    # eye coords
    left_eye_pts = [get_landmark_coords(face_landmarks[i], img_w, img_h) for i in LEFT_EYE]
    right_eye_pts = [get_landmark_coords(face_landmarks[i], img_w, img_h) for i in RIGHT_EYE]
    mouth_pts = [get_landmark_coords(face_landmarks[i], img_w, img_h) for i in MOUTH]
    ear_l = eye_aspect_ratio(left_eye_pts)
    ear_r = eye_aspect_ratio(right_eye_pts)
    mar = mouth_aspect_ratio(mouth_pts)
    success, rot_vec, trans_vec = estimate_head_pose(face_landmarks, image_shape)
    return {"ear_l": ear_l, "ear_r": ear_r, "ear": (ear_l+ear_r)/2.0, "mar": mar, "head_pose": (rot_vec, trans_vec)}