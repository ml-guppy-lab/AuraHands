"""
AuraHands - Hand Gesture Recognition System
Feature 1: Finger Tip Detection
"""

import cv2  # Video capture and image processing
import mediapipe as mp  # Hand detection and tracking
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
import urllib.request  # For downloading model file
import os  # For file path operations

# ============================================================================
# DOWNLOAD MODEL IF NEEDED
# ============================================================================

MODEL_PATH = 'hand_landmarker.task'
FACE_MODEL_PATH = 'face_landmarker.task'

# Check if hand model file exists, if not download it
if not os.path.exists(MODEL_PATH):
    print("Downloading hand detection model (one-time setup)...")
    MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task'
    urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    print("Model downloaded successfully!")

# Check if face model file exists, if not download it
if not os.path.exists(FACE_MODEL_PATH):
    print("Downloading face detection model (one-time setup)...")
    FACE_MODEL_URL = 'https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/1/face_landmarker.task'
    urllib.request.urlretrieve(FACE_MODEL_URL, FACE_MODEL_PATH)
    print("Face model downloaded successfully!")

# ============================================================================
# SETUP
# ============================================================================

# MediaPipe detects 21 points per hand. We only need the 5 finger tips.
FINGER_TIPS = {
    'Thumb': 4,
    'Index': 8,
    'Middle': 12,
    'Ring': 16,
    'Pinky': 20
}

# Configure MediaPipe HandLandmarker (new API for v0.10+)
base_options = python.BaseOptions(model_asset_path=MODEL_PATH)
options = vision.HandLandmarkerOptions(
    base_options=base_options,
    num_hands=2,  # Detect up to 2 hands
    min_hand_detection_confidence=0.7,
    min_hand_presence_confidence=0.7,
    min_tracking_confidence=0.7
)

# Create hand detector
detector = vision.HandLandmarker.create_from_options(options)

# Configure face detector
face_base_options = python.BaseOptions(model_asset_path=FACE_MODEL_PATH)
face_options = vision.FaceLandmarkerOptions(
    base_options=face_base_options,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    min_tracking_confidence=0.5
)
face_detector = vision.FaceLandmarker.create_from_options(face_options)

# ============================================================================
# MODE TRACKING
# ============================================================================

current_mode = "hand"  # Start with hand detection mode ("hand", "face", or "both")
gesture_cooldown = 0  # Prevent multiple toggles
previous_fist_count = 0  # Track previous frame's fist count for gesture detection

# ============================================================================
# GESTURE DETECTION FUNCTIONS
# ============================================================================

def is_fist(hand_landmarks):
    """
    Detect if hand is making a fist.
    A fist is when all fingers are curled (finger tips close to palm).
    """
    # Get wrist position (palm base)
    wrist = hand_landmarks[0]
    
    # Check if each finger tip is close to the wrist (curled)
    fingers_curled = 0
    
    for finger_name, tip_id in FINGER_TIPS.items():
        tip = hand_landmarks[tip_id]
        
        # Calculate distance from finger tip to wrist
        distance = ((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)**0.5
        
        # If finger tip is close to wrist, it's curled
        # Threshold: 0.15 in normalized coordinates
        if distance < 0.15:
            fingers_curled += 1
    
    # If at least 4 out of 5 fingers are curled, it's a fist
    return fingers_curled >= 4

# ============================================================================
# START WEBCAM
# ============================================================================

cap = cv2.VideoCapture(0)  # 0 = default webcam

if not cap.isOpened():
    print("Error: Could not access webcam!")
    exit()

print("AuraHands started! Press 'q' to quit.")
print("Gestures:")
print("  - ONE FIST: Toggle between hand and face detection")
print("  - TWO FISTS: Enable both hand and face detection")

# ============================================================================
# MAIN LOOP
# ============================================================================

import time

while True:
    ret, frame = cap.read()  # Read frame from webcam
    
    if not ret:
        print("Error: Could not read frame!")
        break
    
    # Flip frame horizontally for mirror effect
    frame = cv2.flip(frame, 1)
    
    h, w, c = frame.shape  # Get frame dimensions
    
    # Convert BGR to RGB (MediaPipe needs RGB)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Create MediaPipe image object
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
    
    # Always detect hands to check for fist gesture
    detection_result = detector.detect(mp_image)
    
    # Decrease gesture cooldown
    if gesture_cooldown > 0:
        gesture_cooldown -= 1
    
    # ========================================================================
    # FIST DETECTION - Check for one or two fists
    # ========================================================================
    current_fist_count = 0
    
    if detection_result.hand_landmarks:
        for hand_landmarks in detection_result.hand_landmarks:
            if is_fist(hand_landmarks):
                current_fist_count += 1
    
    # Only trigger gesture when fist count transitions from 0 to 1+ (gesture starts)
    if current_fist_count > 0 and previous_fist_count == 0 and gesture_cooldown == 0:
        if current_fist_count == 1:
            # ONE FIST: Toggle between hand and face
            if current_mode == "hand":
                current_mode = "face"
            elif current_mode == "face":
                current_mode = "hand"
            else:  # from "both"
                current_mode = "hand"
            print(f"👊 ONE FIST detected! Switched to {current_mode.upper()} mode")
        
        elif current_fist_count == 2:
            # TWO FISTS: Enable both modes
            current_mode = "both"
            print(f"👊👊 TWO FISTS detected! Showing BOTH hand and face detection")
        
        gesture_cooldown = 30  # Wait 30 frames before detecting another gesture
    
    # Update previous fist count
    previous_fist_count = current_fist_count
    
    # ========================================================================
    # RENDER BASED ON CURRENT MODE
    # ========================================================================
    
    if current_mode == "hand":
        # HAND DETECTION MODE
        if detection_result.hand_landmarks:
            # Store finger tip positions for each hand
            all_hands_tips = []
            
            for hand_landmarks in detection_result.hand_landmarks:
                hand_tips = {}
                
                # Collect and draw finger tips for this hand
                for finger_name, tip_id in FINGER_TIPS.items():
                    tip = hand_landmarks[tip_id]
                    
                    # Convert normalized coordinates to pixels
                    tip_x = int(tip.x * w)
                    tip_y = int(tip.y * h)
                    
                    # Store position for connecting lines later
                    hand_tips[finger_name] = (tip_x, tip_y)
                    
                    # Draw yellow circle at finger tip
                    cv2.circle(frame, (tip_x, tip_y), 10, (0, 255, 255), -1)
                    
                    # Draw finger name above the circle
                    cv2.putText(frame, finger_name, (tip_x - 30, tip_y - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
                
                all_hands_tips.append(hand_tips)
            
            # If both hands detected, connect matching fingers
            if len(all_hands_tips) == 2:
                hand1_tips = all_hands_tips[0]
                hand2_tips = all_hands_tips[1]
                
                # Connect each matching finger between the two hands with laser effect
                for finger_name in FINGER_TIPS.keys():
                    pt1 = hand1_tips[finger_name]
                    pt2 = hand2_tips[finger_name]
                    
                    # Laser effect: Draw multiple layers for a glowing neon appearance
                    
                    # Outer glow (widest, semi-transparent)
                    cv2.line(frame, pt1, pt2, (0, 50, 255), 8, cv2.LINE_AA)
                    
                    # Middle glow
                    cv2.line(frame, pt1, pt2, (0, 100, 255), 5, cv2.LINE_AA)
                    
                    # Inner bright core (neon red)
                    cv2.line(frame, pt1, pt2, (0, 0, 255), 2, cv2.LINE_AA)
        
        # Display mode
        cv2.putText(frame, "MODE: HAND", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    elif current_mode == "face":
        # FACE DETECTION MODE
        face_result = face_detector.detect(mp_image)
        
        if face_result.face_landmarks:
            for face_landmarks in face_result.face_landmarks:
                # Draw face mesh (all 478 landmarks)
                for landmark in face_landmarks:
                    px = int(landmark.x * w)
                    py = int(landmark.y * h)
                    cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)
                
                # Highlight key facial features
                # Eyes (landmarks 33, 133, 362, 263)
                eye_indices = [33, 133, 362, 263]
                for idx in eye_indices:
                    landmark = face_landmarks[idx]
                    px = int(landmark.x * w)
                    py = int(landmark.y * h)
                    cv2.circle(frame, (px, py), 5, (255, 0, 255), -1)
                
                # Nose tip (landmark 1)
                nose = face_landmarks[1]
                nose_px = (int(nose.x * w), int(nose.y * h))
                cv2.circle(frame, nose_px, 8, (0, 255, 255), -1)
                cv2.putText(frame, "Nose", (nose_px[0] - 30, nose_px[1] - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        # Display mode
        cv2.putText(frame, "MODE: FACE", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    else:  # current_mode == "both"
        # BOTH HAND AND FACE DETECTION MODE
        
        # Draw hands
        if detection_result.hand_landmarks:
            all_hands_tips = []
            
            for hand_landmarks in detection_result.hand_landmarks:
                hand_tips = {}
                
                for finger_name, tip_id in FINGER_TIPS.items():
                    tip = hand_landmarks[tip_id]
                    tip_x = int(tip.x * w)
                    tip_y = int(tip.y * h)
                    hand_tips[finger_name] = (tip_x, tip_y)
                    
                    # Draw yellow circle at finger tip
                    cv2.circle(frame, (tip_x, tip_y), 8, (0, 255, 255), -1)
                
                all_hands_tips.append(hand_tips)
            
            # Connect matching fingers if both hands present
            if len(all_hands_tips) == 2:
                hand1_tips = all_hands_tips[0]
                hand2_tips = all_hands_tips[1]
                
                for finger_name in FINGER_TIPS.keys():
                    pt1 = hand1_tips[finger_name]
                    pt2 = hand2_tips[finger_name]
                    
                    # Thinner laser for both mode
                    cv2.line(frame, pt1, pt2, (0, 50, 255), 5, cv2.LINE_AA)
                    cv2.line(frame, pt1, pt2, (0, 100, 255), 3, cv2.LINE_AA)
                    cv2.line(frame, pt1, pt2, (0, 0, 255), 1, cv2.LINE_AA)
        
        # Draw face
        face_result = face_detector.detect(mp_image)
        if face_result.face_landmarks:
            for face_landmarks in face_result.face_landmarks:
                # Draw smaller face mesh dots
                for landmark in face_landmarks:
                    px = int(landmark.x * w)
                    py = int(landmark.y * h)
                    cv2.circle(frame, (px, py), 1, (0, 255, 0), -1)
                
                # Highlight key features
                eye_indices = [33, 133, 362, 263]
                for idx in eye_indices:
                    landmark = face_landmarks[idx]
                    px = int(landmark.x * w)
                    py = int(landmark.y * h)
                    cv2.circle(frame, (px, py), 4, (255, 0, 255), -1)
        
        # Display mode
        cv2.putText(frame, "MODE: BOTH", (10, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 255), 2)
    
    # Show instructions
    cv2.putText(frame, "Press 'q' to quit", (10, 30),
               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display the frame
    cv2.imshow("AuraHands - Finger Tip Detection", frame)
    
    # Check if 'q' is pressed to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ============================================================================
# CLEANUP
# ============================================================================

detector.close()  # Release hand detector
face_detector.close()  # Release face detector
cap.release()  # Release webcam
cv2.destroyAllWindows()  # Close all windows

print("AuraHands stopped successfully!")
