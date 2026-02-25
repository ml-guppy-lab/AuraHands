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
import numpy as np  # For fireball animation effects
import time
import math

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
# FIREBALL ANIMATION SETUP
# ============================================================================

fireball_animation_frame = 0  # Animation counter for dynamic effects
fireball_particles = []  # Store particle effects for each fireball

def draw_fireball(frame, center_x, center_y, animation_frame):
    """
    Draw an animated fireball effect at the specified position.
    Creates a layered, glowing fireball with particle effects.
    """
    # Animated pulsing effect (size changes)
    pulse = math.sin(animation_frame * 0.2) * 5 + 25
    base_radius = int(pulse)
    
    # Outer glow (red-orange gradient)
    cv2.circle(frame, (center_x, center_y), base_radius + 20, (0, 50, 255), -1)
    cv2.circle(frame, (center_x, center_y), base_radius + 15, (0, 100, 255), -1)
    cv2.circle(frame, (center_x, center_y), base_radius + 10, (0, 150, 255), -1)
    
    # Mid layer (bright orange)
    cv2.circle(frame, (center_x, center_y), base_radius + 5, (0, 200, 255), -1)
    
    # Inner core (yellow-white hot center)
    cv2.circle(frame, (center_x, center_y), base_radius, (50, 255, 255), -1)
    cv2.circle(frame, (center_x, center_y), int(base_radius * 0.6), (150, 255, 255), -1)
    cv2.circle(frame, (center_x, center_y), int(base_radius * 0.3), (255, 255, 255), -1)
    
    # Add swirling particles around the fireball
    num_particles = 8
    for i in range(num_particles):
        angle = (animation_frame * 0.1 + i * (360 / num_particles)) % 360
        angle_rad = math.radians(angle)
        
        # Particles orbit around the fireball
        particle_distance = base_radius + 25 + math.sin(animation_frame * 0.15 + i) * 10
        px = int(center_x + math.cos(angle_rad) * particle_distance)
        py = int(center_y + math.sin(angle_rad) * particle_distance)
        
        # Draw glowing particles
        particle_size = 3 + int(math.sin(animation_frame * 0.2 + i) * 2)
        cv2.circle(frame, (px, py), particle_size + 2, (0, 100, 255), -1)
        cv2.circle(frame, (px, py), particle_size, (0, 255, 255), -1)

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


def count_extended_fingers(hand_landmarks):
    """
    Count how many fingers are extended (pointing away from palm).
    Returns number of extended fingers (0-5).
    """
    wrist = hand_landmarks[0]
    extended_count = 0
    
    for finger_name, tip_id in FINGER_TIPS.items():
        tip = hand_landmarks[tip_id]
        
        # Calculate distance from finger tip to wrist
        distance = ((tip.x - wrist.x)**2 + (tip.y - wrist.y)**2)**0.5
        
        # If finger tip is far from wrist, it's extended
        # Threshold: 0.2 in normalized coordinates
        if distance > 0.2:
            extended_count += 1
    
    return extended_count

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
print("  - TWO FISTS: Summon fireballs above your palms!")
print("  - TWO FINGERS (both hands in face mode): Yellow eyes mode!")

# ============================================================================
# MAIN LOOP
# ============================================================================

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
            print(f"👊 ONE FIST detected! Switched to {current_mode.upper()} mode")
        
        gesture_cooldown = 30  # Wait 30 frames before detecting another gesture
    
    # Update previous fist count
    previous_fist_count = current_fist_count
    
    # Check if TWO FISTS are currently active (for fireball display)
    show_fireballs = (current_fist_count == 2)
    
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
        
        # Check if both hands are showing 2 fingers
        both_hands_two_fingers = False
        if detection_result.hand_landmarks and len(detection_result.hand_landmarks) == 2:
            hand1_fingers = count_extended_fingers(detection_result.hand_landmarks[0])
            hand2_fingers = count_extended_fingers(detection_result.hand_landmarks[1])
            
            if hand1_fingers == 2 and hand2_fingers == 2:
                both_hands_two_fingers = True
        
        if face_result.face_landmarks:
            for face_landmarks in face_result.face_landmarks:
                
                if both_hands_two_fingers:
                    # Special mode: Hide face mesh, show only yellow iris/pupil
                    
                    # Calculate face size to scale iris proportionally
                    # Use distance between eyes to determine face size
                    left_eye_outer = face_landmarks[33]  # Left eye outer corner
                    right_eye_outer = face_landmarks[263]  # Right eye outer corner
                    
                    # Calculate distance between eyes in pixels
                    eye_distance = ((left_eye_outer.x - right_eye_outer.x) * w)**2 + \
                                   ((left_eye_outer.y - right_eye_outer.y) * h)**2
                    eye_distance = eye_distance ** 0.5
                    
                    # Scale iris size based on face size (proportional to eye distance)
                    # Typical eye distance is around 80-120 pixels, scale accordingly
                    iris_radius = int(eye_distance * 0.05)  # 15% of eye distance
                    pupil_radius = int(iris_radius * 0.4)  # Pupil is 40% of iris
                    
                    # Clamp values to reasonable range
                    iris_radius = max(8, min(iris_radius, 25))
                    pupil_radius = max(3, min(pupil_radius, 10))
                    
                    # Draw yellow iris/pupil over the eye centers (NO face mesh)
                    # Left eye iris center (landmark 468 - left pupil)
                    left_iris = face_landmarks[468]
                    left_x = int(left_iris.x * w)
                    left_y = int(left_iris.y * h)
                    # Draw scaled yellow circle for iris
                    cv2.circle(frame, (left_x, left_y), iris_radius, (0, 255, 255), -1)
                    # Add darker center for pupil effect
                    cv2.circle(frame, (left_x, left_y), pupil_radius, (0, 200, 200), -1)
                    
                    # Right eye iris center (landmark 473 - right pupil)
                    right_iris = face_landmarks[473]
                    right_x = int(right_iris.x * w)
                    right_y = int(right_iris.y * h)
                    # Draw scaled yellow circle for iris
                    cv2.circle(frame, (right_x, right_y), iris_radius, (0, 255, 255), -1)
                    # Add darker center for pupil effect
                    cv2.circle(frame, (right_x, right_y), pupil_radius, (0, 200, 200), -1)
                    
                    cv2.putText(frame, "YELLOW EYES MODE", (10, h - 20),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                else:
                    # Normal mode: Draw full face mesh
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
    
    # ========================================================================
    # FIREBALL RENDERING (independent of mode)
    # ========================================================================
    
    if show_fireballs and detection_result.hand_landmarks:
        # Draw fireballs above each palm when both fists are closed
        if len(detection_result.hand_landmarks) == 2:
            for hand_landmarks in detection_result.hand_landmarks:
                # Get palm position (wrist landmark 0, but move it up a bit)
                wrist = hand_landmarks[0]
                palm_x = int(wrist.x * w)
                palm_y = int(wrist.y * h) - 80  # Position fireball above palm
                
                # Draw animated fireball
                draw_fireball(frame, palm_x, palm_y, fireball_animation_frame)
            
            # Display fireball mode indicator
            cv2.putText(frame, "FIREBALL MODE", (10, h - 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 150, 255), 2)
    
    # Increment animation frame for next iteration
    fireball_animation_frame += 1
    
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
