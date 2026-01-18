# YOLOv8 + Eye Tracking Implementation
# This would be an enhanced version using YOLO for object detection + specialized eye tracking

import cv2
import torch
from ultralytics import YOLO
import numpy as np
import mediapipe as mp
from scipy.spatial.distance import euclidean
import time
import asyncio
import websockets
import json

class YOLOEyeTracker:
    def __init__(self):
        # Load YOLOv8 model for face detection
        self.yolo_model = YOLO('yolov8n-face.pt')  # Face detection model
        
        # MediaPipe for detailed eye landmarks
        self.mp_face_mesh = mp.solutions.face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7
        )
        
        # Eye landmark indices (MediaPipe)
        self.LEFT_EYE_LANDMARKS = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        self.RIGHT_EYE_LANDMARKS = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
        
        # Gaze estimation parameters
        self.gaze_history = []
        self.gaze_smoothing_window = 5
        
        # Eye movement thresholds
        self.SACCADE_THRESHOLD = 0.02  # Minimum movement to detect saccade
        self.FIXATION_DURATION = 0.3   # Time to confirm fixation
        self.BLINK_THRESHOLD = 0.2     # Eye aspect ratio threshold
        
        # Calibration data
        self.calibration_points = {}
        self.is_calibrated = False
        
    def detect_faces_yolo(self, frame):
        """Use YOLOv8 to detect faces in the frame"""
        results = self.yolo_model(frame, verbose=False)
        
        faces = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    # Extract face bounding box
                    x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                    confidence = box.conf[0].cpu().numpy()
                    
                    if confidence > 0.7:  # Face confidence threshold
                        faces.append({
                            'bbox': (int(x1), int(y1), int(x2), int(y2)),
                            'confidence': confidence
                        })
        
        return faces
    
    def extract_eye_landmarks(self, frame, face_bbox):
        """Extract detailed eye landmarks using MediaPipe"""
        x1, y1, x2, y2 = face_bbox
        
        # Crop face region for better landmark detection
        face_roi = frame[y1:y2, x1:x2]
        face_rgb = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
        
        # Get face landmarks
        results = self.mp_face_mesh.process(face_rgb)
        
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0]
            
            # Convert landmarks to absolute coordinates
            h, w = face_roi.shape[:2]
            eye_landmarks = {}
            
            # Left eye landmarks
            left_eye_points = []
            for idx in self.LEFT_EYE_LANDMARKS:
                x = int(landmarks.landmark[idx].x * w) + x1
                y = int(landmarks.landmark[idx].y * h) + y1
                left_eye_points.append((x, y))
            
            # Right eye landmarks
            right_eye_points = []
            for idx in self.RIGHT_EYE_LANDMARKS:
                x = int(landmarks.landmark[idx].x * w) + x1
                y = int(landmarks.landmark[idx].y * h) + y1
                right_eye_points.append((x, y))
            
            eye_landmarks['left_eye'] = left_eye_points
            eye_landmarks['right_eye'] = right_eye_points
            
            return eye_landmarks
        
        return None
    
    def calculate_eye_aspect_ratio(self, eye_points):
        """Calculate Eye Aspect Ratio (EAR) for blink detection"""
        # Vertical eye landmarks
        A = euclidean(eye_points[1], eye_points[5])
        B = euclidean(eye_points[2], eye_points[4])
        
        # Horizontal eye landmark
        C = euclidean(eye_points[0], eye_points[3])
        
        # Eye aspect ratio
        ear = (A + B) / (2.0 * C)
        return ear
    
    def estimate_gaze_direction(self, eye_landmarks):
        """Estimate gaze direction using eye landmarks"""
        if not eye_landmarks:
            return None
        
        left_eye = eye_landmarks['left_eye']
        right_eye = eye_landmarks['right_eye']
        
        # Calculate eye centers
        left_center = np.mean(left_eye, axis=0)
        right_center = np.mean(right_eye, axis=0)
        
        # Calculate pupil positions (simplified - would need iris detection for accuracy)
        # This is a simplified approach - real implementation would use iris detection
        left_pupil = left_center
        right_pupil = right_center
        
        # Calculate gaze vector (simplified)
        gaze_x = (left_pupil[0] + right_pupil[0]) / 2
        gaze_y = (left_pupil[1] + right_pupil[1]) / 2
        
        return (gaze_x, gaze_y)
    
    def detect_eye_movements(self, eye_landmarks):
        """Detect various eye movements and interactions"""
        if not eye_landmarks:
            return None
        
        movements = {
            'blink': False,
            'saccade': False,
            'fixation': False,
            'gaze_direction': None,
            'interaction_point': None
        }
        
        # 1. Blink Detection
        left_ear = self.calculate_eye_aspect_ratio(eye_landmarks['left_eye'][:6])
        right_ear = self.calculate_eye_aspect_ratio(eye_landmarks['right_eye'][:6])
        avg_ear = (left_ear + right_ear) / 2.0
        
        movements['blink'] = avg_ear < self.BLINK_THRESHOLD
        
        # 2. Gaze Direction Estimation
        gaze_point = self.estimate_gaze_direction(eye_landmarks)
        movements['gaze_direction'] = gaze_point
        
        # 3. Saccade Detection (rapid eye movements)
        if gaze_point and len(self.gaze_history) > 0:
            last_gaze = self.gaze_history[-1]
            movement_distance = euclidean(gaze_point, last_gaze)
            movements['saccade'] = movement_distance > self.SACCADE_THRESHOLD
        
        # 4. Update gaze history for smoothing
        if gaze_point:
            self.gaze_history.append(gaze_point)
            if len(self.gaze_history) > self.gaze_smoothing_window:
                self.gaze_history.pop(0)
        
        # 5. Screen Interaction Points (if calibrated)
        if self.is_calibrated and gaze_point:
            movements['interaction_point'] = self.map_gaze_to_screen(gaze_point)
        
        return movements
    
    def map_gaze_to_screen(self, gaze_point):
        """Map gaze coordinates to screen interaction points"""
        # This would map eye gaze to specific UI elements
        # Requires calibration data for accuracy
        if not self.is_calibrated:
            return None
        
        # Simplified mapping - real implementation would use calibration matrix
        screen_x = gaze_point[0] / 640 * 1920  # Assuming 640x480 -> 1920x1080
        screen_y = gaze_point[1] / 480 * 1080
        
        return (screen_x, screen_y)
    
    def calibrate_gaze(self, calibration_points):
        """Calibrate gaze tracking for accurate screen mapping"""
        self.calibration_points = calibration_points
        self.is_calibrated = True
        
    def process_frame(self, frame):
        """Main processing function"""
        # 1. Detect faces using YOLOv8
        faces = self.detect_faces_yolo(frame)
        
        if not faces:
            return None
        
        # Use the most confident face
        best_face = max(faces, key=lambda x: x['confidence'])
        
        # 2. Extract eye landmarks
        eye_landmarks = self.extract_eye_landmarks(frame, best_face['bbox'])
        
        if not eye_landmarks:
            return None
        
        # 3. Detect eye movements and interactions
        movements = self.detect_eye_movements(eye_landmarks)
        
        # 4. Add face bounding box info
        movements['face_bbox'] = best_face['bbox']
        movements['face_confidence'] = best_face['confidence']
        
        return movements

# Usage Example:
def main():
    tracker = YOLOEyeTracker()
    cap = cv2.VideoCapture(0)
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
            
        # Process frame for eye movements
        result = tracker.process_frame(frame)
        
        if result:
            # Handle different eye interactions
            if result['blink']:
                print("Blink detected!")
                # Trigger UI action
                
            if result['saccade']:
                print("Eye movement detected!")
                
            if result['interaction_point']:
                x, y = result['interaction_point']
                print(f"Looking at screen position: ({x}, {y})")
                # Map to UI elements
        
        # Display frame with annotations
        cv2.imshow('YOLOv8 Eye Tracking', frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()