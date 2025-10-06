import warnings
import os
import threading
import time
from flask import Flask, render_template, request, redirect, url_for, flash, jsonify
import webbrowser
from PIL import Image, ImageTk, ImageDraw
import tkinter as tk
from tkinter import Toplevel, Label

# Suppress protobuf warnings
warnings.filterwarnings("ignore", message=".*GetPrototype.*")
warnings.filterwarnings("ignore", category=UserWarning, module="google.protobuf.symbol_database")
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import datetime
import pickle
import tkinter as tk
from tkinter import messagebox, Text, Entry, Label, Button, simpledialog
import cv2
import numpy as np
from PIL import Image, ImageTk
import mediapipe as mp
from deepface import DeepFace
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import math
from collections import defaultdict

# Flask app for registration
app = Flask(__name__)
app.secret_key = 'face_recognition_secret_key'

# Global variables for camera access
global_camera = None
registration_mode = False
registration_photos = []
registration_name = ""
registration_angles = ["Front View", "Left Profile", "Right Profile", "Slight Smile"]
completed_angles = []

# HEAD POSE ESTIMATOR CLASS
class HeadPoseEstimator:
    def __init__(self):
        try:
            # Initialize separate MediaPipe Face Mesh instance for head pose
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            print("✓ HeadPoseEstimator MediaPipe FaceMesh initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing HeadPoseEstimator MediaPipe FaceMesh: {e}")
            self.face_mesh = None
            self.mp_face_mesh = None
        
        # 3D model points for head pose estimation (in millimeters)
        self.model_points = np.array([
            [0.0, 0.0, 0.0],           # Nose tip
            [0.0, -330.0, -65.0],      # Chin
            [-225.0, 170.0, -135.0],   # Left eye left corner
            [225.0, 170.0, -135.0],    # Right eye right corner
            [-150.0, -150.0, -125.0],  # Left mouth corner
            [150.0, -150.0, -125.0]   # Right mouth corner
        ], dtype=np.float64)
        
        # Camera calibration parameters
        self.focal_length = 1000.0
        self.camera_center_x = 320.0
        self.camera_center_y = 240.0
        
        # Create camera matrix with individual scalar values
        self.camera_matrix = np.array([
            [self.focal_length, 0.0, self.camera_center_x],
            [0.0, self.focal_length, self.camera_center_y],
            [0.0, 0.0, 1.0]
        ], dtype=np.float64)
        
        # Distortion coefficients
        self.dist_coeffs = np.zeros((4, 1), dtype=np.float64)
        
        # Head rotation thresholds
        self.rotation_threshold = 70.0
        
        # Key landmark indices for head pose
        self.key_landmarks = [1, 199, 33, 263, 61, 291]
        
    def extract_head_pose_landmarks(self, face_roi, bbox=None):
        """Extract specific landmarks for head pose estimation"""
        if self.face_mesh is None:
            return None
            
        try:
            if face_roi is None or face_roi.size == 0:
                return None
                
            # Convert to RGB
            rgb_face = cv2.cvtColor(face_roi, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_face)
            
            if results.multi_face_landmarks and len(results.multi_face_landmarks) > 0:
                face_landmarks = results.multi_face_landmarks[0]
                h, w = face_roi.shape[:2]
                
                image_points = []
                
                # Extract key landmarks for head pose
                for idx in self.key_landmarks:
                    if idx < len(face_landmarks.landmark):
                        landmark = face_landmarks.landmark[idx]
                        x = float(landmark.x * w)
                        y = float(landmark.y * h)
                        image_points.append([x, y])
                
                # Ensure we have exactly 6 points
                if len(image_points) == 6:
                    return np.array(image_points, dtype=np.float64)
                    
        except Exception as e:
            print(f"Error extracting head pose landmarks: {e}")
        
        return None
    
    def calculate_head_pose(self, image_points, face_roi):
        """Calculate head pose angles (yaw, pitch, roll)"""
        if image_points is None:
            return None, None, None, False
            
        if len(image_points) != 6:
            return None, None, None, False
            
        try:
            h, w = face_roi.shape[:2]
            
            # Create camera matrix for this face ROI
            camera_matrix = np.array([
                [float(w), 0.0, float(w/2)],
                [0.0, float(w), float(h/2)],
                [0.0, 0.0, 1.0]
            ], dtype=np.float64)
            
            # Ensure consistent data types
            model_points = self.model_points.astype(np.float64)
            image_points = image_points.astype(np.float64)
            dist_coeffs = np.zeros((4, 1), dtype=np.float64)
            
            # Solve PnP
            success, rotation_vector, translation_vector = cv2.solvePnP(
                model_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                flags=cv2.SOLVEPNP_ITERATIVE
            )
            
            if not success:
                return None, None, None, False
            
            # Convert to rotation matrix
            rotation_matrix, _ = cv2.Rodrigues(rotation_vector)
            
            # Calculate Euler angles
            pitch, yaw, roll = self.rotation_matrix_to_euler_angles(rotation_matrix)
            
            # Check stability
            is_stable = (abs(yaw) < self.rotation_threshold and 
                        abs(pitch) < self.rotation_threshold and 
                        abs(roll) < self.rotation_threshold)
            
            return float(yaw), float(pitch), float(roll), bool(is_stable)
            
        except Exception as e:
            print(f"Error calculating head pose: {e}")
            return None, None, None, False
    
    def rotation_matrix_to_euler_angles(self, R):
        """Convert rotation matrix to Euler angles"""
        try:
            sy = math.sqrt(R[0, 0] * R[0, 0] + R[1, 0] * R[1, 0])
            
            singular = sy < 1e-6
            
            if not singular:
                x = math.atan2(R[2, 1], R[2, 2])  # Roll
                y = math.atan2(-R[2, 0], sy)      # Pitch  
                z = math.atan2(R[1, 0], R[0, 0])  # Yaw
            else:
                x = math.atan2(-R[1, 2], R[1, 1])
                y = math.atan2(-R[2, 0], sy)
                z = 0
            
            return math.degrees(x), math.degrees(y), math.degrees(z)
            
        except Exception as e:
            print(f"Error in euler conversion: {e}")
            return 0.0, 0.0, 0.0
    
    def should_maintain_login(self, left_right_turn, up_down_nod, side_tilt):
        """INCREASED TOLERANCE - Allow more head movement"""
        if None in [left_right_turn, up_down_nod, side_tilt]:
            return True
            
        # VERY RELAXED thresholds - allow significant head movement
        return (abs(left_right_turn) < 80 and abs(up_down_nod) < 60 and abs(side_tilt) < 50)
    
    def get_head_status(self, left_right_turn, up_down_nod, side_tilt):
        """Better categorization with more tolerance"""
        if None in [left_right_turn, up_down_nod, side_tilt]:
            return "Unknown"
            
        max_angle = max(abs(left_right_turn), abs(up_down_nod), abs(side_tilt))
        
        if max_angle < 20:  # Very stable
            return "Face Forward"
        elif max_angle < 45:  # Normal movement
            return "Natural Head Movement"
        elif max_angle < 75:  # Larger movement but still OK
            return "Head Turning - Keep Logged In"
        else:
            return "Extreme Head Movement"

# ENHANCED PERSON TRACKER WITH ID PERSISTENCE
class PersonTracker:
    def __init__(self, max_disappeared=30, max_distance=150):
        self.next_id = 0
        self.persons = {}  # ID -> {key_landmarks, face_roi, name, similarity, frames_active}
        self.disappeared = {}  # ID -> frame count
        self.max_disappeared = max_disappeared
        self.max_distance = max_distance
        self.frame_count = 0
        
        # NEW: ID Persistence mechanism
        self.person_id_history = {}  # name -> person_id (persistent mapping)
        self.inactive_persons = {}  # Temporarily store inactive persons
        self.face_encodings_cache = {}  # person_id -> face_encoding for matching
        self.reidentification_threshold = 0.85  # Threshold for re-identification

    def deregister(self, person_id):
        """Remove a person from tracking but preserve their ID mapping"""
        if person_id in self.persons:
            person_info = self.persons[person_id]
            name = person_info.get('name', 'Unknown Person')
            
            # Store in inactive persons for potential re-identification
            if name != 'Unknown Person':
                self.inactive_persons[person_id] = {
                    'name': name,
                    'last_seen': self.frame_count,
                    'face_encoding': self.face_encodings_cache.get(person_id),
                    'key_landmarks': person_info.get('key_landmarks'),
                    'last_bbox': person_info.get('bbox')
                }
                print(f"Person ID {person_id} ({name}) moved to inactive list")
            
            del self.persons[person_id]
        
        if person_id in self.disappeared:
            del self.disappeared[person_id]

    def get_person_info(self, person_id):
        """Get person information by ID"""
        return self.persons.get(person_id, None)

    def update_person_recognition(self, person_id, name, similarity, confidence=None):
        """Update person's recognition information"""
        if person_id in self.persons:
            # Update person ID history mapping
            if name != 'Unknown Person':
                self.person_id_history[name] = person_id
                
            self.persons[person_id]['name'] = name
            self.persons[person_id]['similarity'] = similarity
            if confidence is not None:
                self.persons[person_id]['confidence'] = confidence

    def calculate_distance(self, point1, point2):
        """Calculate Euclidean distance between two points"""
        if not point1 or not point2:
            return float('inf')
        
        x1, y1 = point1
        x2, y2 = point2
        return math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)

    def get_landmark_centroid(self, landmarks):
        """Calculate centroid from facial landmarks"""
        if not landmarks:
            return None
            
        x_sum = sum(point[0] for point in landmarks)
        y_sum = sum(point[1] for point in landmarks)
        return (x_sum // len(landmarks), y_sum // len(landmarks))
    
    def calculate_landmark_distance(self, landmarks1, landmarks2):
        """Calculate distance between two sets of landmarks"""
        if not landmarks1 or not landmarks2 or len(landmarks1) != len(landmarks2):
            return float('inf')
            
        total_distance = 0
        for (x1, y1), (x2, y2) in zip(landmarks1, landmarks2):
            total_distance += math.sqrt((x1 - x2) ** 2 + (y1 - y2) ** 2)
            
        return total_distance / len(landmarks1)  # Average distance
    
    def try_reidentify_person(self, detection, face_encoding=None):
        """Try to re-identify a person from inactive list"""
        if not self.inactive_persons or not detection:
            return None
            
        detection_landmarks = detection.get('key_landmarks', [])
        detection_centroid = detection.get('centroid')
        
        best_match_id = None
        best_similarity = 0.0
        
        # Check inactive persons for potential matches
        for inactive_id, inactive_data in list(self.inactive_persons.items()):
            # Skip if too old (more than 100 frames ago)
            if self.frame_count - inactive_data['last_seen'] > 100:
                continue
                
            similarity_score = 0.0
            
            # 1. Compare face encodings if available
            if face_encoding is not None and inactive_data.get('face_encoding') is not None:
                try:
                    face_similarity = cosine_similarity(
                        face_encoding.reshape(1, -1),
                        inactive_data['face_encoding'].reshape(1, -1)
                    )[0][0]
                    similarity_score += face_similarity * 0.6  # 60% weight
                except:
                    pass
            
            # 2. Compare landmarks if available
            if detection_landmarks and inactive_data.get('key_landmarks'):
                landmark_distance = self.calculate_landmark_distance(
                    detection_landmarks, inactive_data['key_landmarks']
                )
                if landmark_distance < self.max_distance:
                    landmark_similarity = max(0, (self.max_distance - landmark_distance) / self.max_distance)
                    similarity_score += landmark_similarity * 0.3  # 30% weight
            
            # 3. Compare position proximity
            if detection_centroid and inactive_data.get('last_bbox'):
                last_bbox = inactive_data['last_bbox']
                last_centroid = (last_bbox[0] + last_bbox[2]//2, last_bbox[1] + last_bbox[3]//2)
                position_distance = self.calculate_distance(detection_centroid, last_centroid)
                if position_distance < self.max_distance * 2:  # Allow larger distance for position
                    position_similarity = max(0, (self.max_distance * 2 - position_distance) / (self.max_distance * 2))
                    similarity_score += position_similarity * 0.1  # 10% weight
            
            # Check if this is the best match
            if similarity_score > best_similarity and similarity_score > self.reidentification_threshold:
                best_similarity = similarity_score
                best_match_id = inactive_id
        
        if best_match_id is not None:
            print(f"Re-identified person ID {best_match_id} with similarity {best_similarity:.3f}")
            # Remove from inactive list
            del self.inactive_persons[best_match_id]
            return best_match_id
            
        return None
    
    def register(self, key_landmarks, face_roi, bbox, confidence, face_encoding=None):
        """Register a new person with unique ID using facial landmarks"""
        centroid = self.get_landmark_centroid(key_landmarks) if key_landmarks else self.get_bbox_centroid(bbox)
        
        # Prepare detection data for re-identification
        detection_data = {
            'key_landmarks': key_landmarks,
            'centroid': centroid,
            'bbox': bbox,
            'face_encoding': face_encoding
        }
        
        # Try to re-identify from inactive persons first
        reidentified_id = self.try_reidentify_person(detection_data, face_encoding)
        
        if reidentified_id is not None:
            # Reactivate the person with their original ID
            person_id = reidentified_id
            print(f"Reactivating person ID {person_id}")
        else:
            # Create new ID
            person_id = self.next_id
            self.next_id += 1
            print(f"Registering new person ID {person_id}")

        self.persons[person_id] = {
            'key_landmarks': key_landmarks,
            'centroid': centroid,
            'face_roi': face_roi,
            'bbox': bbox,
            'name': 'Unknown Person',
            'similarity': 0.0,
            'confidence': confidence,
            'frames_active': 1,
            'last_seen': self.frame_count
        }
        
        # Cache face encoding for this person
        if face_encoding is not None:
            self.face_encodings_cache[person_id] = face_encoding
            
        return person_id
    
    def get_bbox_centroid(self, bbox):
        """Fallback to bbox centroid if landmarks unavailable"""
        x, y, w, h = bbox
        return (x + w // 2, y + h // 2)
    
    def update(self, face_detections, face_encodings=None):
        """Update tracking with new detections using facial landmarks"""
        self.frame_count += 1
        
        # Prepare detection data with provided landmarks
        detection_data = []
        for i, detection in enumerate(face_detections):
            key_landmarks = detection.get('key_landmarks', [])
            centroid = self.get_landmark_centroid(key_landmarks) if key_landmarks else self.get_bbox_centroid(detection['bbox'])
            
            face_encoding = None
            if face_encodings and i < len(face_encodings):
                face_encoding = face_encodings[i]
            
            detection_data.append({
                'key_landmarks': key_landmarks,
                'centroid': centroid,
                'face': detection['face'],
                'bbox': detection['bbox'],
                'confidence': detection.get('confidence', 0.5),
                'full_landmarks': detection.get('full_landmarks'),
                'face_encoding': face_encoding
            })
        
        if len(detection_data) == 0:
            for person_id in list(self.persons.keys()):
                if person_id not in self.disappeared:
                    self.disappeared[person_id] = 0
                self.disappeared[person_id] += 1
                if self.disappeared[person_id] > self.max_disappeared:
                    self.deregister(person_id)
            return {}
        
        if len(self.persons) == 0:
            tracking_results = {}
            for i, detection in enumerate(detection_data):
                person_id = self.register(
                    detection['key_landmarks'], 
                    detection['face'], 
                    detection['bbox'], 
                    detection['confidence'],
                    detection.get('face_encoding')
                )
                tracking_results[person_id] = detection
            return tracking_results
        
        existing_persons = list(self.persons.values())
        person_ids = list(self.persons.keys())
        tracking_results = {}
        used_detection_indices = set()
        
        for person_idx, person_id in enumerate(person_ids):
            if person_id in self.disappeared:
                continue
                
            person_info = existing_persons[person_idx]
            min_distance = float('inf')
            min_index = -1
            
            for i, detection in enumerate(detection_data):
                if i in used_detection_indices:
                    continue
                    
                # Multi-factor matching
                distance = float('inf')
                
                # 1. Landmark-based matching (primary)
                if person_info['key_landmarks'] and detection['key_landmarks']:
                    landmark_distance = self.calculate_landmark_distance(
                        person_info['key_landmarks'], detection['key_landmarks']
                    )
                    distance = landmark_distance
                
                # 2. Face encoding matching (if available)
                if (person_id in self.face_encodings_cache and 
                    detection.get('face_encoding') is not None):
                    try:
                        face_similarity = cosine_similarity(
                            self.face_encodings_cache[person_id].reshape(1, -1),
                            detection['face_encoding'].reshape(1, -1)
                        )[0][0]
                        
                        # Convert similarity to distance (higher similarity = lower distance)
                        face_distance = (1.0 - face_similarity) * 100
                        
                        # Combine with landmark distance
                        if distance != float('inf'):
                            distance = (distance + face_distance) / 2
                        else:
                            distance = face_distance
                            
                    except Exception as e:
                        print(f"Error in face encoding comparison: {e}")
                
                # 3. Fallback to centroid matching
                if distance == float('inf'):
                    distance = self.calculate_distance(
                        person_info['centroid'], detection['centroid']
                    )
                
                if distance < min_distance and distance < self.max_distance:
                    min_distance = distance
                    min_index = i
            
            if min_index != -1:
                detection = detection_data[min_index]
                self.persons[person_id]['key_landmarks'] = detection['key_landmarks']
                self.persons[person_id]['centroid'] = detection['centroid']
                self.persons[person_id]['face_roi'] = detection['face']
                self.persons[person_id]['bbox'] = detection['bbox']
                self.persons[person_id]['confidence'] = detection['confidence']
                self.persons[person_id]['frames_active'] += 1
                self.persons[person_id]['last_seen'] = self.frame_count
                
                # Update face encoding cache
                if detection.get('face_encoding') is not None:
                    self.face_encodings_cache[person_id] = detection['face_encoding']
                
                tracking_results[person_id] = detection
                used_detection_indices.add(min_index)
                
                if person_id in self.disappeared:
                    del self.disappeared[person_id]
            else:
                if person_id not in self.disappeared:
                    self.disappeared[person_id] = 0
                self.disappeared[person_id] += 1
        
        # Register new detections
        for i, detection in enumerate(detection_data):
            if i not in used_detection_indices:
                person_id = self.register(
                    detection['key_landmarks'], 
                    detection['face'], 
                    detection['bbox'], 
                    detection['confidence'],
                    detection.get('face_encoding')
                )
                tracking_results[person_id] = detection
        
        # Clean up disappeared persons
        for person_id in list(self.disappeared.keys()):
            if self.disappeared[person_id] > self.max_disappeared:
                self.deregister(person_id)
        
        return tracking_results

# WEB REGISTRATION SERVER
class WebRegistrationServer:
    def __init__(self, face_recognition_app):
        self.face_app = face_recognition_app
        self.registration_photos = []
        self.current_photo_count = 0
        
    def start_server(self):
        """Start Flask server in a separate thread"""
        def run_server():
            app.run(host='localhost', port=8000, debug=False, use_reloader=False)
        
        server_thread = threading.Thread(target=run_server, daemon=True)
        server_thread.start()
        print("Registration server started at http://localhost:8000")

# Flask routes (keeping existing routes)
@app.route('/')
def index():
    return render_template('register.html')

@app.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        name = request.form.get('name')
        if name:
            global registration_name, registration_mode, completed_angles
            registration_name = name.strip()
            registration_mode = True
            completed_angles = []
            
            # Pause main recognition system
            if global_camera:
                global_camera.pause_recognition = True
                
            return redirect(url_for('capture'))
    return render_template('register.html')

@app.route('/capture')
def capture():
    return render_template('capture.html', name=registration_name, angles=registration_angles)

@app.route('/get_camera_frame')
def get_camera_frame():
    """Return current camera frame as base64 with face detection overlay"""
    global global_camera
    if global_camera and global_camera.cap:
        ret, frame = global_camera.cap.read()
        if ret:
            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Add face detection overlay for registration
            face_info = global_camera.detect_faces_mediapipe(rgb_frame)
            
            # Draw on RGB frame
            for face_data in face_info:
                bbox = face_data['bbox']
                x, y, width, height = bbox
                confidence = face_data['confidence']
                
                # Draw face bounding box (RGB colors)
                cv2.rectangle(rgb_frame, (x, y), (x + width, y + height), (0, 255, 0), 2)
                
                # Draw confidence
                cv2.putText(rgb_frame, f'Face: {confidence:.2f}', (x, y-10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                
                # Draw landmarks
                if 'full_landmarks' in face_data and global_camera.mp_drawing:
                    global_camera.mp_drawing.draw_landmarks(
                        image=rgb_frame,
                        landmark_list=face_data['full_landmarks'],
                        connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                        landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                            color=(255, 255, 255), thickness=1, circle_radius=1
                        ),
                        connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
                    )
            
            # Add angle guidelines (RGB colors)
            h, w = rgb_frame.shape[:2]
            center_x, center_y = w//2, h//2
            
            # Draw center guidelines
            cv2.line(rgb_frame, (center_x, 0), (center_x, h), (100, 100, 255), 1)
            cv2.line(rgb_frame, (0, center_y), (w, center_y), (100, 100, 255), 1)
            
            # Add instruction text
            current_angle = registration_angles[len(completed_angles)] if len(completed_angles) < 4 else "Complete"
            cv2.putText(rgb_frame, f'Position: {current_angle}', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Convert back to BGR for encoding
            frame_to_encode = cv2.cvtColor(rgb_frame, cv2.COLOR_RGB2BGR)
            _, buffer = cv2.imencode('.jpg', frame_to_encode)
            import base64
            frame_base64 = base64.b64encode(buffer).decode('utf-8')
            return jsonify({'frame': frame_base64})
    return jsonify({'frame': None})

@app.route('/capture_photo', methods=['POST'])
def capture_photo():
    """Capture and save only the detected face region in its original RGB format."""
    global global_camera, registration_photos, registration_name, completed_angles
    
    if global_camera and global_camera.cap:
        ret, frame = global_camera.cap.read()
        if ret:
            # Flip frame for consistency
            frame = cv2.flip(frame, 1)
            
            # Convert to RGB for processing
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Detect face in the frame
            face_info = global_camera.detect_faces_mediapipe(rgb_frame)
            
            if len(face_info) > 0:
                # Get the best face (highest confidence)
                best_face = max(face_info, key=lambda x: x['confidence'])
                face_roi_rgb = best_face['face']  # This is already the RGB cropped face
                confidence = best_face['confidence']
                
                if confidence < 0.7:
                    return jsonify({
                        'success': False,
                        'message': f'Face confidence too low ({confidence:.2f}). Please position your face clearly.'
                    })
                
                # --- Main Change: Save the RGB image directly using PIL ---
                
                # Create person folder
                person_folder = os.path.join('./face_database', registration_name)
                os.makedirs(person_folder, exist_ok=True)
                
                # Define file path
                current_angle = registration_angles[len(completed_angles)]
                timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
                filename = f'{registration_name}_{current_angle.replace(" ", "_")}_{timestamp}.jpg'
                filepath = os.path.join(person_folder, filename)
                
                try:
                    # Create a PIL Image from the RGB numpy array
                    pil_image = Image.fromarray(face_roi_rgb)
                    # Save the image using PIL, which correctly handles RGB
                    pil_image.save(filepath)
                    success = True
                except Exception as e:
                    print(f"Error saving image with PIL: {e}")
                    success = False
                
                if success:
                    registration_photos.append(filepath)
                    completed_angles.append(current_angle)
                    
                    return jsonify({
                        'success': True,
                        'message': f'{current_angle} captured successfully!',
                        'photo_count': len(registration_photos),
                        'confidence': confidence,
                        'current_angle': current_angle,
                        'completed_angles': completed_angles,
                        'next_angle': registration_angles[len(completed_angles)] if len(completed_angles) < 4 else None
                    })
                else:
                    return jsonify({
                        'success': False,
                        'message': 'Failed to save face image. Please try again.'
                    })
            else:
                return jsonify({
                    'success': False,
                    'message': 'No face detected. Please position your face clearly in the camera.'
                })
    
    return jsonify({'success': False, 'message': 'Camera not available.'})


@app.route('/complete_registration', methods=['POST'])
def complete_registration():
    """Complete the registration process"""
    global registration_photos, registration_name, global_camera, registration_mode, completed_angles
    
    if len(registration_photos) >= 4 and len(completed_angles) >= 4:
        try:
            # Process the photos and create face encodings
            successful_encodings = []
            
            for photo_path in registration_photos:
                # Since we saved BGR face regions, directly process them
                face_roi = cv2.imread(photo_path)
                if face_roi is not None:
                    encoding = global_camera.get_face_encoding(face_roi)
                    if encoding is not None:
                        successful_encodings.append(encoding)
            
            if len(successful_encodings) >= 3:  # Need at least 3 successful encodings
                # Average all encodings
                averaged_encoding = np.mean(successful_encodings, axis=0)
                normalized_encoding = averaged_encoding / np.linalg.norm(averaged_encoding)
                
                # Add to face database
                global_camera.face_database[registration_name] = normalized_encoding
                
                # Reset registration data
                registration_photos.clear()
                completed_angles.clear()
                registration_mode = False
                
                # Resume main recognition system
                if global_camera:
                    global_camera.pause_recognition = False
                
                return jsonify({
                    'success': True,
                    'message': f'{registration_name} registered successfully with {len(successful_encodings)} face encodings!'
                })
            else:
                return jsonify({
                    'success': False,
                    'message': 'Not enough valid face encodings. Please try again with clearer photos.'
                })
                
        except Exception as e:
            return jsonify({
                'success': False,
                'message': f'Registration failed: {str(e)}'
            })
    
    return jsonify({
        'success': False,
        'message': 'Need exactly 4 photos from different angles to complete registration.'
    })

@app.route('/cancel_registration', methods=['POST'])
def cancel_registration():
    """Cancel registration and resume main camera"""
    global registration_mode, registration_photos, completed_angles, global_camera
    
    registration_mode = False
    registration_photos.clear()
    completed_angles.clear()
    
    # Resume main recognition system
    if global_camera:
        global_camera.pause_recognition = False
    
    return jsonify({'success': True, 'message': 'Registration cancelled'})

# ATTENDANCE LOGGER CLASS
class AttendanceLogger:
    def __init__(self, log_file='attendance_log.xlsx'):
        self.log_file = log_file
        if os.path.exists(log_file):
            self.df = pd.read_excel(log_file)
        else:
            self.df = pd.DataFrame(columns=['Name', 'Login Time', 'Logout Time', 'Duration (minutes)', 'Date'])

    def log_login(self, name):
        login_time = datetime.datetime.now()
        date_str = login_time.strftime('%Y-%m-%d')
        
        new_row = pd.DataFrame([{
            'Name': name,
            'Login Time': login_time.strftime('%Y-%m-%d %H:%M:%S'),
            'Logout Time': '',
            'Duration (minutes)': 0,
            'Date': date_str
        }])
        self.df = pd.concat([self.df, new_row], ignore_index=True)
        self.save()
        return login_time

    def log_logout(self, name):
        logout_time = datetime.datetime.now()
        
        mask = (self.df['Name'] == name) & (self.df['Logout Time'] == '')
        if mask.any():
            last_login_idx = self.df[mask].index[-1]
            login_time_str = self.df.at[last_login_idx, 'Login Time']
            login_time = datetime.datetime.strptime(login_time_str, '%Y-%m-%d %H:%M:%S')
            
            duration = (logout_time - login_time).total_seconds() / 60.0
            self.df.at[last_login_idx, 'Logout Time'] = logout_time.strftime('%Y-%m-%d %H:%M:%S')
            self.df.at[last_login_idx, 'Duration (minutes)'] = round(duration, 2)
            self.save()
            return duration
        else:
            return None

    def get_total_time_today(self, name):
        today = datetime.datetime.now().strftime('%Y-%m-%d')
        mask = (self.df['Name'] == name) & (self.df['Date'] == today) & (self.df['Duration (minutes)'] > 0)
        if mask.any():
            return self.df[mask]['Duration (minutes)'].sum()
        return 0

    def save(self):
        self.df.to_excel(self.log_file, index=False)

# MAIN FACE RECOGNITION APP WITH 5-FRAME ROI EVALUATION
class FaceRecognitionApp:
    def __init__(self):
        # Initialize global camera reference
        global global_camera
        global_camera = self
        self.person_tracker = PersonTracker(max_disappeared=950, max_distance=200)
        self.person_recognition_history = {}  # person_id -> [(name, similarity), ...]
        
        # ROI entry tracking for 5-frame evaluation
        self.person_roi_entry = {}  # person_id -> frame_count when first entered ROI
        self.person_evaluation_complete = {}  # person_id -> bool (evaluation done)
        self.roi_evaluation_frames = 5  # Number of frames to evaluate on ROI entry
        
        # Initialize camera variables first
        self.cap = None
        self.most_recent_capture_arr = None
        self.most_recent_capture_pil = None
        self.auto_login_threshold = 0.56  # Higher threshold for finalized identities
        self.auto_login_cooldown = {}
        self.last_auto_login_time = {}
        self.pause_recognition = False  # New: pause recognition during registration
        
        # Initialize attendance logger
        self.attendance_logger = AttendanceLogger()
        
        # Login/logout tracking - DON'T clear person IDs on logout
        self.logged_in_users = {}
        self.logged_out_users = {}  # Track logged out users but keep their IDs
        
        # Initialize MediaPipe Face Mesh for landmarks and detection with error handling
        try:
            self.mp_face_mesh = mp.solutions.face_mesh
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                static_image_mode=False,
                max_num_faces=5,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            self.mp_drawing = mp.solutions.drawing_utils
            print("✓ MediaPipe FaceMesh initialized successfully")
        except Exception as e:
            print(f"❌ Error initializing MediaPipe FaceMesh: {e}")
            self.face_mesh = None
            self.mp_face_mesh = None
            self.mp_drawing = None
        
        # Load FaceNet model using DeepFace
        self.facenet_model = self.load_pretrained_facenet_model()
        
        # Load face database from uploaded folder
        self.face_database = self.load_face_database('./face_database')
        print(f"Loaded {len(self.face_database)} people in database")
        
        # Recognition parameters
        self.recognition_threshold = 0.50
        self.face_recognition_cache = {}
        self.recognition_history = {}
        
        # ROI parameters
        self.roi_x1 = 0
        self.roi_y1 = 0
        self.roi_x2 = 0
        self.roi_y2 = 0
        self.roi_enabled = True
        
        # HEAD POSE ESTIMATOR INTEGRATION
        self.head_pose_estimator = HeadPoseEstimator()
        self.person_head_poses = {}  # person_id -> head pose data
        
        # Initialize web server
        self.web_server = WebRegistrationServer(self)
        self.web_server.start_server()
        
        # GUI setup
        self.main_window = tk.Tk()
        self.main_window.title("Integrated Face Recognition & Registration System")
        self.main_window.state('zoomed')
        
        self.setup_gui()

    def finalize_name_assignment(self, person_id):
        """Finalize name assignment after 5-frame ROI evaluation period"""
        if person_id not in self.person_recognition_history:
            return 'Unknown Person', 0.0
        
        history = self.person_recognition_history[person_id]
        
        if len(history) < 3:  # Need at least 3 frames of data
            print(f"⚠️ ID {person_id} insufficient data for evaluation - assigning Unknown")
            return 'Unknown Person', 0.0
        
        # Take the recognition data from evaluation period only
        evaluation_data = history[-self.roi_evaluation_frames:] if len(history) >= self.roi_evaluation_frames else history
        
        # Count recognitions for each name
        name_counts = {}
        total_similarity = 0
        valid_recognitions = 0
        
        for name, similarity in evaluation_data:
            if name != 'Unknown Person' and similarity >= 0.65:  # Lower threshold for evaluation
                name_counts[name] = name_counts.get(name, 0) + 1
                total_similarity += similarity
                valid_recognitions += 1
        
        if name_counts and valid_recognitions >= 3:  # Need at least 3 valid recognitions
            most_frequent_name = max(name_counts, key=name_counts.get)
            frequency = name_counts[most_frequent_name]
            
            # Check if majority recognition (60% or more)
            if frequency >= len(evaluation_data) * 0.6 and total_similarity / valid_recognitions >= 0.55:
                avg_similarity = total_similarity / valid_recognitions
                print(f"✅ Assigning name '{most_frequent_name}' to ID {person_id} based on {frequency}/{len(evaluation_data)} recognitions (avg: {avg_similarity:.3f})")
                return most_frequent_name, avg_similarity
        
        print(f"⚠️ ID {person_id} did not meet consistency requirements - assigning Unknown")
        return 'Unknown Person', 0.0

    def recognize_face(self, face_roi, person_id):
        """Simplified face recognition for evaluation period"""
        if self.facenet_model is None or len(self.face_database) == 0:
            return 'Unknown Person', 0.0
        
        try:
            current_encoding = self.get_face_encoding(face_roi)
            if current_encoding is None:
                return 'Unknown Person', 0.0
            
            best_match = None
            best_similarity = 0.0
            
            # Compare with all known faces in database
            for name, stored_encoding in self.face_database.items():
                try:
                    similarity = cosine_similarity(
                        current_encoding.reshape(1, -1),
                        stored_encoding.reshape(1, -1)
                    )[0][0]
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = name
                except:
                    continue
            
            # Use recognition threshold
            recognition_threshold = 0.55
            
            # Determine current frame recognition
            current_frame_name = 'Unknown Person'
            if best_match and best_similarity >= recognition_threshold:
                current_frame_name = best_match
            
            return current_frame_name, best_similarity
            
        except Exception as e:
            print(f"Error in face recognition: {e}")
            return 'Unknown Person', 0.0

    # HEAD POSE METHODS
    def get_head_pose_for_person(self, face_roi, person_id):
        """Get head pose data with simple terms"""
        try:
            if face_roi is None or face_roi.size == 0:
                return None
                
            pose_landmarks = self.head_pose_estimator.extract_head_pose_landmarks(face_roi)
            
            if pose_landmarks is not None:
                # Calculate head pose
                yaw, pitch, roll, is_stable = self.head_pose_estimator.calculate_head_pose(pose_landmarks, face_roi)
                
                # Store pose data with SIMPLE TERMS
                pose_data = {
                    'left_right_turn': yaw,      # Simple name for yaw
                    'up_down_nod': pitch,        # Simple name for pitch  
                    'side_tilt': roll,           # Simple name for roll
                    'is_stable': is_stable,
                    'should_maintain_login': self.head_pose_estimator.should_maintain_login(yaw, pitch, roll),
                    'status': self.head_pose_estimator.get_head_status(yaw, pitch, roll),
                    'timestamp': datetime.datetime.now()
                }
                
                # Update person's pose history
                if person_id not in self.person_head_poses:
                    self.person_head_poses[person_id] = []
                
                self.person_head_poses[person_id].append(pose_data)
                
                # Keep more history for better stability
                if len(self.person_head_poses[person_id]) > 10:  # Increased from 5
                    self.person_head_poses[person_id] = self.person_head_poses[person_id][-10:]
                
                return pose_data
                
        except Exception as e:
            print(f"Error getting head pose: {e}")
        
        return None

    def check_auto_logout_with_head_pose(self, currently_detected_names):
        """
        MODIFIED: Head movement now only protects login status if the user is INSIDE the ROI.
        """
        if self.pause_recognition:
            return
            
        current_time = datetime.datetime.now()
        
        # HEAD MOVEMENT NOW PROTECTS AGAINST LOGOUT, BUT ONLY IF INSIDE THE ROI
        for person_id, person_info in self.person_tracker.persons.items():
            name = person_info.get('name', 'Unknown Person')
            bbox = person_info.get('bbox')
            
            # Condition: Only proceed if the user is a known, logged-in user AND is inside the ROI
            if name == 'Unknown Person' or name not in self.logged_in_users or not self.is_face_in_roi(bbox):
                continue
            
            # Check recent head pose data for users inside the ROI
            recent_poses = self.person_head_poses.get(person_id, [])
            head_movement_detected = False
            
            if recent_poses:
                for pose in recent_poses[-5:]:  # Check the last 5 poses
                    status = pose.get('status', 'Unknown')
                    if any(indicator in status for indicator in ['Movement', 'Turning', 'Head']):
                        head_movement_detected = True
                        break
            
            # If head movement is detected for a user inside the ROI, extend their login time
            if head_movement_detected:
                self.last_auto_login_time[name] = current_time
                print(f"Head movement detected for {name} inside ROI - protecting login")
        
        # INCREASED timeout and better protection for users who disappear
        to_logout = []
        for logged_in_user in list(self.logged_in_users.keys()):
            if logged_in_user not in currently_detected_names:
                last_seen = self.last_auto_login_time.get(logged_in_user, None)
                
                # Find person ID for head movement check
                user_person_id = None
                for pid, pinfo in self.person_tracker.persons.items():
                    if pinfo.get('name') == logged_in_user:
                        user_person_id = pid
                        break
                
                # EXTENDED timeout - especially during head movement
                timeout_seconds = 0  # Base timeout for disappearance
                
                if user_person_id and user_person_id in self.person_head_poses:
                    recent_poses = self.person_head_poses[user_person_id][-3:]
                    if recent_poses:
                        latest_status = recent_poses[-1].get('status', 'Unknown')
                        # This part still helps extend the timeout if the last known action was head movement
                        if any(indicator in latest_status for indicator in ['Movement', 'Turning', 'Head']):
                            timeout_seconds = 5  # Much longer timeout during movement
                            print(f"Extended timeout for {logged_in_user} due to recent head movement")
                
                if last_seen is None or (current_time - last_seen).total_seconds() > timeout_seconds:
                    to_logout.append(logged_in_user)
        
        for name in to_logout:
            print(f"[TIMEOUT_LOGOUT] Logging out {name} after extended timeout")
            self.logout_user(name)



    def immediate_logout_outside_roi(self, tracking_results):
        """
        MODIFIED: Immediately logs out any user who is detected outside the ROI.
        This function now ignores head movement and only considers position for immediate logout.
        """
        # Iterate through all currently tracked persons from this frame
        for person_id, detection in tracking_results.items():
            person_info = self.person_tracker.get_person_info(person_id)
            if not person_info:
                continue
                
            name = person_info.get('name', 'Unknown Person')
            
            # Check for the specific condition: a known, logged-in user is now outside the ROI
            if (name in self.logged_in_users and
                name != 'Unknown Person' and
                not self.is_face_in_roi(detection['bbox'])):
                
                # If the condition is met, log them out immediately without waiting
                print(f"[IMMEDIATE_ROI_LOGOUT] {name} is outside the ROI. Logging out now.")
                self.logout_user_preserve_id(name, person_id)
                
                # Apply a cooldown period to prevent an instant re-login if the user is near the ROI border
                current_time = datetime.datetime.now()
                self.auto_login_cooldown[name] = current_time


    def load_pretrained_facenet_model(self):
        """Load FaceNet using DeepFace"""
        try:
            print("Initializing DeepFace FaceNet model...")
            test_img = np.zeros((160, 160, 3), dtype=np.uint8)
            DeepFace.represent(test_img, model_name='Facenet', enforce_detection=False)
            print("✓ Successfully initialized DeepFace FaceNet model")
            return True
        except Exception as e:
            print(f"Error initializing DeepFace FaceNet model: {e}")
            return None

    def preprocess_face_for_facenet(self, face_image):
        """Improved face preprocessing for FaceNet"""
        try:
            if face_image.shape[0] < 50 or face_image.shape[1] < 50:
                return None
            
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            resized = cv2.resize(rgb_image, (160, 160), interpolation=cv2.INTER_CUBIC)
            
            lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            
            normalized = enhanced_rgb.astype(np.float32) / 255.0
            return normalized
            
        except Exception as e:
            print(f"Error in face preprocessing: {e}")
            return None

    def get_face_encoding(self, face_image):
        """Generate FaceNet face encoding from image using DeepFace"""
        if self.facenet_model is None:
            print("FaceNet model not loaded")
            return None
            
        try:
            preprocessed = self.preprocess_face_for_facenet(face_image)
            if preprocessed is None:
                return None
            
            embedding_objs = DeepFace.represent(
                img_path=preprocessed,
                model_name='Facenet',
                enforce_detection=False,
                detector_backend='skip'
            )
            
            if embedding_objs and len(embedding_objs) > 0:
                embedding = np.array(embedding_objs[0]['embedding'])
                normalized_embedding = embedding / np.linalg.norm(embedding)
                return normalized_embedding
            else:
                return None
                
        except Exception as e:
            print(f"Error generating DeepFace FaceNet encoding: {e}")
            return None

    def calculate_roi(self, frame_width, frame_height):
        """Calculate Region of Interest coordinates"""
        roi_width = int(frame_width * 0.35)
        roi_height = int(frame_height * 0.6)  
        
        self.roi_x1 = (frame_width - roi_width) // 2
        self.roi_y1 = (frame_height - roi_height) // 2  
        self.roi_x2 = self.roi_x1 + roi_width
        self.roi_y2 = self.roi_y1 + roi_height

    def draw_roi(self, image):
        """Draw ROI rectangle on the image"""
        if self.roi_enabled and not self.pause_recognition:
            cv2.rectangle(image, (self.roi_x1, self.roi_y1), (self.roi_x2, self.roi_y2), (255, 0, 0), 4)
            
            label_text = "FACE DETECTION ZONE"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.9
            font_thickness = 2
            
            (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)
            text_x = self.roi_x1 + (self.roi_x2 - self.roi_x1 - text_width) // 2
            text_y = self.roi_y1 - 15
            
            cv2.rectangle(image, (text_x - 5, text_y - text_height - 5), 
                        (text_x + text_width + 5, text_y + 5), (0, 0, 0), -1)
            cv2.putText(image, label_text, (text_x, text_y), font, font_scale, (255, 0, 0), font_thickness)

    def detect_faces_mediapipe(self, rgb_image):
        """Detect faces using MediaPipe FaceMesh and extract landmarks"""
        if self.face_mesh is None:
            print("MediaPipe FaceMesh not available")
            return []
            
        try:
            h, w, _ = rgb_image.shape
            results = self.face_mesh.process(rgb_image)
            
            face_info = []
            if results.multi_face_landmarks:
                key_indices = [1, 33, 61, 199, 263, 291]
                for face_landmarks in results.multi_face_landmarks:
                    # Get bbox from landmarks
                    x_coords = [lm.x * w for lm in face_landmarks.landmark]
                    y_coords = [lm.y * h for lm in face_landmarks.landmark]
                    x = max(0, min(x_coords))
                    y = max(0, min(y_coords))
                    width = max(x_coords) - x
                    height = max(y_coords) - y
                    
                    # Expand bbox a bit
                    margin = 0
                    x = max(0, x - margin)
                    y = max(0, y - margin)
                    width = min(w - x, width + 2 * margin)
                    height = min(h - y, height + 2 * margin)
                    
                    if width > 80 and height > 80:
                        # Crop face ROI (which is already in RGB)
                        face_roi_rgb = rgb_image[int(y):int(y + height), int(x):int(x + width)]
                        
                        # Key landmarks in pixel coordinates
                        key_landmarks_pixel = [(int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)) for idx in key_indices]
                        
                        # *** FIX IS HERE ***
                        # The incorrect conversion to BGR has been removed.
                        # We now pass the RGB cropped face directly.
                        face_info.append({
                            'face': face_roi_rgb,
                            'bbox': (int(x), int(y), int(width), int(height)),
                            'confidence': 0.9,  # Approximate for FaceMesh
                            'full_landmarks': face_landmarks,
                            'key_landmarks': key_landmarks_pixel
                        })
            
            return face_info
            
        except Exception as e:
            print(f"Error in MediaPipe face detection: {e}")
            return []


    def detect_faces_full_frame(self, rgb_image):
        """Detect faces in the entire frame for tracking purposes using FaceMesh"""
        if self.face_mesh is None:
            print("MediaPipe FaceMesh not available")
            return []
            
        try:
            h, w, _ = rgb_image.shape
            results = self.face_mesh.process(rgb_image)
            
            face_info = []
            if results.multi_face_landmarks:
                key_indices = [1, 33, 61, 199, 263, 291]
                for face_landmarks in results.multi_face_landmarks:
                    # Get bbox from landmarks
                    x_coords = [lm.x * w for lm in face_landmarks.landmark]
                    y_coords = [lm.y * h for lm in face_landmarks.landmark]
                    x = max(0, min(x_coords))
                    y = max(0, min(y_coords))
                    width = max(x_coords) - x
                    height = max(y_coords) - y
                    
                    # Expand bbox a bit
                    margin = 0
                    x = max(0, x - margin)
                    y = max(0, y - margin)
                    width = min(w - x, width + 2 * margin)
                    height = min(h - y, height + 2 * margin)
                    
                    if width > 50 and height > 50:  # Minimum size for tracking
                        # Crop face ROI (RGB then to BGR)
                        face_roi_rgb = rgb_image[int(y):int(y + height), int(x):int(x + width)]
                        face_roi = cv2.cvtColor(face_roi_rgb, cv2.COLOR_RGB2BGR)
                        
                        # Key landmarks in pixel coordinates
                        key_landmarks_pixel = [(int(face_landmarks.landmark[idx].x * w), int(face_landmarks.landmark[idx].y * h)) for idx in key_indices]
                        
                        # Centroid from bbox
                        centroid = (int(x + width // 2), int(y + height // 2))
                        
                        face_info.append({
                            'face': face_roi,
                            'bbox': (int(x), int(y), int(width), int(height)),
                            'confidence': 0.9,
                            'full_landmarks': face_landmarks,
                            'key_landmarks': key_landmarks_pixel,
                            'centroid': centroid
                        })
            
            return face_info
            
        except Exception as e:
            print(f"Error in MediaPipe face detection: {e}")
            return []

    def process_webcam(self):
        """Process webcam frames with 5-frame ROI evaluation logic"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.most_recent_capture_arr = frame.copy()
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                h, w, _ = frame.shape
                if self.roi_x2 == 0 or self.roi_y2 == 0:
                    self.calculate_roi(w, h)
                
                if self.pause_recognition:
                    overlay = np.zeros((h, w, 3), dtype=np.uint8)
                    overlay[:, :] = (50, 50, 100)
                    img_rgb = cv2.addWeighted(img_rgb, 0.7, overlay, 0.3, 0)
                    
                    font = cv2.FONT_HERSHEY_SIMPLEX
                    text = "REGISTRATION MODE ACTIVE"
                    subtext = "Recognition paused - Use web interface to register"
                    
                    (text_width, text_height), _ = cv2.getTextSize(text, font, 1.5, 3)
                    (sub_width, sub_height), _ = cv2.getTextSize(subtext, font, 0.8, 2)
                    
                    text_x = (w - text_width) // 2
                    text_y = (h // 2) - 50
                    sub_x = (w - sub_width) // 2
                    sub_y = text_y + 60
                    
                    cv2.putText(img_rgb, text, (text_x, text_y), font, 1.5, (255, 255, 0), 3)
                    cv2.putText(img_rgb, subtext, (sub_x, sub_y), font, 0.8, (255, 255, 255), 2)
                else:
                    self.draw_roi(img_rgb)
                    
                    try:
                        all_faces = self.detect_faces_full_frame(img_rgb)
                        
                        # Generate face encodings for tracking
                        face_encodings = []
                        for face_data in all_faces:
                            encoding = self.get_face_encoding(face_data['face'])
                            face_encodings.append(encoding)
                        
                        # Update tracking with face encodings
                        tracking_results = self.person_tracker.update(all_faces, face_encodings)
                        
                        recognized_names = []
                        current_time = datetime.datetime.now()
                        
                        for person_id, detection in tracking_results.items():
                            person_info = self.person_tracker.get_person_info(person_id)
                            if not person_info:
                                continue
                            
                            bbox = detection['bbox']
                            prev_name = person_info.get('name', 'Unknown Person')
                            prev_similarity = person_info.get('similarity', 0.0)
                            confidence = detection['confidence']
                            
                            pose_data = self.get_head_pose_for_person(detection['face'], person_id)
                            
                            if self.is_face_in_roi(bbox):
                                # Check if this is first time in ROI for this person
                                if person_id not in self.person_roi_entry:
                                    self.person_roi_entry[person_id] = self.person_tracker.frame_count
                                    self.person_evaluation_complete[person_id] = False
                                    print(f"Person {person_id} entered ROI - starting 5-frame evaluation")
                                
                                # Only perform recognition if evaluation is not complete
                                if not self.person_evaluation_complete.get(person_id, True):
                                    print(f"Person {person_id} is inside ROI - performing recognition (evaluation active)")
                                    name, similarity = self.recognize_face(detection['face'], person_id)
                                    print(f"Person {person_id} recognized as {name} with similarity {similarity:.2f}")
                                    
                                    # Update recognition history during evaluation
                                    if person_id not in self.person_recognition_history:
                                        self.person_recognition_history[person_id] = []
                                    self.person_recognition_history[person_id].append((name, similarity))
                                    
                                    # Keep only evaluation period data
                                    if len(self.person_recognition_history[person_id]) > self.roi_evaluation_frames:
                                        self.person_recognition_history[person_id] = self.person_recognition_history[person_id][-self.roi_evaluation_frames:]
                                    
                                    # Check if 5-frame evaluation period is complete
                                    frames_since_roi_entry = self.person_tracker.frame_count - self.person_roi_entry[person_id]
                                    if frames_since_roi_entry >= self.roi_evaluation_frames:
                                        # Finalize the name assignment
                                        final_name, final_similarity = self.finalize_name_assignment(person_id)
                                        name, similarity = final_name, final_similarity
                                        self.person_evaluation_complete[person_id] = True
                                        print(f"✅ Evaluation complete for ID {person_id}: Final assignment = {final_name}")
                                    else:
                                        # During evaluation, show as Unknown
                                        name, similarity = 'Unknown Person', similarity
                                else:
                                    # Evaluation complete - maintain the assigned name
                                    name = person_info.get('name', 'Unknown Person')
                                    similarity = person_info.get('similarity', 0.0)
                                    print(f"Person {person_id} evaluation complete - maintaining assigned name: {name}")
                            else:
                                name, similarity = prev_name, prev_similarity
                                print(f"Person {person_id} outside ROI - maintaining recognition: {name} ({similarity:.2f})")
                            
                            # CONSERVATIVE Auto-login logic - only for finalized identities
                            # MODIFIED Auto-login logic for immediate re-login of known users
                            if (name != 'Unknown Person' and
                                similarity >= self.auto_login_threshold and
                                name not in self.logged_in_users and
                                name not in self.auto_login_cooldown and
                                (self.person_evaluation_complete.get(person_id, False) or name in self.logged_out_users)):  # <-- KEY CHANGE

                                if self.is_face_in_roi(bbox):
                                    # Determine the reason for login for clearer console output
                                    login_reason = "finalized identity"
                                    if name in self.logged_out_users:
                                        login_reason = "immediate re-login for known user"

                                    print(f"[AUTO_LOGIN] Auto-logging in {name} ({login_reason})")
                                    self.auto_login_cooldown[name] = current_time
                                    self.last_auto_login_time[name] = current_time
                                    self.auto_login_user(name, similarity)
                                else:
                                    # This condition is met if a known user is tracked but not in the ROI yet
                                    if self.person_evaluation_complete.get(person_id, False) or name in self.logged_out_users:
                                        print(f"[AUTO_LOGIN_PENDING] Known user {name} tracked outside ROI. Login will occur upon re-entry.")


                            
                            # Clear cooldown after sufficient time
                            if (name in self.auto_login_cooldown and 
                                (current_time - self.auto_login_cooldown[name]).seconds > 10):
                                print(f"[AUTO_LOGIN] Clearing cooldown for {name}")
                                del self.auto_login_cooldown[name]
                            
                            self.last_recognized_name = name
                            
                            # Update person's recognition info
                            self.person_tracker.update_person_recognition(person_id, name, similarity, confidence)
                            recognized_names.append(name)
                            
                            self.draw_person_tracking_info_with_head_pose(
                                img_rgb, person_id, person_info, bbox, name, similarity, pose_data, detection
                            )
                        
                        # Process logout logic - IMMEDIATE logout has priority
                        self.immediate_logout_outside_roi(tracking_results)
                        
                        # Then check timeout-based logout
                        self.check_auto_logout_with_head_pose(recognized_names)
                        self.update_status_with_tracking(recognized_names)
                        
                    except Exception as e:
                        print(f"Error in face processing: {e}")
                
                self.update_attendance_display()
                
                self.most_recent_capture_pil = Image.fromarray(img_rgb)
                imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
                self.webcam_label.imgtk = imgtk
                self.webcam_label.configure(image=imgtk)
        
        if hasattr(self, 'webcam_label'):
            self.webcam_label.after(20, self.process_webcam)

    def draw_person_tracking_info_with_head_pose(self, img_rgb, person_id, person_info, bbox, name, similarity, pose_data, detection):
        """Enhanced tracking info display with head pose and landmarks"""
        # Draw landmarks first
        full_landmarks = detection.get('full_landmarks')
        if full_landmarks and self.mp_drawing:
            self.mp_drawing.draw_landmarks(
                image=img_rgb,
                landmark_list=full_landmarks,
                connections=mp.solutions.face_mesh.FACEMESH_TESSELATION,
                landmark_drawing_spec=mp.solutions.drawing_utils.DrawingSpec(
                    color=(0, 255, 0), thickness=1, circle_radius=1
                ),
                connection_drawing_spec=mp.solutions.drawing_styles.get_default_face_mesh_tesselation_style()
            )
        
        x, y, width, height = bbox
        in_roi = self.is_face_in_roi(bbox)
        
        # Choose colors based on ROI status and recognition
        if not in_roi:
            if name == 'Unknown Person':
                box_color = (200, 100, 100)  # Light red
                track_color = (150, 75, 75)
            elif name in self.logged_in_users:
                box_color = (100, 200, 100)  # Light green
                track_color = (75, 150, 75)
            else:
                box_color = (200, 200, 100)  # Light yellow
                track_color = (150, 150, 75)
            line_style = 2
        else:
            if name == 'Unknown Person':
                box_color = (255, 0, 0)  # Red
                track_color = (150, 0, 0)
            elif name in self.logged_in_users:
                box_color = (0, 255, 0)  # Green
                track_color = (0, 150, 0)
            else:
                box_color = (255, 255, 0)  # Yellow
                track_color = (150, 150, 0)
            line_style = 3
        
        # Draw main face box
        cv2.rectangle(img_rgb, (x, y), (x + width, y + height), box_color, line_style)
        
        # HEAD POSE DISPLAY
        if pose_data and pose_data['left_right_turn'] is not None:
            left_right = pose_data['left_right_turn']
            up_down = pose_data['up_down_nod']
            side_tilt = pose_data['side_tilt']
            status = pose_data['status']
            
            cv2.putText(img_rgb, f"Head: {status}", (x, y-60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
            cv2.putText(img_rgb, f"L/R:{left_right:.0f} U/D:{up_down:.0f} Tilt:{side_tilt:.0f}", (x, y-45), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 255), 1)
            
            if pose_data['should_maintain_login']:
                cv2.putText(img_rgb, "IDENTITY STABLE", (x, y+height+35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)
            else:
                cv2.putText(img_rgb, "HEAD MOVING - MONITORING", (x, y+height+35), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 0), 1)
        
        # Regular tracking info with evaluation status
        roi_status = " [IN ROI]" if in_roi else " [TRACKED]"
        login_status = " [LOGGED IN]" if name in self.logged_in_users else ""
        logout_status = " [LOGGED OUT]" if name in self.logged_out_users else ""
        frames_active = person_info.get('frames_active', 0)
        detection_confidence = person_info.get('confidence', 0.0)
        
        # Show evaluation status
        evaluation_status = ""
        if person_id in self.person_roi_entry and not self.person_evaluation_complete.get(person_id, True):
            frames_evaluated = self.person_tracker.frame_count - self.person_roi_entry[person_id]
            evaluation_status = f" [EVAL: {frames_evaluated}/{self.roi_evaluation_frames}]"
        elif self.person_evaluation_complete.get(person_id, False):
            evaluation_status = " [FINALIZED]"
        
        # Show if this is a persistent ID
        persistent_indicator = ""
        if name in self.person_tracker.person_id_history and self.person_tracker.person_id_history[name] == person_id:
            persistent_indicator = " [PERSISTENT ID]"
        
        display_text = f"ID:{person_id} {name}{roi_status}{login_status}{logout_status}{evaluation_status}{persistent_indicator}"
        info_text = f"Det Conf: {detection_confidence:.2f} | Rec Sim: {similarity:.1%} | Frames: {frames_active}"
        
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.7
        font_thickness = 2
        
        # Main label background and text
        (text_width, text_height), _ = cv2.getTextSize(display_text, font, font_scale, font_thickness)
        text_x = max(0, x)
        text_y = max(text_height + 10, y - 10)
        
        cv2.rectangle(img_rgb, 
                    (text_x - 5, text_y - text_height - 5), 
                    (text_x + text_width + 5, text_y + 5), 
                    box_color, -1)
        cv2.putText(img_rgb, display_text, (text_x, text_y), 
                    font, font_scale, (0, 0, 0), font_thickness)
        
        # Additional tracking info
        (info_width, info_height), _ = cv2.getTextSize(info_text, font, 0.5, 1)
        info_y = y + height + 20
        
        cv2.rectangle(img_rgb, 
                    (x - 2, info_y - info_height - 2), 
                    (x + info_width + 2, info_y + 2), 
                    track_color, -1)
        cv2.putText(img_rgb, info_text, (x, info_y), 
                    font, 0.5, (255, 255, 255), 1)

    def is_face_in_roi(self, bbox):
        """Check if a face bounding box overlaps with ROI"""
        if not self.roi_enabled:
            return True
        
        x, y, width, height = bbox
        face_center_x = x + width // 2
        face_center_y = y + height // 2
        
        # Check if face center is within ROI
        return (self.roi_x1 <= face_center_x <= self.roi_x2 and 
                self.roi_y1 <= face_center_y <= self.roi_y2)

    def load_face_database(self, folder_path):
        """Load face database from folder"""
        face_database = {}
        
        if not os.path.exists(folder_path):
            os.makedirs(folder_path, exist_ok=True)
            return face_database
        
        for person_folder in os.listdir(folder_path):
            person_path = os.path.join(folder_path, person_folder)
            if os.path.isdir(person_path):
                person_encodings = []
                valid_images = 0
                
                for img_file in os.listdir(person_path):
                    if img_file.lower().endswith(('.jpg', '.jpeg', '.png', '.bmp')):
                        img_path = os.path.join(person_path, img_file)
                        image = cv2.imread(img_path)
                        if image is not None:
                            encoding = self.get_face_encoding(image)
                            if encoding is not None:
                                person_encodings.append(encoding)
                                valid_images += 1
                
                if person_encodings and valid_images >= 1:
                    face_database[person_folder] = np.mean(person_encodings, axis=0)
                    print(f"Loaded {valid_images} images for {person_folder}")
        
        return face_database

    def logout_user_preserve_id(self, name, person_id):
        """Logout a user but preserve their person ID for future re-identification"""
        if name in self.logged_in_users:
            duration = self.attendance_logger.log_logout(name)
            if duration:
                total_today = self.attendance_logger.get_total_time_today(name)
                messagebox.showinfo("Logout Successful", 
                                  f"{name} logged out\n"
                                  f"Session duration: {duration:.1f} minutes\n"
                                  f"Total time today: {total_today:.1f} minutes\n"
                                  f"Person ID {person_id} preserved for re-login")
                
                # Move from logged_in to logged_out but preserve tracking
                del self.logged_in_users[name]
                self.logged_out_users[name] = person_id  # Track logged out users with their IDs
                
                if name in self.last_auto_login_time:
                    del self.last_auto_login_time[name]
                if name in self.auto_login_cooldown:
                    del self.auto_login_cooldown[name]
                
                self.update_attendance_display()

    def logout_user(self, name):
        """Standard logout method - calls the new preserve ID method"""
        # Find the person ID for this name
        person_id = None
        for pid, person_info in self.person_tracker.persons.items():
            if person_info.get('name') == name:
                person_id = pid
                break
        
        if person_id is None:
            # Fallback - check person_id_history
            person_id = self.person_tracker.person_id_history.get(name, 'Unknown')
        
        self.logout_user_preserve_id(name, person_id)

    def login_user(self, name):
        """Login a user"""
        if name != 'Unknown Person' and name not in self.logged_in_users:
            login_time = self.attendance_logger.log_login(name)
            self.logged_in_users[name] = login_time
            
            # Remove from logged out if present
            if name in self.logged_out_users:
                del self.logged_out_users[name]
            
            total_today = self.attendance_logger.get_total_time_today(name)
            messagebox.showinfo("Login Successful", 
                              f"{name} logged in at {login_time.strftime('%H:%M:%S')}\n"
                              f"Total time today: {total_today:.1f} minutes")
            self.update_attendance_display()

    def auto_login_user(self, name, similarity):
        """Auto login user and show custom notification with photo"""
        if name != 'Unknown Person' and name not in self.logged_in_users:
            login_time = self.attendance_logger.log_login(name)
            self.logged_in_users[name] = login_time
            
            # Remove from logged out if present
            if name in self.logged_out_users:
                person_id = self.logged_out_users[name]
                del self.logged_out_users[name]
                print(f"Auto-login: {name} restored with preserved ID {person_id}")
            
            self.show_auto_login_with_photo(name, similarity, login_time)
            
            self.update_attendance_display()

    def show_auto_login_with_photo(self, name, similarity, login_time):
        """Creates a custom Toplevel window on the right side of the main window"""
        try:
            person_folder = os.path.join('./face_database', name)
            photo_path = None
            if os.path.exists(person_folder):
                for filename in os.listdir(person_folder):
                    if 'Front_View' in filename and filename.lower().endswith(('.jpg', '.jpeg', '.png')):
                        photo_path = os.path.join(person_folder, filename)
                        break

            win = Toplevel(self.main_window)
            win.title("Auto Login Successful")
            win.config(bg='white')
            win.resizable(False, False)
            win.transient(self.main_window)

            img_tk = None
            if photo_path:
                img = Image.open(photo_path).convert("RGBA")
                size = (100, 100)
                mask = Image.new('L', size, 0)
                draw = ImageDraw.Draw(mask)
                draw.ellipse((0, 0) + size, fill=255)
                img = img.resize(size, Image.Resampling.LANCZOS)
                img.putalpha(mask)
                img_tk = ImageTk.PhotoImage(img)

            if img_tk:
                img_label = Label(win, image=img_tk, bg='white')
                img_label.image = img_tk
                img_label.pack(side="left", padx=15, pady=15)

            total_today = self.attendance_logger.get_total_time_today(name)
            message = (
                f"🟢 {name} auto logged in!\n\n"
                f"Similarity: {similarity:.1%}\n"
                f"Time: {login_time.strftime('%H:%M:%S')}\n"
                f"Total time today: {total_today:.1f} minutes"
            )
            
            text_label = Label(win, text=message, justify="left", font=('Arial', 10), bg='white')
            text_label.pack(side="left", padx=(0, 15), pady=15)

            win.update_idletasks()
            main_x = self.main_window.winfo_x()
            main_y = self.main_window.winfo_y()
            main_width = self.main_window.winfo_width()
            main_height = self.main_window.winfo_height()
            
            win_width = win.winfo_width()
            win_height = win.winfo_height()

            x = main_x + main_width - win_width - 10
            y = main_y + 80
            win.geometry(f"+{x}+{y}")
            
            win.after(4000, win.destroy)

        except Exception as e:
            print(f"Failed to create custom login notification: {e}")
            total_today = self.attendance_logger.get_total_time_today(name)
            messagebox.showinfo("Auto Login", 
                              f"🟢 {name} auto logged in!\n"
                              f"Similarity: {similarity:.1%}\n"
                              f"Time: {login_time.strftime('%H:%M:%S')}\n"
                              f"Total time today: {total_today:.1f} minutes")

    def manual_login(self):
        """Manual login"""
        if not hasattr(self, 'last_recognized_name') or self.last_recognized_name == 'Unknown Person':
            messagebox.showwarning("No User", "No recognized user found.")
            return
        self.login_user(self.last_recognized_name)

    def manual_logout(self):
        """Manual logout"""
        if not self.logged_in_users:
            messagebox.showwarning("No Users", "No users are currently logged in.")
            return
        
        if len(self.logged_in_users) == 1:
            name = list(self.logged_in_users.keys())[0]
            self.logout_user(name)
        else:
            names = list(self.logged_in_users.keys())
            choice = simpledialog.askstring("Logout", 
                                           f"Multiple users: {', '.join(names)}\nEnter name:")
            if choice and choice in self.logged_in_users:
                self.logout_user(choice)

    def update_attendance_display(self):
        """Update attendance display"""
        status_parts = []
        
        if self.logged_in_users:
            status_parts.append("Logged in: " + ", ".join(self.logged_in_users.keys()))
        
        if self.logged_out_users:
            status_parts.append("Recently logged out: " + ", ".join(self.logged_out_users.keys()))
        
        if not status_parts:
            logged_in_text = "No users logged in"
        else:
            logged_in_text = " | ".join(status_parts)
            
        if self.pause_recognition:
            logged_in_text += " | REGISTRATION MODE - Recognition paused"
        
        self.attendance_label.config(text=logged_in_text)

    def open_registration_page(self):
        """Open the web registration page"""
        try:
            webbrowser.open('http://localhost:8000/register')
            messagebox.showinfo("Registration", 
                              "Registration page opened!\n\n"
                              "🎯 Registration Features:\n"
                              "• Recognition camera paused during registration\n"
                              "• 4-angle photo capture with validation\n"
                              "• Only face regions saved (not full frames)\n"
                              "• Visual progress indicators\n"
                              "• Automatic resume after completion")
        except Exception as e:
            messagebox.showerror("Error", f"Could not open registration page: {e}")

    def toggle_roi(self):
        """Toggle ROI on/off"""
        self.roi_enabled = not self.roi_enabled
        status = "enabled" if self.roi_enabled else "disabled"
        print(f"ROI {status}")

    def update_status_with_tracking(self, recognized_names):
        """Update status with tracking information"""
        roi_status = "ROI Enabled" if self.roi_enabled else "ROI Disabled"
        model_status = "DeepFace FaceNet (Auto-Login: 74%)"
        
        total_tracked = len(self.person_tracker.persons)
        active_tracked = len([p for p in self.person_tracker.persons.values() 
                            if self.person_tracker.frame_count - p['last_seen'] <= 5])
        
        inactive_count = len(self.person_tracker.inactive_persons)
        tracking_status = f"Tracked: {active_tracked}/{total_tracked} | Inactive: {inactive_count}"
        
        if recognized_names:
            known_names = [name for name in recognized_names if name != 'Unknown Person']
            unknown_count = recognized_names.count('Unknown Person')
            
            status_parts = []
            if known_names:
                unique_known = list(set(known_names))
                status_parts.append(f"Recognized: {', '.join(unique_known)}")
            if unknown_count > 0:
                status_parts.append(f"Unknown: {unknown_count}")
            
            status_text = " | ".join(status_parts) + f" | {tracking_status} | DB: {len(self.face_database)} | {model_status} | {roi_status}"
        else:
            status_text = f"{model_status} - {len(self.face_database)} people | {tracking_status} | {roi_status}"
        
        self.status_label.config(text=status_text)

    def setup_gui(self):
        """Setup GUI components"""
        screen_width = self.main_window.winfo_screenwidth()
        screen_height = self.main_window.winfo_screenheight()
        
        webcam_width = screen_width - 100
        webcam_height = screen_height - 200
        
        self.webcam_label = Label(self.main_window, bg='black')
        self.webcam_label.place(x=50, y=80, width=webcam_width, height=webcam_height)
        
        self.status_label = Label(
            self.main_window, 
            text="Integrated Face Recognition & Registration System", 
            font=('Arial', 16, 'bold'),
            bg='lightblue',
            fg='black'
        )
        self.status_label.place(x=0, y=0, width=screen_width, height=40)
        
        self.attendance_label = Label(
            self.main_window, 
            text="No users logged in | Registration server: http://localhost:8000", 
            font=('Arial', 14),
            bg='lightyellow',
            fg='black'
        )
        self.attendance_label.place(x=0, y=40, width=screen_width, height=35)
        
        button_y = screen_height - 120
        button_width = 150
        button_height = 40
        
        self.login_button = Button(
            self.main_window, 
            text="🟢 LOGIN", 
            command=self.manual_login,
            bg='lightgreen',
            fg='black',
            font=('Arial', 12, 'bold'),
            relief='raised',
            bd=3
        )
        self.login_button.place(x=50, y=button_y, width=button_width, height=button_height)
        
        self.logout_button = Button(
            self.main_window, 
            text="🔴 LOGOUT", 
            command=self.manual_logout,
            bg='lightcoral',
            fg='black',
            font=('Arial', 12, 'bold'),
            relief='raised',
            bd=3
        )
        self.logout_button.place(x=220, y=button_y, width=button_width, height=button_height)
        
        self.register_button = Button(
            self.main_window, 
            text="👤 REGISTER USER", 
            command=self.open_registration_page,
            bg='lightblue',
            fg='black',
            font=('Arial', 12, 'bold'),
            relief='raised',
            bd=3
        )
        self.register_button.place(x=390, y=button_y, width=button_width+20, height=button_height)
        
        self.roi_button = Button(
            self.main_window, 
            text="🎯 TOGGLE ROI", 
            command=self.toggle_roi,
            bg='lightpink',
            fg='black',
            font=('Arial', 12, 'bold'),
            relief='raised',
            bd=3
        )
        self.roi_button.place(x=570, y=button_y, width=button_width, height=button_height)
        
        self.exit_button = Button(
            self.main_window, 
            text="❌ EXIT", 
            command=self.on_closing,
            bg='orange',
            fg='black',
            font=('Arial', 12, 'bold'),
            relief='raised',
            bd=3
        )
        self.exit_button.place(x=screen_width-200, y=button_y, width=button_width, height=button_height)
        
        self.add_webcam()

    def add_webcam(self):
        """Initialize webcam"""
        try:
            for camera_index in [0, 1, 2]:
                self.cap = cv2.VideoCapture(camera_index)
                if self.cap.isOpened():
                    self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
                    self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
                    self.cap.set(cv2.CAP_PROP_FPS, 30)
                    print(f"Camera {camera_index} opened successfully")
                    break
                else:
                    self.cap.release()
                    self.cap = None
            
            if self.cap is None:
                print("No camera found!")
                return
            
            self.process_webcam()
            
        except Exception as e:
            print(f"Error initializing webcam: {e}")

    def start(self):
        """Start the application"""
        try:
            self.main_window.protocol("WM_DELETE_WINDOW", self.on_closing)
            self.main_window.bind('<Escape>', lambda e: self.on_closing())
            self.main_window.bind('<Key-r>', lambda e: self.toggle_roi())
            self.main_window.bind('<Key-R>', lambda e: self.toggle_roi())
            self.main_window.focus_set()
            self.main_window.mainloop()
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            self.cleanup()

    def on_closing(self):
        """Handle window closing"""
        for name in list(self.logged_in_users.keys()):
            self.logout_user(name)
        self.cleanup()
        self.main_window.destroy()

    def cleanup(self):
        """Cleanup resources with enhanced error handling"""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        
        # Safe cleanup for MediaPipe FaceMesh
        if hasattr(self, 'face_mesh') and self.face_mesh is not None:
            try:
                self.face_mesh.close()
            except (AttributeError, Exception) as e:
                print(f"Warning: Could not properly close face_mesh: {e}")
        
        # Also cleanup the head pose estimator's face mesh
        if hasattr(self, 'head_pose_estimator') and self.head_pose_estimator:
            if hasattr(self.head_pose_estimator, 'face_mesh') and self.head_pose_estimator.face_mesh is not None:
                try:
                    self.head_pose_estimator.face_mesh.close()
                except (AttributeError, Exception) as e:
                    print(f"Warning: Could not properly close head pose face_mesh: {e}")
        
        cv2.destroyAllWindows()

if __name__ == "__main__":
    try:
        app = FaceRecognitionApp()
        app.start()
    except Exception as e:
        print(f"❌ Failed to start application: {e}")
        import traceback
        traceback.print_exc()
