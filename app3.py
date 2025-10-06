import warnings
import os

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

# Attendance Logger Class
class AttendanceLogger:
    def __init__(self, log_file='attendance_log.xlsx'):
        self.log_file = log_file
        # Load or create DataFrame for attendance log
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

class FaceRecognitionApp:
    def __init__(self):
        # Initialize camera variables first
        self.cap = None
        self.most_recent_capture_arr = None
        self.most_recent_capture_pil = None
        self.auto_login_threshold = 0.74  # 85% similarity threshold for auto login
        self.auto_login_cooldown = {}  # Prevent multiple rapid logins
        self.last_auto_login_time = {}  # Track last auto login time per person
        # Initialize attendance logger
        self.attendance_logger = AttendanceLogger()
        
        # Login/logout tracking
        self.logged_in_users = {}  # {name: login_time}
        
        # Initialize MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.mp_drawing = mp.solutions.drawing_utils
        self.face_detection = self.mp_face_detection.FaceDetection(
            model_selection=0, min_detection_confidence=0.6)
        
        # Load FaceNet model using DeepFace
        self.facenet_model = self.load_pretrained_facenet_model()
        
        # Load face database from uploaded folder
        self.face_database = self.load_face_database('./face_database')
        print(f"Loaded {len(self.face_database)} people in database")
        
        # Recognition parameters - ADJUSTED FOR BETTER ACCURACY
        self.recognition_threshold = 0.65  # Lowered for more flexibility
        self.face_recognition_cache = {}
        self.recognition_history = {}
        
        # ROI parameters - will be calculated based on frame size
        self.roi_x1 = 0
        self.roi_y1 = 0
        self.roi_x2 = 0
        self.roi_y2 = 0
        self.roi_enabled = True  # Enable/disable ROI functionality
        
        # GUI setup - Full screen
        self.main_window = tk.Tk()
        self.main_window.title("DeepFace FaceNet Face Recognition System - Attendance Tracker with ROI")
        self.main_window.state('zoomed')  # Full screen on Windows
        
        self.setup_gui()
    
    def load_pretrained_facenet_model(self):
        """Load FaceNet using DeepFace (no local keras file needed)"""
        try:
            print("Initializing DeepFace FaceNet model...")
            # Test DeepFace FaceNet by creating a dummy embedding
            test_img = np.zeros((160, 160, 3), dtype=np.uint8)
            DeepFace.represent(test_img, model_name='Facenet', enforce_detection=False)
            print("✓ Successfully initialized DeepFace FaceNet model")
            print("✓ Model: FaceNet (128-dimensional embeddings)")
            print("✓ No local model file required")
            return True
        except Exception as e:
            print(f"Error initializing DeepFace FaceNet model: {e}")
            print("\nTroubleshooting:")
            print("1. Install DeepFace: pip install deepface")
            print("2. Ensure internet connection for model download")
            print("3. DeepFace will download models automatically on first use")
            return None
    
    def preprocess_face_for_facenet(self, face_image):
        """
        Improved face preprocessing for FaceNet
        """
        try:
            # Ensure the image has proper dimensions
            if face_image.shape[0] < 50 or face_image.shape[1] < 50:
                return None
            
            # Convert BGR to RGB for DeepFace
            rgb_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            
            # Resize to standard size for better consistency
            resized = cv2.resize(rgb_image, (160, 160), interpolation=cv2.INTER_CUBIC)
            
            # Enhance contrast and brightness
            lab = cv2.cvtColor(resized, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            l = clahe.apply(l)
            enhanced = cv2.merge([l, a, b])
            enhanced_rgb = cv2.cvtColor(enhanced, cv2.COLOR_LAB2RGB)
            
            # Normalize the image
            normalized = enhanced_rgb.astype(np.float32) / 255.0
            
            return normalized
            
        except Exception as e:
            print(f"Error in face preprocessing: {e}")
            return None
    
    def get_face_encoding(self, face_image):
        """Generate FaceNet face encoding from image using DeepFace with improved preprocessing"""
        if self.facenet_model is None:
            print("Warning: No DeepFace FaceNet model available.")
            return None
            
        try:
            # Preprocess the face image
            preprocessed = self.preprocess_face_for_facenet(face_image)
            if preprocessed is None:
                return None
            
            # Generate embedding using DeepFace FaceNet
            embedding_objs = DeepFace.represent(
                img_path=preprocessed,
                model_name='Facenet',
                enforce_detection=False,  # Don't enforce detection for preprocessed faces
                detector_backend='skip'   # Skip detection since we already have face ROI
            )
            
            if embedding_objs and len(embedding_objs) > 0:
                embedding = np.array(embedding_objs[0]['embedding'])
                # Normalize the embedding for better comparison
                normalized_embedding = embedding / np.linalg.norm(embedding)
                return normalized_embedding
            else:
                return None
                
        except Exception as e:
            print(f"Error generating DeepFace FaceNet encoding: {e}")
            return None
    
    def get_facenet_encoding_from_file(self, image_path):
        """Generate DeepFace FaceNet encoding from image file with improved preprocessing"""
        try:
            # Read the image
            image = cv2.imread(image_path)
            if image is None:
                print(f"Could not read image: {image_path}")
                return None
            
            # Use the same preprocessing as real-time recognition
            return self.get_face_encoding(image)
                
        except Exception as e:
            print(f"Error loading image {image_path}: {e}")
            return None
    
    def calculate_roi(self, frame_width, frame_height):
        """Calculate Region of Interest coordinates for center position"""
        roi_width = int(frame_width * 0.35)   # Slightly larger ROI
        roi_height = int(frame_height * 0.6)  
        
        self.roi_x1 = (frame_width - roi_width) // 2
        self.roi_y1 = (frame_height - roi_height) // 2  
        self.roi_x2 = self.roi_x1 + roi_width
        self.roi_y2 = self.roi_y1 + roi_height

    def draw_roi(self, image):
        """Draw ROI rectangle on the image"""
        if self.roi_enabled:
            # Draw main ROI rectangle in red with thicker border
            cv2.rectangle(image, (self.roi_x1, self.roi_y1), (self.roi_x2, self.roi_y2), (255, 0, 0), 4)
            
            # Add ROI label - positioned above the ROI box
            label_text = "FACE DETECTION ZONE"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.9
            font_thickness = 2
            
            # Get text size to center it above ROI
            (text_width, text_height), _ = cv2.getTextSize(label_text, font, font_scale, font_thickness)
            text_x = self.roi_x1 + (self.roi_x2 - self.roi_x1 - text_width) // 2
            text_y = self.roi_y1 - 15
            
            # Draw text background for better visibility
            cv2.rectangle(image, (text_x - 5, text_y - text_height - 5), 
                        (text_x + text_width + 5, text_y + 5), (0, 0, 0), -1)
            cv2.putText(image, label_text, (text_x, text_y), font, font_scale, (255, 0, 0), font_thickness)
            
            # Add corner markers for better visibility
            corner_size = 25
            corner_thickness = 4
            
            # Top-left corner
            cv2.line(image, (self.roi_x1, self.roi_y1), (self.roi_x1 + corner_size, self.roi_y1), (255, 0, 0), corner_thickness)
            cv2.line(image, (self.roi_x1, self.roi_y1), (self.roi_x1, self.roi_y1 + corner_size), (255, 0, 0), corner_thickness)
            
            # Top-right corner
            cv2.line(image, (self.roi_x2, self.roi_y1), (self.roi_x2 - corner_size, self.roi_y1), (255, 0, 0), corner_thickness)
            cv2.line(image, (self.roi_x2, self.roi_y1), (self.roi_x2, self.roi_y1 + corner_size), (255, 0, 0), corner_thickness)
            
            # Bottom-left corner
            cv2.line(image, (self.roi_x1, self.roi_y2), (self.roi_x1 + corner_size, self.roi_y2), (255, 0, 0), corner_thickness)
            cv2.line(image, (self.roi_x1, self.roi_y2), (self.roi_x1, self.roi_y2 - corner_size), (255, 0, 0), corner_thickness)
            
            # Bottom-right corner
            cv2.line(image, (self.roi_x2, self.roi_y2), (self.roi_x2 - corner_size, self.roi_y2), (255, 0, 0), corner_thickness)
            cv2.line(image, (self.roi_x2, self.roi_y2), (self.roi_x2, self.roi_y2 - corner_size), (255, 0, 0), corner_thickness)
            
            # Add center crosshair for precise positioning
            center_x = (self.roi_x1 + self.roi_x2) // 2
            center_y = (self.roi_y1 + self.roi_y2) // 2
            crosshair_size = 15
            
            # Horizontal line
            cv2.line(image, (center_x - crosshair_size, center_y), 
                    (center_x + crosshair_size, center_y), (255, 100, 100), 2)
            # Vertical line
            cv2.line(image, (center_x, center_y - crosshair_size), 
                    (center_x, center_y + crosshair_size), (255, 100, 100), 2)

    def detect_faces_in_roi(self, image):
        """Detect faces only within the ROI with better cropping"""
        if not self.roi_enabled:
            return self.detect_faces_mediapipe(image)
        
        h, w, _ = image.shape
        
        # Update ROI if frame size changed
        if self.roi_x2 == 0 or self.roi_y2 == 0:
            self.calculate_roi(w, h)
        
        # Extract ROI from the image
        roi_image = image[self.roi_y1:self.roi_y2, self.roi_x1:self.roi_x2]
        
        # Detect faces in ROI
        rgb_roi = cv2.cvtColor(roi_image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_roi)
        
        face_info = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                roi_h, roi_w, _ = roi_image.shape
                
                # Convert relative coordinates to ROI coordinates
                x_roi = max(0, int(bboxC.xmin * roi_w))
                y_roi = max(0, int(bboxC.ymin * roi_h))
                width = min(roi_w - x_roi, int(bboxC.width * roi_w))
                height = min(roi_h - y_roi, int(bboxC.height * roi_h))
                
                if width > 80 and height > 80:  # Increased minimum size
                    # Convert ROI coordinates back to full image coordinates
                    x_full = self.roi_x1 + x_roi
                    y_full = self.roi_y1 + y_roi
                    
                    # Better padding strategy for face extraction
                    padding_x = int(width * 0.15)  # 15% padding
                    padding_y = int(height * 0.15)
                    
                    x1 = max(0, x_full - padding_x)
                    y1 = max(0, y_full - padding_y)
                    x2 = min(w, x_full + width + padding_x)
                    y2 = min(h, y_full + height + padding_y)
                    
                    face_roi = image[y1:y2, x1:x2]
                    if face_roi.size > 0:
                        face_info.append({
                            'face': face_roi,
                            'bbox': (x_full, y_full, width, height),
                            'confidence': detection.score[0]
                        })
        
        return face_info
    
    def toggle_roi(self):
        """Toggle ROI on/off"""
        self.roi_enabled = not self.roi_enabled
        status = "enabled" if self.roi_enabled else "disabled"
        print(f"ROI {status}")

    def load_face_database(self, folder_path):
        """Load and encode faces from uploaded folder using improved DeepFace FaceNet"""
        face_database = {}
        
        if not os.path.exists(folder_path):
            print(f"Dataset folder {folder_path} not found. Creating empty database.")
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
                        
                        encoding = self.get_facenet_encoding_from_file(img_path)
                        if encoding is not None:
                            person_encodings.append(encoding)
                            valid_images += 1
                
                if person_encodings and valid_images >= 1:
                    # Use average of all encodings for more robust representation
                    face_database[person_folder] = np.mean(person_encodings, axis=0)
                    print(f"Loaded {valid_images} images for {person_folder}")
                else:
                    print(f"Warning: No valid face encodings found for {person_folder}")
        
        return face_database
    
    def detect_faces_mediapipe(self, image):
        """Detect faces using MediaPipe with better cropping"""
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(rgb_image)
        
        face_info = []
        if results.detections:
            for detection in results.detections:
                bboxC = detection.location_data.relative_bounding_box
                h, w, _ = image.shape
                
                x = max(0, int(bboxC.xmin * w))
                y = max(0, int(bboxC.ymin * h))
                width = min(w - x, int(bboxC.width * w))
                height = min(h - y, int(bboxC.height * h))
                
                if width > 80 and height > 80:  # Increased minimum size
                    # Better padding strategy
                    padding_x = int(width * 0.15)  # 15% padding
                    padding_y = int(height * 0.15)
                    
                    x1 = max(0, x - padding_x)
                    y1 = max(0, y - padding_y)
                    x2 = min(w, x + width + padding_x)
                    y2 = min(h, y + height + padding_y)
                    
                    face_roi = image[y1:y2, x1:x2]
                    if face_roi.size > 0:
                        face_info.append({
                            'face': face_roi,
                            'bbox': (x, y, width, height),
                            'confidence': detection.score[0]
                        })
        
        return face_info

    def recognize_face_with_facenet(self, face_roi, face_id):
        """Improved face recognition using DeepFace FaceNet embeddings - returns name and similarity"""
        if self.facenet_model is None:
            return 'Unknown Person', 0.0
        
        if len(self.face_database) == 0:
            return 'Unknown Person', 0.0
        
        try:
            current_encoding = self.get_face_encoding(face_roi)
            if current_encoding is None:
                return 'Unknown Person', 0.0
            
            best_match = None
            best_similarity = 0.0
            similarity_scores = {}
            
            for name, stored_encoding in self.face_database.items():
                try:
                    # Use cosine similarity for comparison
                    similarity = cosine_similarity(
                        current_encoding.reshape(1, -1),
                        stored_encoding.reshape(1, -1)
                    )[0][0]
                    
                    similarity_scores[name] = similarity
                    
                    if similarity > best_similarity:
                        best_similarity = similarity
                        best_match = name
                        
                except Exception as e:
                    print(f"Error comparing with {name}: {e}")
                    continue
            
            # Debug print for similarity scores
            if similarity_scores:
                sorted_scores = sorted(similarity_scores.items(), key=lambda x: x[1], reverse=True)
                print(f"Face recognition scores: {sorted_scores[:3]}")  # Top 3 scores
            
            # Initialize recognition history for this face
            if face_id not in self.recognition_history:
                self.recognition_history[face_id] = []
            
            # Add current recognition to history
            if best_match and best_similarity >= self.recognition_threshold:
                self.recognition_history[face_id].append((best_match, best_similarity))
                
                # Keep only last 5 recognitions
                if len(self.recognition_history[face_id]) > 5:
                    self.recognition_history[face_id] = self.recognition_history[face_id][-5:]
                
                # Use voting mechanism for stability
                if len(self.recognition_history[face_id]) >= 3:
                    recent_recognitions = self.recognition_history[face_id][-3:]
                    name_votes = {}
                    
                    for name, score in recent_recognitions:
                        if score >= self.recognition_threshold:
                            name_votes[name] = name_votes.get(name, 0) + 1
                    
                    if name_votes:
                        # Get the name with most votes
                        most_voted = max(name_votes.items(), key=lambda x: x[1])
                        if most_voted[1] >= 2:  # At least 2 votes out of 3
                            return most_voted[0], best_similarity
                
                return best_match, best_similarity
            else:
                # Clear history for faces that don't meet threshold
                if face_id in self.recognition_history:
                    del self.recognition_history[face_id]
                return 'Unknown Person', best_similarity
            
        except Exception as e:
            print(f"Error in face recognition: {e}")
            return 'Unknown Person', 0.0


    def login_user(self, name):
        """Login a user and start tracking time"""
        if name != 'Unknown Person' and name not in self.logged_in_users:
            login_time = self.attendance_logger.log_login(name)
            self.logged_in_users[name] = login_time
            total_today = self.attendance_logger.get_total_time_today(name)
            messagebox.showinfo("Login Successful", 
                              f"{name} logged in at {login_time.strftime('%H:%M:%S')}\n"
                              f"Total time today: {total_today:.1f} minutes")
            self.update_attendance_display()

    def logout_user(self, name):
        """Logout a user and calculate session duration"""
        if name in self.logged_in_users:
            duration = self.attendance_logger.log_logout(name)
            if duration:
                total_today = self.attendance_logger.get_total_time_today(name)
                messagebox.showinfo("Logout Successful", 
                                  f"{name} logged out\n"
                                  f"Session duration: {duration:.1f} minutes\n"
                                  f"Total time today: {total_today:.1f} minutes")
                del self.logged_in_users[name]
                self.update_attendance_display()

    def manual_login(self):
        """Manual login for recognized users"""
        if not hasattr(self, 'last_recognized_name') or self.last_recognized_name == 'Unknown Person':
            messagebox.showwarning("No User", "No recognized user found. Please face the camera within the red detection zone.")
            return
        
        self.login_user(self.last_recognized_name)

    def manual_logout(self):
        """Manual logout for logged-in users"""
        if not self.logged_in_users:
            messagebox.showwarning("No Users", "No users are currently logged in.")
            return
        
        if len(self.logged_in_users) == 1:
            name = list(self.logged_in_users.keys())[0]
            self.logout_user(name)
        else:
            names = list(self.logged_in_users.keys())
            choice = simpledialog.askstring("Logout", 
                                           f"Multiple users logged in: {', '.join(names)}\n"
                                           f"Enter name to logout:")
            if choice and choice in self.logged_in_users:
                self.logout_user(choice)

    def update_attendance_display(self):
        """Update the attendance display"""
        if self.logged_in_users:
            logged_in_text = "Logged in: " + ", ".join(self.logged_in_users.keys())
        else:
            logged_in_text = "No users logged in"
        
        self.attendance_label.config(text=logged_in_text)

    def upload_face_to_database(self):
        """Capture current frame, detect face, ask for label, and save to database"""
        if self.most_recent_capture_arr is None:
            messagebox.showerror("Error", "No image available from webcam to capture.")
            return

        frame = self.most_recent_capture_arr.copy()
        faces = self.detect_faces_in_roi(frame)  # Use ROI detection
        if len(faces) == 0:
            messagebox.showinfo("No Face Detected", "No face detected in the detection zone.\nPlease position your face clearly within the red rectangle.")
            return

        best_face_data = max(faces, key=lambda x: x['confidence'])
        face_img = best_face_data['face']
        confidence = best_face_data['confidence']

        label = simpledialog.askstring(
            "Face Upload", 
            f"Face detected with {confidence:.2f} confidence.\nEnter the name for this person:",
            parent=self.main_window
        )
        
        if label is None or label.strip() == "":
            messagebox.showwarning("No Label", "No name provided. Face not saved.")
            return

        label = label.strip()

        try:
            person_folder = os.path.join('./face_database', label)
            os.makedirs(person_folder, exist_ok=True)

            timestamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
            img_filename = os.path.join(person_folder, f'{timestamp}.jpg')
            
            success = cv2.imwrite(img_filename, face_img)
            if not success:
                messagebox.showerror("Error", "Failed to save face image.")
                return

            new_encoding = self.get_face_encoding(face_img)
            if new_encoding is not None:
                if label in self.face_database:
                    existing_enc = self.face_database[label]
                    # Weighted average: 70% existing, 30% new
                    self.face_database[label] = (existing_enc * 0.7 + new_encoding * 0.3)
                    # Normalize
                    self.face_database[label] = self.face_database[label] / np.linalg.norm(self.face_database[label])
                    message = f"Updated face data for '{label}'.\nImage saved as: {os.path.basename(img_filename)}"
                else:
                    self.face_database[label] = new_encoding
                    message = f"Added new person '{label}' to database.\nImage saved as: {os.path.basename(img_filename)}"
                
                messagebox.showinfo("Success", message)
                self.status_label.config(text=f"Database updated - {len(self.face_database)} people loaded")
                
            else:
                messagebox.showerror("Error", "Failed to generate face encoding. Please try again with a clearer face image.")
                if os.path.exists(img_filename):
                    os.remove(img_filename)
                    
        except Exception as e:
            messagebox.showerror("Error", f"Failed to save face to database: {str(e)}")
    
    def setup_gui(self):
        """Setup the GUI components for full screen with ROI controls"""
        
        # Get screen dimensions
        screen_width = self.main_window.winfo_screenwidth()
        screen_height = self.main_window.winfo_screenheight()
        
        # Webcam display - larger for full screen
        webcam_width = screen_width - 100
        webcam_height = screen_height - 200
        
        self.webcam_label = Label(self.main_window, bg='black')
        self.webcam_label.place(x=50, y=80, width=webcam_width, height=webcam_height)
        
        # Status label at top
        self.status_label = Label(
            self.main_window, 
            text="DeepFace FaceNet Face Recognition System - Attendance Tracker with ROI", 
            font=('Arial', 16, 'bold'),
            bg='lightblue',
            fg='black'
        )
        self.status_label.place(x=0, y=0, width=screen_width, height=40)
        
        # Attendance display
        self.attendance_label = Label(
            self.main_window, 
            text="No users logged in | Position your face in the red detection zone", 
            font=('Arial', 14),
            bg='lightyellow',
            fg='black'
        )
        self.attendance_label.place(x=0, y=40, width=screen_width, height=35)
        
        # Button panel at bottom
        button_y = screen_height - 120
        button_width = 150
        button_height = 40
        
        # Login button
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
        
        # Logout button
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
        
        # Upload Face button
        self.upload_button = Button(
            self.main_window, 
            text="📷 ADD FACE", 
            command=self.upload_face_to_database,
            bg='lightblue',
            fg='black',
            font=('Arial', 12, 'bold'),
            relief='raised',
            bd=3
        )
        self.upload_button.place(x=390, y=button_y, width=button_width, height=button_height)
        
        # ROI Toggle button
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
        self.roi_button.place(x=560, y=button_y, width=button_width, height=button_height)
        
        # Exit button
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
        
        # Initialize webcam
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
    def check_auto_logout(self, currently_detected_names):
        """Check for automatic logout when person is no longer detected"""
        current_time = datetime.datetime.now()
        
        for logged_in_user in list(self.logged_in_users.keys()):
            if logged_in_user not in currently_detected_names:
                # Check if user has been absent for more than 30 seconds
                if (logged_in_user not in self.last_auto_login_time or
                    (current_time - self.last_auto_login_time.get(logged_in_user, current_time)).seconds > 30):
                    
                    duration = self.attendance_logger.log_logout(logged_in_user)
                    if duration:
                        total_today = self.attendance_logger.get_total_time_today(logged_in_user)
                        messagebox.showinfo("Auto Logout", 
                                        f"🔴 {logged_in_user} automatically logged out\n"
                                        f"Reason: No longer detected\n"
                                        f"Session duration: {duration:.1f} minutes\n"
                                        f"Total time today: {total_today:.1f} minutes")
                        del self.logged_in_users[logged_in_user]
                        if logged_in_user in self.last_auto_login_time:
                            del self.last_auto_login_time[logged_in_user]
                        print(f"AUTO-LOGOUT: {logged_in_user} logged out automatically")
                        self.update_attendance_display()

    def auto_login_user(self, name, similarity):
        """Automatically login a user when similarity exceeds threshold"""
        if name != 'Unknown Person' and name not in self.logged_in_users:
            login_time = self.attendance_logger.log_login(name)
            self.logged_in_users[name] = login_time
            total_today = self.attendance_logger.get_total_time_today(name)
            
            # Show popup notification for auto-login
            messagebox.showinfo("Auto Login Successful", 
                            f"🟢 {name} automatically logged in!\n"
                            f"Similarity: {similarity:.1%}\n"
                            f"Time: {login_time.strftime('%H:%M:%S')}\n"
                            f"Total time today: {total_today:.1f} minutes")
            self.update_attendance_display()
            print(f"AUTO-LOGIN: {name} logged in automatically with {similarity:.1%} similarity")

    def process_webcam(self):
        """Process webcam frames with automatic login at 85% similarity threshold"""
        if self.cap and self.cap.isOpened():
            ret, frame = self.cap.read()
            if ret:
                self.most_recent_capture_arr = frame.copy()
                img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Calculate ROI if not set
                h, w, _ = frame.shape
                if self.roi_x2 == 0 or self.roi_y2 == 0:
                    self.calculate_roi(w, h)
                
                # Draw ROI first
                self.draw_roi(img_rgb)
                
                try:
                    # Use ROI-based face detection
                    face_info = self.detect_faces_in_roi(frame)
                    recognized_names = []
                    current_time = datetime.datetime.now()
                    
                    for idx, face_data in enumerate(face_info):
                        bbox = face_data['bbox']
                        x, y, width, height = bbox
                        confidence = face_data['confidence']
                        
                        face_id = f"face_{x}_{y}_{width}_{height}"
                        # Get both name and similarity score
                        name, similarity = self.recognize_face_with_facenet(face_data['face'], face_id)
                        recognized_names.append(name)
                        
                        # Store last recognized name for manual login
                        if name != 'Unknown Person':
                            self.last_recognized_name = name
                            
                            # AUTO LOGIN LOGIC: Check if similarity > 85% and user not logged in
                            if (similarity >= self.auto_login_threshold and 
                                name not in self.logged_in_users and
                                name not in self.auto_login_cooldown):
                                
                                # Add cooldown to prevent rapid multiple logins
                                self.auto_login_cooldown[name] = current_time
                                self.last_auto_login_time[name] = current_time
                                
                                # Automatically log in the user
                                self.auto_login_user(name, similarity)
                            
                            # Clear cooldown after 5 seconds
                            if (name in self.auto_login_cooldown and 
                                (current_time - self.auto_login_cooldown[name]).seconds > 5):
                                del self.auto_login_cooldown[name]
                        
                        # Choose color based on recognition and login status
                        if name == 'Unknown Person':
                            box_color = (255, 0, 0)  # Red
                            text_bg_color = (255, 0, 0)
                        elif name in self.logged_in_users:
                            box_color = (0, 255, 0)  # Green (logged in)
                            text_bg_color = (0, 255, 0)
                        else:
                            box_color = (255, 255, 0)  # Yellow (recognized but not logged in)
                            text_bg_color = (255, 255, 0)
                        
                        # Draw rectangle and label with similarity score
                        cv2.rectangle(img_rgb, (x, y), (x + width, y + height), box_color, 3)
                        
                        login_status = " [LOGGED IN]" if name in self.logged_in_users else ""
                        # Show similarity percentage for better feedback
                        display_text = f"{name}{login_status} ({similarity:.1%})"
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.9
                        font_thickness = 2
                        
                        (text_width, text_height), _ = cv2.getTextSize(display_text, font, font_scale, font_thickness)
                        text_x = max(0, x + (width - text_width) // 2)
                        text_y = max(text_height + 10, y - 10)
                        
                        # Draw background rectangle for text
                        cv2.rectangle(img_rgb, 
                                    (text_x - 5, text_y - text_height - 5), 
                                    (text_x + text_width + 5, text_y + 5), 
                                    text_bg_color, -1)
                        
                        # Draw the text
                        cv2.putText(img_rgb, display_text, (text_x, text_y), 
                                font, font_scale, (0, 0, 0), font_thickness)
                        
                        # Add auto-login indicator for high similarity
                        if similarity >= self.auto_login_threshold and name != 'Unknown Person':
                            indicator_text = "AUTO-LOGIN READY" if name not in self.logged_in_users else "AUTO-LOGGED IN"
                            indicator_y = y + height + 25
                            cv2.putText(img_rgb, indicator_text, (x, indicator_y), 
                                    font, 0.6, (0, 255, 255), 2)
                    
                    # Update status with auto-login info
                    roi_status = "ROI Enabled" if self.roi_enabled else "ROI Disabled"
                    model_status = "DeepFace FaceNet (Auto-Login: 85%)"
                    
                    if recognized_names:
                        known_names = [name for name in recognized_names if name != 'Unknown Person']
                        unknown_count = recognized_names.count('Unknown Person')
                        
                        status_parts = []
                        if known_names:
                            unique_known = list(set(known_names))
                            status_parts.append(f"Recognized: {', '.join(unique_known)}")
                        if unknown_count > 0:
                            status_parts.append(f"Unknown: {unknown_count}")
                        
                        status_text = " | ".join(status_parts) + f" | Database: {len(self.face_database)} people | {model_status} | {roi_status}"
                    else:
                        status_text = f"{model_status} - {len(self.face_database)} people in database | {roi_status}"

                    self.status_label.config(text=status_text)
                    
                    # Update attendance display
                    if self.logged_in_users:
                        logged_in_text = "Logged in: " + ", ".join(self.logged_in_users.keys())
                    else:
                        logged_in_text = "No users logged in"
                    
                    if self.roi_enabled:
                        logged_in_text += " | Position your face in the red detection zone for auto-login at 85% similarity"
                    
                    self.attendance_label.config(text=logged_in_text)
                    
                except Exception as e:
                    print(f"Error in face processing: {e}")
                
                # Display the frame
                self.most_recent_capture_pil = Image.fromarray(img_rgb)
                imgtk = ImageTk.PhotoImage(image=self.most_recent_capture_pil)
                self.webcam_label.imgtk = imgtk
                self.webcam_label.configure(image=imgtk)
        
        # Schedule next frame
        if hasattr(self, 'webcam_label'):
            self.webcam_label.after(20, self.process_webcam)

    
    def start(self):
        """Start the application"""
        try:
            self.main_window.protocol("WM_DELETE_WINDOW", self.on_closing)
            # Press ESC to exit full screen
            self.main_window.bind('<Escape>', lambda e: self.on_closing())
            # Press R to toggle ROI
            self.main_window.bind('<Key-r>', lambda e: self.toggle_roi())
            self.main_window.bind('<Key-R>', lambda e: self.toggle_roi())
            self.main_window.focus_set()  # Enable key bindings
            self.main_window.mainloop()
        except Exception as e:
            print(f"Error in main loop: {e}")
        finally:
            self.cleanup()
    
    def on_closing(self):
        """Handle window closing"""
        # Logout all users before closing
        for name in list(self.logged_in_users.keys()):
            self.logout_user(name)
        
        self.cleanup()
        self.main_window.destroy()
    
    def cleanup(self):
        """Cleanup resources"""
        if hasattr(self, 'cap') and self.cap:
            self.cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    os.makedirs('./face_database', exist_ok=True)
    
    print("Starting DeepFace FaceNet Face Recognition Attendance System with ROI...")
    print("Features:")
    print("- Full screen camera display")
    print("- Improved DeepFace FaceNet model for accurate face recognition")
    print("- Better face preprocessing and normalization")
    print("- Enhanced similarity comparison")
    print("- MediaPipe for real-time face detection") 
    print("- Region of Interest (ROI) for focused face detection")
    print("- Login/Logout tracking with Excel logging")
    print("- Real-time face recognition with voting mechanism")
    print("- Attendance duration calculation")
    print("\nImprovements Made:")
    print("- Better face preprocessing with CLAHE enhancement")
    print("- Normalized embeddings for consistent comparison")
    print("- Improved padding strategy for face cropping")
    print("- Lowered threshold to 0.65 for better recognition")
    print("- Enhanced voting mechanism for stability")
    print("- Better error handling and debugging output")
    print("\nModel Requirements:")
    print("- DeepFace will automatically download FaceNet model on first use")
    print("- No manual model download required")
    print("\nDependencies:")
    print("- pip install deepface mediapipe opencv-python pandas openpyxl scikit-learn")
    print("\nControls:")
    print("- Green box: User is logged in")
    print("- Yellow box: User recognized but not logged in")
    print("- Red box: Unknown person")
    print("- Red rectangle: Face detection zone (ROI)")
    print("- Press ESC to exit")
    print("- Press R to toggle ROI on/off")
    print("- Click 'TOGGLE ROI' button to enable/disable ROI")
    
    try:
        app = FaceRecognitionApp()
        app.start()
    except Exception as e:
        print(f"Failed to start application: {e}")
