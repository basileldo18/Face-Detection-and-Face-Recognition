# Face-Detection-and-Face-Recognition

Overview
This project is a real-time face recognition and registration system that integrates multiple technologies for robust identification and event tracking. It leverages advanced computer vision, deep learning, and GUI/web server components to provide an end-to-end solution for attendance, login/logout management, and persistent ID tracking.

Key Features
Real-time face recognition using DeepFace's FaceNet and MediaPipe FaceMesh.

Multi-factor person tracking: Combines facial landmarks, face encodings, and centroid matching for persistent ID assignment and re-identification.

Region-of-Interest (ROI) tracking: Provides robust entry/exit detection and assigns identities after a configurable multi-frame evaluation period.

Head pose estimation: Uses 3D geometry and MediaPipe for live pose analysis and login stability.

Attendance logging: Tracks login/logout time and displays attendance status dynamically.

Integrated GUI (Tkinter): Visualizes camera feed, detection info, status bars, and login/logout buttons.

Web registration server (Flask): Supports user registration via photo upload, pausing main recognition, and automatic resumption.

Database-backed face recognition: Loads face features from folders/images, supports adding new users via the registration interface.

Persistent ID assignment: Maintains the same ID for a person even after logout or temporary disappearance, with robust re-identification.

Installation
Dependencies
Ensure the following Python packages are installed:

bash
pip install opencv-python Flask deepface mediapipe scikit-learn pillow numpy pandas
Other libraries used:

tkinter (usually included in standard Python installations)

sklearn

threading

os, math, collections

Additional Requirements
Install DeepFace dependencies for FaceNet model support.

Make sure your webcam drivers are installed and accessible to OpenCV.

For MediaPipe, install the official pip package.

Optionally, configure appropriate GPU support for acceleration with DeepFace and OpenCV.

Usage
Start the Application
bash
python app.py
Functional Flow
GUI launches: Shows main tracking window, status bar, and control buttons.

Live tracking: The app processes webcam frames in real-time, detecting and tracking faces within ROI.

Login/Logout: Users can login/logout using the GUI controls. Automatic login occurs if head pose and recognition are stable.

Registration: Click Register User to open a web registration page (Flask, served at http://localhost:8000/register).

Photo capture: Registration captures images from 4 angles with photo validation.

Attendance: The app records login/logout events, showing accumulated daily attendance time.

Core Modules
HeadPoseEstimator: Estimates yaw, pitch, roll using MediaPipe and OpenCV geometry.

PersonTracker: Manages tracked persons, persistent IDs, and multi-factor matching.

GUI Application (Tkinter): Handles visualization and user interaction.

WebRegistrationServer (Flask): Manages photo registration, pauses real-time recognition during registration.

Face Recognition: DeepFace-based for robust identification with embeddings and cosine similarity.

Configuration
ROI boundary: Can be toggled via GUI; controls when recognition/evaluation is triggered.

Number of frames for evaluation: Default is 5 consecutive frames inside ROI before final assignment.

Recognition thresholds: Adjust for fine-tuning (see code comments for defaults).

Directory Structure
Place your face database images in a folder. The system loads each person's images (subfolders per person) as the face database during startup and registration.

text
project_root/
  app.py
  face_database/
    PersonA/
      FrontView.jpg
      LeftProfile.jpg
    PersonB/
      ...
Technical Highlights
Advanced matching: Combines landmark, encoding, centroid matching for re-identification.

Seamless GUI and browser integration: Tkinter for desktop UI, Flask for web-based registration.

Automatic head pose-based login maintenance: Monitors for significant movement to prevent unauthorized/stale logins.

Persistent identity logic: IDs are recycled and reassigned for returning users after logout or disappearing from view.

Running & Developing
Ensure all dependencies are met.

Start the app (python app.py) in an environment with camera access.

For new registrations, use the web interface while the main recognition system is paused.

Tune the recognition and head pose thresholds as needed for your deployment scenario.

License
Specify your project license here.

Contributions
Feel free to submit improvements/fixes via pull requests.

Credits
This project uses:

OpenCV for image processing

DeepFace for deep face recognition

MediaPipe for facial landmark/pose estimation

Tkinter for GUI

Flask for web registration
