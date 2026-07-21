import os
import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Set page config
st.set_page_config(
    page_title="Face Recognition Check",
    page_icon="👤",
    layout="centered",
    initial_sidebar_state="collapsed"
)

# Custom CSS for rich aesthetics
st.markdown("""
    <style>
        /* Modern typography & colors */
        @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;600;800&display=swap');
        
        html, body, [class*="css"] {
            font-family: 'Outfit', sans-serif;
        }
        
        /* Header styling */
        .title-container {
            text-align: center;
            padding: 2rem 0 1rem 0;
        }
        
        .main-title {
            background: linear-gradient(90deg, #ff4b4b 0%, #ff8585 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            font-weight: 800;
            font-size: 3rem;
            margin-bottom: 0.5rem;
        }
        
        .sub-title {
            color: #8a99ad;
            font-size: 1.1rem;
            font-weight: 300;
        }
        
        /* Status Card styling */
        .status-card {
            background: rgba(255, 255, 255, 0.05);
            border-radius: 12px;
            padding: 1.5rem;
            border: 1px solid rgba(255, 255, 255, 0.1);
            margin-bottom: 2rem;
            backdrop-filter: blur(10px);
            text-align: center;
        }
        
        /* Match Result styling */
        .result-container {
            border-radius: 12px;
            padding: 1.5rem;
            text-align: center;
            margin-top: 1.5rem;
            animation: fadeIn 0.5s ease-out;
        }
        
        .result-match {
            background: rgba(46, 213, 115, 0.15);
            border: 1px solid rgba(46, 213, 115, 0.3);
            color: #2ed573;
        }
        
        .result-no-match {
            background: rgba(255, 71, 87, 0.15);
            border: 1px solid rgba(255, 71, 87, 0.3);
            color: #ff4757;
        }
        
        .result-text {
            font-size: 1.8rem;
            font-weight: 600;
            margin: 0;
        }
        
        .confidence-text {
            font-size: 1rem;
            color: #8a99ad;
            margin-top: 0.5rem;
        }
        
        @keyframes fadeIn {
            from { opacity: 0; transform: translateY(10px); }
            to { opacity: 1; transform: translateY(0); }
        }
    </style>
""", unsafe_allow_html=True)

# Application Header
st.markdown("""
    <div class="title-container">
        <h1 class="main-title">Face Recognition Check</h1>
        <p class="sub-title">Real-time OpenCV face recognition against reference photo</p>
    </div>
""", unsafe_allow_html=True)

REFERENCE_IMAGE_PATH = "person1.jpg.jpg"

@st.cache_resource
def load_and_train_recognizer(image_path):
    if not os.path.exists(image_path):
        return None, None, f"Reference image '{image_path}' not found."
    
    ref_img = cv2.imread(image_path)
    if ref_img is None:
        return None, None, f"Unable to read reference image '{image_path}'."
    
    gray_ref = cv2.cvtColor(ref_img, cv2.COLOR_BGR2GRAY)
    
    # Load Haar Cascade classifier using cv2.data.haarcascades
    cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
    if not os.path.exists(cascade_path) and os.path.exists("haarcascade_frontalface_default.xml"):
        cascade_path = "haarcascade_frontalface_default.xml"
        
    face_cascade = cv2.CascadeClassifier(cascade_path)
    
    # Fallback check if default path yielded empty classifier
    if face_cascade.empty() and os.path.exists("haarcascade_frontalface_default.xml"):
        cascade_path = "haarcascade_frontalface_default.xml"
        face_cascade = cv2.CascadeClassifier(cascade_path)
        
    if face_cascade.empty():
        raise RuntimeError(f"Failed to load Haar Cascade classifier from path: {cascade_path}")
    
    faces = face_cascade.detectMultiScale(gray_ref, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    if len(faces) == 0:
        return None, None, f"No face detected in reference photo '{image_path}'."
    
    # Extract the largest face ROI
    (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
    ref_face_roi = gray_ref[y:y+h, x:x+w]
    
    # Create and train LBPH face recognizer
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    recognizer.train([ref_face_roi], np.array([1]))
    
    return recognizer, face_cascade, None

recognizer, face_cascade, err = load_and_train_recognizer(REFERENCE_IMAGE_PATH)

if err:
    st.error(f"Initialization Error: {err}")
    st.info(f"Please ensure `{REFERENCE_IMAGE_PATH}` exists in the repository root directory.")
else:
    # Display system status card
    st.markdown("""
        <div class="status-card">
            <h4 style="margin-top:0; color:#ff4b4b; font-size: 1.2rem;">Registered Target</h4>
            <p style="margin-bottom:0; font-size: 1.1rem;">Scanning for: <strong>Person1</strong></p>
        </div>
    """, unsafe_allow_html=True)

    # Webcam camera input
    st.write("### Capture Photo")
    picture = st.camera_input("Position your face in the center of the camera frame")

    if picture is not None:
        with st.spinner("Analyzing face..."):
            try:
                # Convert camera input to OpenCV BGR format
                img = Image.open(picture)
                img_np = np.array(img)
                
                if len(img_np.shape) == 3 and img_np.shape[2] == 4:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGBA2BGR)
                elif len(img_np.shape) == 3 and img_np.shape[2] == 3:
                    img_np = cv2.cvtColor(img_np, cv2.COLOR_RGB2BGR)
                
                gray_captured = cv2.cvtColor(img_np, cv2.COLOR_BGR2GRAY)
                
                # Detect faces in captured frame
                faces = face_cascade.detectMultiScale(gray_captured, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
                
                if len(faces) == 0:
                    st.warning("No face detected in the photo. Please adjust lighting and try again.")
                else:
                    # Select largest face ROI
                    (x, y, w, h) = max(faces, key=lambda rect: rect[2] * rect[3])
                    captured_roi = gray_captured[y:y+h, x:x+w]
                    
                    label, confidence = recognizer.predict(captured_roi)
                    
                    # LBPH distance: lower is better match (0 = perfect match)
                    CONFIDENCE_THRESHOLD = 75.0
                    
                    if confidence < CONFIDENCE_THRESHOLD:
                        st.markdown(f"""
                            <div class="result-container result-match">
                                <p class="result-text">Match: Person1</p>
                                <p class="confidence-text">Confidence Distance: {confidence:.1f}</p>
                            </div>
                        """, unsafe_allow_html=True)
                        st.balloons()
                    else:
                        st.markdown(f"""
                            <div class="result-container result-no-match">
                                <p class="result-text">No match found</p>
                                <p class="confidence-text">Confidence Distance: {confidence:.1f}</p>
                            </div>
                        """, unsafe_allow_html=True)
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
