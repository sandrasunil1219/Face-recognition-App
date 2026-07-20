import os
import pickle
import numpy as np
import streamlit as st
import face_recognition
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
        <p class="sub-title">Verify identity against saved face encodings in real-time</p>
    </div>
""", unsafe_allow_html=True)

# Helper function to load encodings
@st.cache_data
def load_encodings(file_path):
    if not os.path.exists(file_path):
        return None, "File not found."
    try:
        with open(file_path, "rb") as f:
            data = pickle.load(f)
        return data, None
    except Exception as e:
        return None, str(e)

# Load database
encodings_file = "face_encodings.pkl"
data, err = load_encodings(encodings_file)

if err:
    st.error(f"Error loading face encodings: {err}")
    st.info("Please make sure `face_encodings.pkl` exists in the application directory.")
else:
    # Display system status card
    known_name = data.get('name', 'Unknown')
    st.markdown(f"""
        <div class="status-card">
            <h4 style="margin-top:0; color:#ff4b4b; font-size: 1.2rem;">Registered Target</h4>
            <p style="margin-bottom:0; font-size: 1.1rem;">Scanning for: <strong>{known_name}</strong></p>
        </div>
    """, unsafe_allow_html=True)

    # Webcam camera input
    st.write("### Capture Photo")
    picture = st.camera_input("Position your face in the center of the camera frame")

    if picture is not None:
        with st.spinner("Analyzing face..."):
            try:
                # Load image
                img = Image.open(picture)
                img_array = np.array(img)
                
                # Convert to RGB (face_recognition expects RGB)
                if img_array.shape[-1] == 4:
                    img_array = img_array[:, :, :3]
                
                # Find face encodings
                face_locations = face_recognition.face_locations(img_array)
                
                if not face_locations:
                    st.warning("No face detected in the photo. Please adjust lighting and try again.")
                else:
                    # Encode captured face
                    captured_encodings = face_recognition.face_encodings(img_array, face_locations)
                    
                    if captured_encodings:
                        known_encoding = data['encoding']
                        
                        # Compare face encodings
                        matches = face_recognition.compare_faces([known_encoding], captured_encodings[0], tolerance=0.6)
                        # Compute distance to show confidence
                        face_distances = face_recognition.face_distance([known_encoding], captured_encodings[0])
                        distance = face_distances[0]
                        
                        # Convert distance to a percentage-like confidence score
                        if distance < 0.6:
                            confidence = (1 - distance / 1.2) * 100 # Maps 0.0 -> 100%, 0.6 -> 50%
                        else:
                            confidence = (1 - distance) * 100
                            confidence = max(0.0, min(49.9, confidence))
                        
                        if matches[0]:
                            st.markdown(f"""
                                <div class="result-container result-match">
                                    <p class="result-text">Match: {known_name}</p>
                                    <p class="confidence-text">Confidence Score: {confidence:.1f}% (Distance: {distance:.3f})</p>
                                </div>
                            """, unsafe_allow_html=True)
                            st.balloons()
                        else:
                            st.markdown(f"""
                                <div class="result-container result-no-match">
                                    <p class="result-text">No match found</p>
                                    <p class="confidence-text">Similarity score too low (Distance: {distance:.3f})</p>
                                </div>
                            """, unsafe_allow_html=True)
                    else:
                        st.warning("Failed to extract face features. Please try again.")
            except Exception as e:
                st.error(f"An error occurred during processing: {str(e)}")
