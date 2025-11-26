import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import pickle
from tensorflow import keras
from gtts import gTTS
import os
from collections import deque
import tempfile

# Page configuration
st.set_page_config(
    page_title="SignSpeak AI",
    page_icon="🤟",
    layout="wide"
)

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 3rem;
        font-weight: bold;
        text-align: center;
        color: #1E88E5;
        margin-bottom: 0;
    }
    .sub-header {
        text-align: center;
        color: #666;
        margin-bottom: 2rem;
    }
    .prediction-box {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        text-align: center;
        color: white;
        font-size: 4rem;
        font-weight: bold;
        margin: 1rem 0;
        box-shadow: 0 10px 25px rgba(0,0,0,0.2);
    }
    .sentence-box {
        background: #f8f9fa;
        padding: 1.5rem;
        border-radius: 10px;
        border-left: 5px solid #1E88E5;
        margin: 1rem 0;
        font-size: 1.5rem;
        min-height: 80px;
    }
    .stats-box {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        border: 2px solid #e0e0e0;
        margin: 0.5rem 0;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model_and_encoder():
    """Load trained model and label encoder"""
    model = keras.models.load_model('models/asl_model.h5')
    with open('models/label_encoder.pkl', 'rb') as f:
        label_encoder = pickle.load(f)
    return model, label_encoder

class ASLTranslator:
    def __init__(self, model, label_encoder):
        self.model = model
        self.label_encoder = label_encoder
        
        # Initialize MediaPipe
        self.mp_hands = mp.solutions.hands
        self.mp_drawing = mp.solutions.drawing_utils
        self.hands = self.mp_hands.Hands(
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7,
            max_num_hands=1
        )
        
        # Prediction smoothing
        self.prediction_buffer = deque(maxlen=5)
        self.confidence_threshold = 0.7
        
    def extract_landmarks(self, hand_landmarks):
        """Extract hand landmarks"""
        landmarks = []
        for landmark in hand_landmarks.landmark:
            landmarks.extend([landmark.x, landmark.y, landmark.z])
        return np.array(landmarks)
    
    def predict_sign(self, landmarks):
        """Predict sign from landmarks"""
        landmarks = landmarks.reshape(1, -1)
        predictions = self.model.predict(landmarks, verbose=0)
        
        confidence = np.max(predictions)
        predicted_class = np.argmax(predictions)
        predicted_label = self.label_encoder.inverse_transform([predicted_class])[0]
        
        return predicted_label, confidence
    
    def get_stable_prediction(self, landmarks):
        """Get stable prediction using buffer"""
        prediction, confidence = self.predict_sign(landmarks)
        
        if confidence > self.confidence_threshold:
            self.prediction_buffer.append(prediction)
            
            if len(self.prediction_buffer) >= 3:
                # Get most common prediction
                most_common = max(set(self.prediction_buffer), key=self.prediction_buffer.count)
                return most_common, confidence
        
        return prediction, confidence

def text_to_speech(text):
    """Convert text to speech"""
    if text:
        tts = gTTS(text=text, lang='en', slow=False)
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp3') as fp:
            tts.save(fp.name)
            return fp.name
    return None

def main():
    # Header
    st.markdown('<h1 class="main-header">🤟 SignSpeak AI</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Real-Time Sign Language Translator</p>', unsafe_allow_html=True)
    
    # Check if model exists
    if not os.path.exists('models/asl_model.h5'):
        st.error("⚠️ Model not found! Please train the model first by running `train_model.py`")
        st.stop()
    
    # Load model
    try:
        model, label_encoder = load_model_and_encoder()
        translator = ASLTranslator(model, label_encoder)
        st.success("✅ Model loaded successfully!")
    except Exception as e:
        st.error(f"❌ Error loading model: {str(e)}")
        st.stop()
    
    # Sidebar
    st.sidebar.title("⚙️ Settings")
    
    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.5,
        max_value=1.0,
        value=0.7,
        step=0.05
    )
    translator.confidence_threshold = confidence_threshold
    
    show_landmarks = st.sidebar.checkbox("Show Hand Landmarks", value=True)
    show_confidence = st.sidebar.checkbox("Show Confidence", value=True)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📋 Instructions")
    st.sidebar.markdown("""
    1. Click **Start Camera**
    2. Show ASL signs to the camera
    3. Hold each sign for 2-3 seconds
    4. Predicted letters appear below
    5. Click **Add to Sentence** to build words
    6. Use **Speak** to hear the sentence
    """)
    
    st.sidebar.markdown("---")
    st.sidebar.markdown("### ℹ️ About")
    st.sidebar.info(f"Trained on {len(label_encoder.classes_)} ASL signs")
    
    # Main interface
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.subheader("📹 Live Camera Feed")
        camera_placeholder = st.empty()
    
    with col2:
        st.subheader("🎯 Current Prediction")
        prediction_placeholder = st.empty()
        confidence_placeholder = st.empty()
        
        st.markdown("---")
        
        add_button = st.button("➕ Add to Sentence", use_container_width=True)
        clear_button = st.button("🗑️ Clear Sentence", use_container_width=True)
    
    # Sentence building
    st.markdown("---")
    st.subheader("📝 Sentence Builder")
    
    sentence_col1, sentence_col2 = st.columns([3, 1])
    
    with sentence_col1:
        sentence_placeholder = st.empty()
    
    with sentence_col2:
        speak_button = st.button("🔊 Speak", use_container_width=True)
    
    # Statistics
    st.markdown("---")
    stats_col1, stats_col2, stats_col3 = st.columns(3)
    
    with stats_col1:
        total_predictions = st.empty()
    with stats_col2:
        avg_confidence = st.empty()
    with stats_col3:
        letters_added = st.empty()
    
    # Camera control
    start_camera = st.checkbox("Start Camera", value=False)
    
    # Session state
    if 'sentence' not in st.session_state:
        st.session_state.sentence = ""
    if 'current_prediction' not in st.session_state:
        st.session_state.current_prediction = ""
    if 'current_confidence' not in st.session_state:
        st.session_state.current_confidence = 0.0
    if 'total_pred_count' not in st.session_state:
        st.session_state.total_pred_count = 0
    if 'confidence_sum' not in st.session_state:
        st.session_state.confidence_sum = 0.0
    
    # Handle buttons
    if add_button and st.session_state.current_prediction:
        st.session_state.sentence += st.session_state.current_prediction
        st.rerun()
    
    if clear_button:
        st.session_state.sentence = ""
        st.rerun()
    
    if speak_button and st.session_state.sentence:
        audio_file = text_to_speech(st.session_state.sentence)
        if audio_file:
            st.audio(audio_file, format='audio/mp3')
            os.unlink(audio_file)
    
    # Camera processing
    if start_camera:
        cap = cv2.VideoCapture(0)
        
        try:
            while start_camera:
                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to access camera")
                    break
                
                frame = cv2.flip(frame, 1)
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame
                results = translator.hands.process(rgb_frame)
                
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        # Draw landmarks
                        if show_landmarks:
                            translator.mp_drawing.draw_landmarks(
                                frame,
                                hand_landmarks,
                                translator.mp_hands.HAND_CONNECTIONS
                            )
                        
                        # Extract and predict
                        landmarks = translator.extract_landmarks(hand_landmarks)
                        prediction, confidence = translator.get_stable_prediction(landmarks)
                        
                        # Update session state
                        st.session_state.current_prediction = prediction
                        st.session_state.current_confidence = confidence
                        st.session_state.total_pred_count += 1
                        st.session_state.confidence_sum += confidence
                        
                        # Display on frame
                        cv2.putText(frame, f"{prediction} ({confidence:.2f})", 
                                  (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 
                                  1.5, (0, 255, 0), 3)
                
                # Display frame
                camera_placeholder.image(frame, channels="BGR", use_container_width=True)
                
                # Update UI
                if st.session_state.current_prediction:
                    prediction_placeholder.markdown(
                        f'<div class="prediction-box">{st.session_state.current_prediction}</div>',
                        unsafe_allow_html=True
                    )
                    
                    if show_confidence:
                        confidence_placeholder.metric(
                            "Confidence",
                            f"{st.session_state.current_confidence*100:.1f}%"
                        )
                
                # Update sentence
                sentence_placeholder.markdown(
                    f'<div class="sentence-box">{st.session_state.sentence}</div>',
                    unsafe_allow_html=True
                )
                
                # Update stats
                total_predictions.metric("Total Predictions", st.session_state.total_pred_count)
                
                if st.session_state.total_pred_count > 0:
                    avg_conf = (st.session_state.confidence_sum / st.session_state.total_pred_count) * 100
                    avg_confidence.metric("Avg Confidence", f"{avg_conf:.1f}%")
                
                letters_added.metric("Letters Added", len(st.session_state.sentence))
                
        finally:
            cap.release()
    else:
        camera_placeholder.info("👆 Check 'Start Camera' to begin")
        prediction_placeholder.markdown(
            '<div class="prediction-box">-</div>',
            unsafe_allow_html=True
        )
        sentence_placeholder.markdown(
            f'<div class="sentence-box">{st.session_state.sentence}</div>',
            unsafe_allow_html=True
        )

if __name__ == "__main__":
    main()