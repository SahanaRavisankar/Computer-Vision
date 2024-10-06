import cv2
import streamlit as st
import time
import os
from datetime import datetime
import numpy as np
from PIL import Image
import tensorflow as tf
from model import generate_caption, get_caption_model  # Ensure this imports correctly
import easyocr


st.set_page_config(
    page_title="World in a Frame",
    page_icon="📷",  # Replace with your chosen emoji
    layout="centered",
    initial_sidebar_state="expanded"
)

def image_to_text(image_path):
    # Initialize the EasyOCR reader
    reader = easyocr.Reader(['en'])
    
    # Perform OCR on the image
    result = reader.readtext(image_path, detail=0)
    
    # Join the result text into a single string
    text = ' '.join(result)
    
    return text

from gtts import gTTS
from playsound import playsound

def text_to_speech(text, audio_file_path):
    # Convert text to speech
    tts = gTTS(text=text, lang='en')
    
    # Save the speech to a file
    tts.save(audio_file_path)
    
    # Play the audio in the background without opening a file system window
    playsound(audio_file_path)

# Create directories for storing images and audio
os.makedirs('captured_images', exist_ok=True)
os.makedirs('captions', exist_ok=True)

# Initialize state for webcam and image saving
if 'webcam_on' not in st.session_state:
    st.session_state['webcam_on'] = False

if 'toggle_count' not in st.session_state:
    st.session_state['toggle_count'] = 0

# Sidebar for navigation
st.sidebar.title("Navigation")
page = st.sidebar.selectbox("Select a page", ("Home", "History"))

# Function to capture and store images every 5 seconds and generate captions
def run_webcam():
    cap = cv2.VideoCapture(0)
    last_capture_time = time.time()

    # Load the caption model with error handling
    try:
        if 'caption_model' not in st.session_state:
            st.session_state['caption_model'] = get_caption_model()  # Load model once
    except Exception as e:
        print(f"Error loading caption model: {e}")
        return  # Exit if the model couldn't be loaded

    # Initialize a placeholder for the image display
    FRAME_WINDOW = st.empty()

    while st.session_state['webcam_on']:
        ret, frame = cap.read()
        if not ret:
            st.warning("Could not access the webcam.")
            break

        # Convert the frame from BGR to RGB to display it correctly in Streamlit
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        FRAME_WINDOW.image(frame_rgb)

        # Capture image every 5 seconds
        current_time = time.time()
        if current_time - last_capture_time >= 5:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"captured_images/pic_{timestamp}.jpg"
            cv2.imwrite(filename, frame)  # Save the image
            last_capture_time = current_time

            try:
                img = Image.open(filename)
                if img.mode == 'RGBA':
                    img = img.convert('RGB')

                img = img.resize((299, 299))
                img_array = np.array(img) / 255.0

                if img_array.shape[-1] != 3:
                    raise ValueError("The captured image doesn't have 3 channels (RGB).")

                img_tensor = tf.convert_to_tensor(img_array, dtype=tf.float32)
                caption = generate_caption(img_tensor, st.session_state['caption_model'])
                st.write("Generated Caption:", caption)

            except Exception as e:
                print(f"Error processing image: {e}")
                st.error("An error occurred while processing the image. Please check the console for details.")

    cap.release()

# Display content based on the selected page
if page == "Home":
    st.title("World in a Frame")

    st.subheader("Webcam real-time Capture")
    # Toggle webcam and increase toggle count
    if st.button('Press Here to Toggle Webcam'):
        st.session_state['webcam_on'] = not st.session_state['webcam_on']
        st.session_state['toggle_count'] += 1

    # If button pressed second time, reload page using JavaScript
    if st.session_state['toggle_count'] > 1:
        st.session_state['toggle_count'] = 0  # Reset count
        st.markdown("""<script>location.reload();</script>""", unsafe_allow_html=True)

    # Display webcam status
    if st.session_state['webcam_on']:
        st.success("Webcam is ON")
        run_webcam()
    else:
        st.info("Webcam is OFF")

    # Image Captioning
    st.subheader("Upload to get real-time Text")
    uploaded_file = st.file_uploader("Upload an image file", type=["jpg", "jpeg", "png"])

    if uploaded_file is not None:
        img = Image.open(uploaded_file)
        
        st.image(img, caption="Uploaded Image", use_column_width=True)

        # Save the uploaded image temporarily
        image_path = f"captured_images/temp_image_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        img.save(image_path)

        try:
            # Extract text from the uploaded image using OCR
            extracted_text = image_to_text(image_path)
            st.write("Extracted Text:", extracted_text)

            # Generate audio from the extracted text
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            audio_file_path = f"captions/caption_{timestamp}.mp3"
            text_to_speech(extracted_text, audio_file_path)

            #st.success("Audio generated and saved successfully.")
            # Audio is saved but not played or displayed in the Home page

        except Exception as e:
            st.error(f"An error occurred: {e}")
    else:
        st.warning("Please upload an image file.")

elif page == "History":
    st.title("Audio History")

    # List audio files in the 'captions' folder
    captions_folder = 'captions'
    audio_files = [f for f in os.listdir(captions_folder) if f.endswith('.mp3')]
    audio_files.sort(key=lambda x: os.path.getmtime(os.path.join(captions_folder, x)), reverse=True)  # Sort by modification time

    # Display audio files with play buttons and timestamps
    st.write("Generated Audio Files:")
    if audio_files:
        for audio_file in audio_files:
            audio_path = os.path.join(captions_folder, audio_file)
            # Get the modification time and format it
            mod_time = os.path.getmtime(audio_path)
            timestamp = datetime.fromtimestamp(mod_time).strftime('%Y-%m-%d %H:%M:%S')

            # Create two columns: one for audio, one for timestamp
            col1, col2 = st.columns([3, 1])  # Create two columns

            with col1:
                st.audio(audio_path, format="audio/mp3")  # Display audio player
            
            with col2:
                st.write(timestamp)  # Display the timestamp
    else:
        st.write("No audio files available in history.")
