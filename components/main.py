import cohere
import streamlit as st
import os
import st_audiorec as staud
import json
import parselmouth
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns

from pydub import AudioSegment
from dotenv import load_dotenv
from datetime import datetime
from glob import glob

audio_directory = './audio'
feedback_file_path = './data/feedback.json'
previous_audio_data = None
matplotlib.use("Agg")

# SECTION ARE FUNCTIONS
def load_feedback():

    try:
        if os.path.exists(feedback_file_path):
            with open(feedback_file_path, 'r') as file:
                try:
                    return json.load(file)
                except json.JSONDecodeError:
                    return {}
        else:
            return {}
    except Exception as e:
        st.error(f"Error loading feedback: {e}")
        return {}

def save_feedback(feedback):
    with open(feedback_file_path, 'w') as file:
        json.dump(feedback, file)

def record_audio():
    global previous_audio_data  
    audio_data = staud.st_audiorec()

    if audio_data is not None:
        # Check if the new audio data is different from the previous one before saving
        if previous_audio_data is None or audio_data != previous_audio_data:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            file_name = f'audio_{timestamp}.wav'
            file_path = os.path.join(audio_directory, file_name)  

            with open(file_path, 'wb') as f:
                f.write(audio_data)

            st.success(f"Audio saved to {file_path}")
            previous_audio_data = audio_data

        else:
            st.write("")
            #same audio as previous, do nothing

def audio_clip():
    record_audio()

def create_main_content(): 

    feedback_data = load_feedback()
    st.header('Rate voices and help us fine tune our search algorithm')
    st.write("Voice Recordings:")

    if st.checkbox("Show Audio Files"):
        # Create a list to store audio components
        audio_files = [file for file in os.listdir('./audio') if file.endswith('.wav')]

        def load_audio_file(file_name):
            if file_name.startswith('./'):
                file_name = file_name[1:]
            file_path = os.path.join(audio_directory, file_name) 
            print("this is ", file_path)
            try:
                sound = parselmouth.Sound(file_path)
                print("this is sound", sound)
                return sound
            except Exception as e:
                print(f"Error loading audio file: {e}")
                return None

        for audio_name in audio_files:
            sns.set()
            plt.rcParams['figure.dpi'] = 100
            file_path = f'./audio/{audio_name}'
            st.write(f"**{audio_name}**")
            st.audio(file_path, format='audio/wav')
            parselresults = load_audio_file(audio_name)
            
            col1, col2, col3 = st.columns(3)
            if col1.button('Like', key=f'{audio_name}_like'):
                feedback_data[audio_name] = 'Like'
                save_feedback(feedback_data)
                
                if parselresults:
                    plt.figure()
                    plt.plot(parselresults.xs(), parselresults.values.T)
                    plt.xlim([parselresults.xmin, parselresults.xmax])
                    plt.xlabel("time [s]")
                    plt.ylabel("amplitude")
                    st.pyplot(plt.gcf())
            if col2.button('Dislike', key=f'{audio_name}_dislike'):
                feedback_data[audio_name] = 'Dislike'
                save_feedback(feedback_data)
            if col3.button('Undecided', key=f'{audio_name}_undecided'):
                feedback_data[audio_name] = 'Undecided'
                save_feedback(feedback_data)
        
    if st.checkbox("Show Feedback Summary"):
        if feedback_data:

            st.write("Feedback Summary:")

            # Prepare data for the summary table
            summary_data = {'File Name': [], 'Like': [], 'Dislike': [], 'Undecided': []}

            for file_name, file_feedback in feedback_data.items():
                summary_data['File Name'].append(file_name)
                summary_data['Like'].append(1 if file_feedback == 'Like' else 0)
                summary_data['Dislike'].append(1 if file_feedback == 'Dislike' else 0)
                summary_data['Undecided'].append(1 if file_feedback == 'Undecided' else 0)

            # Convert to a Pandas DataFrame for better display
            import pandas as pd
            df = pd.DataFrame(summary_data)

            # Display the summary table
            st.table(df)

        else:
            st.write("No feedback data available.")
