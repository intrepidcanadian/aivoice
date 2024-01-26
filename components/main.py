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

from parselmouth.praat import call
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


# mysp=__import__("my-voice-analysis")
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
                    feedback_data = json.load(file)

                    for key in feedback_data:
                        if not isinstance(feedback_data[key], dict):
                            feedback_data[key] = {'Feedback': feedback_data[key]}
                    
                    return feedback_data
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

def convert_mp3_to_wav(directory):
    for file in os.listdir(directory):
        if file.endswith('.mp3'):
            mp3_path = os.path.join(directory, file)
            wav_path = os.path.join(directory, os.path.splitext(file)[0] + '.wav')

            # Convert mp3 to wav
            audio = AudioSegment.from_mp3(mp3_path)
            audio.export(wav_path, format="wav")

            os.remove(mp3_path)

def measurePitch(voiceID, f0min, f0max, unit):

    sound = parselmouth.Sound(voiceID) # read the sound
    pitch = call(sound, "To Pitch", 0.0, f0min, f0max) #create a praat pitch object
    meanF0 = call(pitch, "Get mean", 0, 0, unit) # get mean pitch
    stdevF0 = call(pitch, "Get standard deviation", 0 ,0, unit) # get standard deviation
    harmonicity = call(sound, "To Harmonicity (cc)", 0.01, 75, 0.1, 1.0)
    hnr = call(harmonicity, "Get mean", 0, 0)
    pointProcess = call(sound, "To PointProcess (periodic, cc)", f0min, f0max)
    localJitter = call(pointProcess, "Get jitter (local)", 0, 0, 0.0001, 0.02, 1.3)
    localabsoluteJitter = call(pointProcess, "Get jitter (local, absolute)", 0, 0, 0.0001, 0.02, 1.3)
    rapJitter = call(pointProcess, "Get jitter (rap)", 0, 0, 0.0001, 0.02, 1.3)
    ppq5Jitter = call(pointProcess, "Get jitter (ppq5)", 0, 0, 0.0001, 0.02, 1.3)
    ddpJitter = call(pointProcess, "Get jitter (ddp)", 0, 0, 0.0001, 0.02, 1.3)
    localShimmer =  call([sound, pointProcess], "Get shimmer (local)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    localdbShimmer = call([sound, pointProcess], "Get shimmer (local_dB)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq3Shimmer = call([sound, pointProcess], "Get shimmer (apq3)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    aqpq5Shimmer = call([sound, pointProcess], "Get shimmer (apq5)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    apq11Shimmer =  call([sound, pointProcess], "Get shimmer (apq11)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    ddaShimmer = call([sound, pointProcess], "Get shimmer (dda)", 0, 0, 0.0001, 0.02, 1.3, 1.6)
    
    return meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer

def runPCA(df):
    #Z-score the Jitter and Shimmer measurements
    features = ['localJitter', 'localabsoluteJitter', 'rapJitter', 'ppq5Jitter', 'ddpJitter',
                'localShimmer', 'localdbShimmer', 'apq3Shimmer', 'apq5Shimmer', 'apq11Shimmer', 'ddaShimmer']
    # Separating out the features
    x = df.loc[:, features].values
    # Separating out the target
    #y = df.loc[:,['target']].values
    # Standardizing the features
    x = StandardScaler().fit_transform(x)
    #PCA
    pca = PCA(n_components=2)
    principalComponents = pca.fit_transform(x)
    principalDf = pd.DataFrame(data = principalComponents, columns = ['JitterPCA', 'ShimmerPCA'])
    principalDf
    return principalDf

def audio_clip():
    record_audio()

def create_main_content(): 

    feedback_data = load_feedback()

    st.header('Fine-tune your search by providing feedback on the below voice recordings')
    st.write("Voice Recordings:")

    convert_mp3_to_wav(audio_directory)
    audio_files = [file for file in os.listdir(audio_directory) if file.endswith('.wav')]
    
    if st.checkbox("Process Audio Files for Pitch and Shimmer"):
        for audio_name in audio_files:
            file_path = os.path.join(audio_directory, audio_name)
            sound = parselmouth.Sound(file_path)

            # Calculate pitch and shimmer
            (meanF0, stdevF0, hnr, localJitter, localabsoluteJitter, rapJitter, ppq5Jitter, ddpJitter, localShimmer, localdbShimmer, apq3Shimmer, aqpq5Shimmer, apq11Shimmer, ddaShimmer) = measurePitch(sound, 75, 500, "Hertz")

            # Update feedback data with pitch and shimmer values
            feedback_data.setdefault(audio_name, {}).update({
                'Pitch': meanF0,
                'PitchSD': stdevF0,
                'HNR': hnr,
                'Jitter': localJitter
            })
        save_feedback(feedback_data)

    if st.checkbox("Show Audio Files"):

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

            #calculate gender based on mysp library
            # file_path_without_extension = os.path.splitext(audio_name)[0]
            # print("this is the file path extension", file_path_without_extension)
            # gender_prediction = mysp.myspf0med(file_path_without_extension, audio_directory)  
            
            col1, col2, col3 = st.columns(3)
            if col1.button('Like', key=f'{audio_name}_like'):
                feedback_data.setdefault(audio_name, {}).update({'Like': 1, 'Dislike': 0, 'Undecided': 0})
                save_feedback(feedback_data)
                parselresults = load_audio_file(audio_name)
                plt.figure()
                plt.plot(parselresults.xs(), parselresults.values.T)
                plt.xlim([parselresults.xmin, parselresults.xmax])
                plt.xlabel("time [s]")
                plt.ylabel("amplitude")
                st.pyplot(plt.gcf())
                
            if col2.button('Dislike', key=f'{audio_name}_dislike'):
                feedback_data.setdefault(audio_name, {}).update({'Like': 0, 'Dislike': 1, 'Undecided': 0})
                save_feedback(feedback_data)
            if col3.button('Undecided', key=f'{audio_name}_undecided'):
                feedback_data.setdefault(audio_name, {}).update({'Like': 0, 'Dislike': 0, 'Undecided': 1})
                save_feedback(feedback_data)

    if st.checkbox("Show Feedback Summary"):
        if feedback_data:

            st.write("Feedback Summary:")

            summary_data = {
                'File Name': [], 
                'Like': [], 
                'Dislike': [], 
                'Undecided': [], 
                'Pitch': [],  # meanF0Hz
                'PitchSD': [],  # stdevF0Hz
                'HNR': [],     # HNR
                'Jitter': []   # localJitter
            }

            for file_name, file_feedback in feedback_data.items():
                summary_data['File Name'].append(file_name)
                if isinstance(file_feedback, dict):
                    summary_data['Like'].append(file_feedback.get('Like', 0))
                    summary_data['Dislike'].append(file_feedback.get('Dislike', 0))
                    summary_data['Undecided'].append(file_feedback.get('Undecided', 0))
                    summary_data['Pitch'].append(file_feedback.get('Pitch', 'N/A'))
                    summary_data['PitchSD'].append(file_feedback.get('PitchSD', 'N/A'))
                    summary_data['HNR'].append(file_feedback.get('HNR', 'N/A'))
                    summary_data['Jitter'].append(file_feedback.get('Jitter', 'N/A'))
                else:
                    # Handle non-dict feedback
                    summary_data['Like'].append(0)
                    summary_data['Dislike'].append(0)
                    summary_data['Undecided'].append(0)
                    summary_data['Pitch'].append('N/A')
                    summary_data['PitchSD'].append('N/A')
                    summary_data['HNR'].append('N/A')
                    summary_data['Jitter'].append('N/A')

            import pandas as pd
            df = pd.DataFrame(summary_data)

            # Display the summary table and save it so that it can be drawn to app.py
            st.table(df)
            st.session_state['df'] = df

        else:
            st.write("No feedback data available.")
