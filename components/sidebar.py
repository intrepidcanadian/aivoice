import cohere
import streamlit as st
import os
import json

from collections import Counter
from dotenv import load_dotenv
from datetime import datetime
from glob import glob

load_dotenv()
api_key = os.getenv('COHERE_API_KEY')
co = cohere.Client(api_key)

def get_audio_clip_count():
    audio_files = glob('./audio/*.wav')
    print(audio_files)
    return len(audio_files)

def count_feedback():
    feedback_file_path = './data/feedback.json'
    print(feedback_file_path)
    try:
        if os.path.exists(feedback_file_path):
            with open(feedback_file_path, 'r') as file:
                feedback_data = json.load(file)
                num_likes = sum(1 for feedback in feedback_data.values() if feedback == 'Like')
                num_dislikes = sum(1 for feedback in feedback_data.values() if feedback == 'Dislike')
                num_undecided = sum(1 for feedback in feedback_data.values() if feedback == 'Undecided')
                feedback_counts = {
                    'Likes': num_likes,
                    'Dislikes': num_dislikes,
                    'Undecided': num_undecided
                }
                print(feedback_counts)
                return feedback_counts
             
        else:
            return {'Likes': 0, 'Dislikes': 0, 'Undecided': 0}
    except Exception as e:
        st.error(f"Error loading feedback: {e}")
        return {'Likes': 0, 'Dislikes': 0, 'Undecided': 0}

def create_sidebar():
    st.sidebar.title('Radical AIFMC x Cohere GenerativeAI Hackathon')
    st.sidebar.caption('January 2024')

    #Display the number of audio clips stored
    num_audio_clips = get_audio_clip_count()
    st.sidebar.metric(label="Recordings", value=num_audio_clips)

    #Count and display feedback
    feedback_counts = count_feedback()
    st.sidebar.metric(label="Likes", value=feedback_counts['Likes'])
    st.sidebar.metric(label="Dislikes", value=feedback_counts['Dislikes'])
    st.sidebar.metric(label="Undecided", value=feedback_counts['Undecided'])


