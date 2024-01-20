# app.py

import streamlit as st
from components.sidebar import create_sidebar
from components.main import create_main_content, audio_clip

st.set_page_config(layout="wide")

st.title("Submit a sample of your voice so we can help find matches for you!")

audio_clip()

create_main_content()

create_sidebar()
