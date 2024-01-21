# app.py

import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns
import cohere
import numpy as np
import pandas as pd
import plotly.express as px

from sklearn.cluster import KMeans
from components.sidebar import create_sidebar
from components.main import create_main_content, audio_clip
from sklearn.manifold import TSNE

from sklearn.decomposition import PCA

st.set_page_config(layout="wide")

create_main_content()

# audio_clip()

create_sidebar()


st.header("Use Cohere Embeddings to File Name (contains name and some description) and Cluster based on User Feedback / Audio Characteristics")

if 'df' not in st.session_state:
    st.session_state['df'] = pd.DataFrame()

api_key = st.session_state.get('api_key', '')
print(api_key)

if api_key:
    co = cohere.Client(api_key)
    df = st.session_state['df']
    file_names = df['File Name'].tolist()
    embeddings = co.embed(texts=file_names, model='embed-english-v3.0', input_type='classification').embeddings
    # print(embeddings)
    embeddings = np.array(embeddings)

    pca = PCA(n_components=2)
    reduced_embeddings = pca.fit_transform(embeddings)
    features_for_clustering = np.hstack([reduced_embeddings, df[['Like', 'Dislike']].values])

    kmeans = KMeans(n_clusters=2)
    clusters = kmeans.fit_predict(features_for_clustering)
    df['Cluster'] = clusters

    sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=df['Cluster'])
    fig = px.scatter(
        x=reduced_embeddings[:, 0],
        y=reduced_embeddings[:, 1],
        color=df['Cluster'], 
        labels={'color': 'Cluster'},
        hover_name=df['File Name']
    )

    fig.update_traces(textposition='top center')

    st.plotly_chart(fig)

    # #show graph for voice characteristics

    tsne = TSNE(n_components=2, random_state=0, perplexity=30, learning_rate=100, n_iter=1000)
    reduced_embeddings = tsne.fit_transform(embeddings)

    audio_features = df[['Pitch', 'PitchSD', 'HNR', 'Jitter']]
    normalized_audio_features = (audio_features - audio_features.mean()) / audio_features.std()

    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=normalized_audio_features['Pitch'], ax=axs[0, 0])
    axs[0, 0].set_title('Pitch')

    sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=normalized_audio_features['PitchSD'], ax=axs[0, 1])
    axs[0, 1].set_title('PitchSD')

    sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=normalized_audio_features['HNR'], ax=axs[1, 0])
    axs[1, 0].set_title('HNR')

    sns.scatterplot(x=reduced_embeddings[:, 0], y=reduced_embeddings[:, 1], hue=normalized_audio_features['Jitter'], ax=axs[1, 1])
    axs[1, 1].set_title('Jitter')

    st.pyplot(fig)