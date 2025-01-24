import streamlit as st
import pandas as pd
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm  # Progress bar for sentiment analysis

# Load the sentiment analysis pipeline
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# File loading function
@st.cache_data
def load_file(file):
    try:
        if file.name.endswith(".jsonl"):
            df = pd.read_json(file, lines=True)
        elif file.name.endswith(".json"):
            df = pd.read_json(file)
        else:
            st.error("Unsupported file format. Please upload a JSON or JSONL file.")
            return None
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# Sentiment analysis
@st.cache_data
def analyze_sentiment(conversations, batch_size=32):
    sentiment_pipeline = load_sentiment_pipeline()
    sentiments = []
    for i in tqdm(range(0, len(conversations), batch_size), desc="Processing Sentiment"):
        batch = conversations[i:i+batch_size]
        batch = [text[:256] for text in batch]
        results = sentiment_pipeline(batch)
        for res in results:
            label = res['label']
            if label == "LABEL_2":
                sentiments.append("Negative")
            elif label == "LABEL_1":
                sentiments.append("Neutral")
            else:
                sentiments.append("Positive")
    return sentiments

# Clustering
@st.cache_data
def cluster_conversations(conversations, num_clusters=5):
    vectorizer = TfidfVectorizer(max_features=200, stop_words="english")
    tfidf_matrix = vectorizer.fit_transform(conversations)
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=50)
    clusters = kmeans.fit_predict(tfidf_matrix)
    topic_labels = {0: "Programming", 1: "Technical Issues", 2: "App Dev", 3: "Feedback", 4: "Misc"}
    similarity_matrix = cosine_similarity(tfidf_matrix)
    for idx, row in enumerate(similarity_matrix):
        if sum(row > 0.2) <= 1:
            clusters[idx] = len(topic_labels) - 1
    topics = [topic_labels[cluster] for cluster in clusters]
    return topics

st.title("Conversation Analysis App")

uploaded_file = st.file_uploader("Upload JSON/JSONL file", type=["jsonl", "json"])

if uploaded_file:
    df = load_file(uploaded_file)
    if df is not None:
        st.success("File loaded successfully!")
        if "conversations" in df.columns:
            df["conversations"] = df["conversations"].apply(
                lambda x: " ".join([c["value"] for c in x]) if isinstance(x, list) else str(x)
            )
            if "topic" not in df.columns:
                df["topic"] = cluster_conversations(df["conversations"])
            if "sentiment" not in df.columns:
                df["sentiment"] = analyze_sentiment(df["conversations"])
            st.write(df.head())
        else:
            st.error("Missing 'conversations' column.")
else:
    st.info("Upload a file to analyze.")


