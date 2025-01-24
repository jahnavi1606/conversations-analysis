import streamlit as st
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity

# Simplified sentiment analysis using TextBlob
@st.cache_data
def analyze_sentiment(conversations):
    sentiments = []
    for text in conversations:
        blob = TextBlob(text)
        polarity = blob.sentiment.polarity
        if polarity > 0.1:
            sentiments.append("Positive")
        elif polarity < -0.1:
            sentiments.append("Negative")
        else:
            sentiments.append("Neutral")
    return sentiments

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

# Clustering function
@st.cache_data
def cluster_conversations(conversations, num_clusters=5):
    vectorizer = TfidfVectorizer(max_features=100, stop_words="english")
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

# Streamlit app interface
def main():
    st.title("Conversation Analysis App")
    
    uploaded_file = st.file_uploader("Upload JSON/JSONL file", type=["jsonl", "json"])
    
    if uploaded_file:
        df = load_file(uploaded_file)
        if df is not None:
            st.success("File loaded successfully!")
            
            if "conversations" in df.columns:
                # Preprocess conversations
                df["conversations"] = df["conversations"].apply(
                    lambda x: " ".join([c["value"] for c in x]) if isinstance(x, list) else str(x)
                )
                
                # Clustering
                if "topic" not in df.columns:
                    st.info("Clustering conversations. This might take a moment...")
                    df["topic"] = cluster_conversations(df["conversations"])
                
                # Sentiment Analysis
                if "sentiment" not in df.columns:
                    st.info("Analyzing sentiment. Please wait...")
                    df["sentiment"] = analyze_sentiment(df["conversations"])
                
                # Display results
                st.write(df.head())
                
                # Download button
                st.download_button(
                    label="Download Results as CSV",
                    data=df.to_csv(index=False),
                    file_name="conversation_analysis_results.csv",
                    mime="text/csv",
                )
            else:
                st.error("Missing 'conversations' column.")
    else:
        st.info("Upload a file to analyze.")

if __name__ == "__main__":
    main()
