import streamlit as st
import pandas as pd
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans
from sklearn.metrics.pairwise import cosine_similarity

# Load the sentiment analysis pipeline using a model that supports Neutral sentiment
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="cardiffnlp/twitter-roberta-base-sentiment")

# Read and validate the uploaded file
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

# Sentiment analysis with Neutral support
def get_sentiment(text):
    """Get sentiment using a model that includes neutral sentiment."""
    result = sentiment_analyzer(text[:512])[0]  # Truncate to avoid token limit
    label = result['label']
    
    # Assign sentiment labels based on the returned 'label'
    if label == "LABEL_2":  # Negative sentiment (model-specific)
        return "Negative"
    elif label == "LABEL_1":  # Neutral sentiment
        return "Neutral"
    else:  # Positive sentiment (model-specific)
        return "Positive"

# Analyze sentiment (batch processing and truncation)
@st.cache_data
def analyze_sentiment(conversations, batch_size=64):
    sentiment_pipeline = load_sentiment_pipeline()
    sentiments = []
    for i in range(0, len(conversations), batch_size):
        batch = conversations[i:i+batch_size]
        batch = [text[:256] for text in batch]  # Truncate text to 256 characters
        results = sentiment_pipeline(batch)
        
        for res in results:
            label = res['label']
            # Map labels to Positive, Negative, Neutral
            if label == "LABEL_2":
                sentiments.append("Negative")
            elif label == "LABEL_1":
                sentiments.append("Neutral")
            else:
                sentiments.append("Positive")
    return sentiments

# Cluster conversations into topics
@st.cache_data
def cluster_conversations(conversations, num_clusters=5):
    vectorizer = TfidfVectorizer(max_features=300, stop_words="english")  # Fewer features for speed
    tfidf_matrix = vectorizer.fit_transform(conversations)

    # Use MiniBatchKMeans for faster clustering
    kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=100)
    clusters = kmeans.fit_predict(tfidf_matrix)

    # Assign topic labels
    topic_labels = {
        0: "Programming Practices",
        1: "Technical Issues",
        2: "App Development",
        3: "Customer Feedback",
        4: "Misc"
    }

    # Assign "Misc" for isolated conversations
    similarity_matrix = cosine_similarity(tfidf_matrix)
    for idx, row in enumerate(similarity_matrix):
        if sum(row > 0.2) <= 1:  # Threshold for isolation
            clusters[idx] = len(topic_labels) - 1  # Assign "Misc"

    topics = [topic_labels[cluster] for cluster in clusters]
    return topics, topic_labels

# Streamlit Layout
st.title("Conversation Analysis App")

# File uploader
uploaded_file = st.file_uploader("Upload a JSONL or JSON file", type=["jsonl", "json"])

if uploaded_file:
    # Load the data
    df = load_file(uploaded_file)

    if df is not None:
        st.success("File loaded successfully!")

        # Ensure 'conversations' column exists and preprocess it
        if "conversations" in df.columns:
            df["conversations"] = df["conversations"].apply(
                lambda x: " ".join([c["value"] for c in x]) if isinstance(x, list) else str(x)
            )

            # Sidebar navigation
            st.sidebar.title("Navigation")
            screen = st.sidebar.radio("Choose a screen", ["Counts", "Sessions"])

            # Cluster conversations into topics (if not already done)
            if "topic" not in df.columns:
                topics, topic_labels = cluster_conversations(df["conversations"])
                df["topic"] = topics

            # Analyze sentiment for each conversation (if not already done)
            if "sentiment" not in df.columns:
                df["sentiment"] = analyze_sentiment(df["conversations"])

            if screen == "Counts":
                # Screen 1: Counts
                st.header("Counts")
                st.subheader("Conversation Count by Topic")
                topic_counts = df["topic"].value_counts()
                st.write(topic_counts)

                st.subheader("Conversation Count by Sentiment")
                sentiment_counts = df["sentiment"].value_counts()
                st.write(sentiment_counts)

            elif screen == "Sessions":
                # Screen 2: Sessions
                st.header("Sessions")
                st.subheader("Conversations with Topics and Sentiments")
                page = st.slider("Select Page", 1, (len(df) // 50) + 1, 1)
                start_idx = (page - 1) * 50
                end_idx = start_idx + 50
                st.write(df[["conversations", "topic", "sentiment"]].iloc[start_idx:end_idx])
        else:
            st.error("The uploaded file does not contain a 'conversations' column.")
else:
    st.info("Please upload a file to begin analysis.")
