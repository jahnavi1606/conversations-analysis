import streamlit as st
import pandas as pd
from transformers import pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import LatentDirichletAllocation

# Load the sentiment analysis pipeline once and cache it
@st.cache_resource
def load_sentiment_pipeline():
    return pipeline("sentiment-analysis", model="distilbert-base-uncased")

# Cache the file reading logic
@st.cache_data
def load_file(file):
    try:
        # Check file extension or content
        if file.name.endswith(".jsonl"):
            df = pd.read_json(file, lines=True)
        elif file.name.endswith(".json"):
            df = pd.read_json(file)
        else:
            raise ValueError("Unsupported file format. Please upload a JSONL or JSON file.")
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# Cache the sentiment analysis results
@st.cache_data
def analyze_sentiment(conversations, batch_size=32):
    sentiment_pipeline = load_sentiment_pipeline()
    sentiments = []
    for i in range(0, len(conversations), batch_size):
        batch = conversations[i:i+batch_size]
        try:
            results = sentiment_pipeline(batch)
            sentiments.extend([res['label'].lower() for res in results])
        except Exception as e:
            sentiments.extend(['neutral'] * len(batch))  # Default to 'neutral' on error
    return sentiments

# Cache the topic modeling process
@st.cache_data
def get_topics(conversations, num_topics=5):
    vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(conversations)

    lda = LatentDirichletAllocation(n_components=num_topics, random_state=42)
    topics = lda.fit_transform(tfidf_matrix)

    return lda, topics, vectorizer

# Streamlit Layout
st.title('Conversation Analysis App')

# Sidebar for navigation
st.sidebar.title("Navigation")
option = st.sidebar.radio("Select Screen", ["Count", "Sessions"])

# Upload file
uploaded_file = st.sidebar.file_uploader("Choose a JSONL or JSON file", type=["jsonl", "json"])

if uploaded_file is not None:
    # Load the file
    df = load_file(uploaded_file)
    if df is not None:
        st.sidebar.success("Data loaded successfully!")

        # Combine all messages into one conversation
        if 'conversations' in df.columns:
            df['conversations'] = df['conversations'].apply(
                lambda x: " ".join([c['value'] for c in x]) if isinstance(x, list) else str(x)
            )

            # Get Topics using LDA
            lda, topics, vectorizer = get_topics(df['conversations'])
            df['topic'] = topics.argmax(axis=1)

            # Assign topic labels
            topic_labels = {0: "Stove Issues", 1: "Technical Support", 2: "General Questions", 3: "Miscellaneous", 4: "Feedback"}
            df['topic_label'] = df['topic'].map(topic_labels)

            # Assign sentiments
            df['sentiment'] = analyze_sentiment(df['conversations'])

            # Handle different screens
            if option == "Count":
                # Topic Count
                st.header("Count")
                st.subheader("Topic Count")
                topic_count = df['topic_label'].value_counts()
                st.write(topic_count)

                # Sentiment Count
                st.subheader("Sentiment Count")
                sentiment_count = df['sentiment'].value_counts()
                st.write(sentiment_count)

            elif option == "Sessions":
                # Sessions - Display a paginated list of conversations with topics and sentiments
                st.header("Sessions")
                st.subheader("Conversations with Topics and Sentiments")
                page = st.slider("Select Page", 1, (len(df) // 50) + 1, 1)
                start_idx = (page - 1) * 50
                end_idx = page * 50
                st.write(df[['conversations', 'topic_label', 'sentiment']].iloc[start_idx:end_idx])
        else:
            st.error("The uploaded file does not contain a 'conversations' column. Please check the file format.")
else:
    st.write("Please upload a JSONL or JSON file to get started.")