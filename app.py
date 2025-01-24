import streamlit as st
import pandas as pd
from textblob import TextBlob
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import MiniBatchKMeans

# Load the file and return a DataFrame
def load_file(file):
    try:
        if file.name.endswith(".jsonl"):
            df = pd.read_json(file, lines=True)
        else:
            df = pd.read_json(file)
        return df
    except Exception as e:
        st.error(f"Error reading file: {e}")
        return None

# Perform sentiment analysis using TextBlob
def simple_sentiment_analysis(texts):
    def get_sentiment(text):
        try:
            analysis = TextBlob(text)
            if analysis.sentiment.polarity > 0:
                return "Positive"
            elif analysis.sentiment.polarity < 0:
                return "Negative"
            else:
                return "Neutral"
        except Exception:
            return "Unknown"
    
    return [get_sentiment(text) for text in texts]

# Cluster conversations using TF-IDF and MiniBatchKMeans
def cluster_conversations(conversations, num_clusters=5):
    vectorizer = TfidfVectorizer(max_features=50, stop_words="english")
    try:
        tfidf_matrix = vectorizer.fit_transform(conversations)
        kmeans = MiniBatchKMeans(n_clusters=num_clusters, random_state=42, batch_size=50)
        clusters = kmeans.fit_predict(tfidf_matrix)

        # Define topic labels
        topic_labels = {
            0: "Topic 1",
            1: "Topic 2",
            2: "Topic 3",
            3: "Topic 4",
            4: "Topic 5"
        }
        return [topic_labels.get(cluster, "Other") for cluster in clusters]
    except Exception as e:
        st.error(f"Error clustering conversations: {e}")
        return ["Unknown"] * len(conversations)

# Preprocess conversation data for analysis
def preprocess_conversations(df):
    if "conversations" in df.columns:
        df["conversations"] = df["conversations"].apply(
            lambda x: " ".join([c["value"] for c in x]) if isinstance(x, list) else str(x)
        )
    else:
        st.warning("The input file must contain a 'conversations' column.")
    return df

# Counts screen: Display aggregated counts
def counts_page(df):
    st.header("Counts Overview")

    # Topic Counts
    st.subheader("Topic Count")
    topic_counts = df["topic"].value_counts().reset_index()
    topic_counts.columns = ["Topic", "Count"]
    st.table(topic_counts)

    # Sentiment Counts
    st.subheader("Sentiment Count")
    sentiment_counts = df["sentiment"].value_counts().reset_index()
    sentiment_counts.columns = ["Sentiment", "Count"]
    st.table(sentiment_counts)

# Sessions screen: Display conversation details
def sessions_page(df):
    st.header("Sessions Overview")
    
    # Paginated view of conversations
    st.subheader("Conversation Details")
    page_size = 50
    page_number = st.number_input("Page Number", min_value=1, max_value=(len(df) // page_size) + 1, step=1)
    start_idx = (page_number - 1) * page_size
    end_idx = start_idx + page_size
    
    paginated_df = df.iloc[start_idx:end_idx]
    st.table(paginated_df[["conversations", "topic", "sentiment"]].reset_index().rename(
        columns={"index": "Conversation No", "conversations": "Conversation", "topic": "Topic", "sentiment": "Sentiment"}
    ))

# Main application
def main():
    st.title("Conversation Analyzer")
    
    uploaded_file = st.file_uploader("Upload JSON/JSONL", type=["jsonl", "json"])
    if uploaded_file:
        df = load_file(uploaded_file)
        if df is not None:
            df = preprocess_conversations(df)
            if "conversations" in df.columns:
                # Generate topics and sentiments if not already present
                if "topic" not in df.columns:
                    df["topic"] = cluster_conversations(df["conversations"])
                if "sentiment" not in df.columns:
                    df["sentiment"] = simple_sentiment_analysis(df["conversations"])
                
                # Page navigation
                page = st.sidebar.radio("Select Page", ["Counts", "Sessions"])
                if page == "Counts":
                    counts_page(df)
                elif page == "Sessions":
                    sessions_page(df)
            else:
                st.error("The uploaded file does not contain a 'conversations' column.")

if __name__ == "__main__":
    main()
