import streamlit as st
import pickle
import re
import numpy as np

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords
import nltk

nltk.download('stopwords')

# ---------------------------------
# Page configuration
# ---------------------------------
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    layout="centered"
)

# ---------------------------------
# Load model and objects
# ---------------------------------
model = load_model("final_imdb_lstm_modelll.keras")

with open("final_tokenizer.pkl", "rb") as f:
    tokenizer = pickle.load(f)

with open("final_tfidff.pkl", "rb") as f:
    tfidf = pickle.load(f)

feature_names = tfidf.get_feature_names_out()
MAX_LEN = 200

# ---------------------------------
# Text preprocessing (SAME AS TRAINING)
# ---------------------------------
stop_words = set(stopwords.words('english'))

important_negative_words = {
    'not', 'no', 'never', 'nor', 'none',
    'cannot', 'cant', 'wont', 'didnt', 'doesnt',
    'isnt', 'wasnt', 'werent', 'shouldnt',
    'wouldnt', 'couldnt',

    'bad', 'worse', 'worst', 'boring', 'awful', 'terrible',
    'horrible', 'poor', 'pathetic', 'disappointing',
    'annoying', 'dull', 'stupid', 'mess', 'ridiculous',
    'predictable', 'weak', 'cheap', 'lazy', 'ugly',
    'waste', 'wasted', 'wasting', 'overrated', 'underwhelming',
    'pointless', 'meaningless', 'unwatchable', 'forgettable',
    'nonsense', 'mediocre', 'painful'
}

stop_words = stop_words - important_negative_words


def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)

    words = text.split()
    words = [w for w in words if w not in stop_words]

    return " ".join(words)

# ---------------------------------
# Influential words (TF-IDF)
# ---------------------------------
generic_words = {'movie', 'film', 'story'}

def get_influential_words(text, top_n=6):
    processed = preprocess_text(text)
    vec = tfidf.transform([processed])
    scores = vec.toarray()[0]

    top_indices = np.argsort(scores)[-top_n:]
    words = [
        feature_names[i]
        for i in top_indices
        if scores[i] > 0 and feature_names[i] not in generic_words
    ]

    return words[::-1]

# ---------------------------------
# Prediction logic
# ---------------------------------
def predict_sentiment(text):
    processed = preprocess_text(text)

    seq = tokenizer.texts_to_sequences([processed])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post')

    prob = model.predict(pad)[0][0]

    sentiment = "Positive" if prob >= 0.5 else "Negative"
    confidence = prob if prob >= 0.5 else 1 - prob

    words = get_influential_words(text)

    return sentiment, confidence, words

# ---------------------------------
# Sidebar navigation
# ---------------------------------
page = st.sidebar.radio(
    "Navigation",
    ["About IMDB Sentiment Analysis", "Analyze Review"]
)

# =================================
# PAGE 1: EXPLANATION
# =================================
if page == "About IMDB Sentiment Analysis":
    st.title("IMDB Movie Review Sentiment Analysis")

    st.markdown("""
    ### Project Overview
    This application analyzes movie reviews and predicts whether the sentiment
    expressed is **positive** or **negative**.

    ### Dataset
    - IMDB Movie Reviews Dataset
    - 50,000 labeled reviews
    - Balanced positive and negative samples

    ### Model Used
    - LSTM (Long Short-Term Memory)
    - Word Embeddings
    - Binary classification using sigmoid activation

    ### Workflow
    1. Text preprocessing (cleaning and stopword removal)
    2. Tokenization and padding
    3. Sentiment prediction using LSTM
    4. Explanation using TF-IDF influential words

    ### Why LSTM?
    - Captures context
    - Understands word order
    - Performs well on sequential text data

    ### Output
    - Sentiment label (Positive / Negative)
    - Confidence score
    - Influential words contributing to the decision
    """)

# =================================
# PAGE 2: WORKING ANALYZER
# =================================
else:
    st.title("Movie Review Sentiment Analyzer")

    review = st.text_area(
        "Enter your movie review:",
        height=180,
        placeholder="Type your review here..."
    )

    if st.button("Analyze"):
        if review.strip() == "":
            st.warning("Please enter a review.")
        else:
            sentiment, confidence, words = predict_sentiment(review)

            st.subheader("Result")
            st.write(f"**Sentiment:** {sentiment}")
            st.write(f"**Confidence:** {confidence:.2f}")

            st.subheader("Influential Words")
            if words:
                for w in words:
                    st.write(f"- {w}")
            else:
                st.write("No influential words detected.")
