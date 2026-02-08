import streamlit as st
import pickle
import re
import numpy as np
import nltk

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords

# ---------------------------------
# Download stopwords safely
# ---------------------------------
@st.cache_resource
def load_stopwords():
    nltk.download("stopwords")
    return set(stopwords.words("english"))

stop_words = load_stopwords()

# ---------------------------------
# Page configuration
# ---------------------------------
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------------------------
# Dark + Light Mode CSS
# ---------------------------------
st.markdown("""
<style>
:root {
    --bg-main: #f8f9fa;
    --bg-card: #ffffff;
    --text-main: #2d3748;
    --text-secondary: #4a5568;
    --accent: #667eea;
    --accent-2: #764ba2;
}

@media (prefers-color-scheme: dark) {
    :root {
        --bg-main: #121212;
        --bg-card: #1e1e1e;
        --text-main: #e5e7eb;
        --text-secondary: #cbd5e1;
        --accent: #8b9cff;
        --accent-2: #a78bfa;
    }
}

.stApp {
    background-color: var(--bg-main);
}

.info-box {
    background: var(--bg-card);
    padding: 20px;
    border-radius: 12px;
    border-left: 5px solid var(--accent);
    margin: 15px 0;
    color: var(--text-main);
}

h1, h2, h3, h4 {
    color: var(--text-main);
}

.stButton>button {
    width: 100%;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
    color: white;
    font-size: 18px;
    font-weight: bold;
    padding: 14px;
    border-radius: 10px;
    border: none;
}

.word-badge {
    display: inline-block;
    padding: 10px 20px;
    margin: 6px;
    background: linear-gradient(135deg, var(--accent) 0%, var(--accent-2) 100%);
    color: white;
    border-radius: 25px;
    font-weight: 600;
}

.sentiment-display {
    font-size: 42px;
    font-weight: bold;
    text-align: center;
    padding: 20px;
    margin: 20px 0;
    border-radius: 15px;
    color: white;
}

.positive {
    background: linear-gradient(135deg, #11998e, #38ef7d);
}

.negative {
    background: linear-gradient(135deg, #ee0979, #ff6a00);
}

#MainMenu {visibility: hidden;}
footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# ---------------------------------
# Load model and tokenizer
# ---------------------------------
@st.cache_resource
def load_models():
    model = load_model("final_imdb_lstm_modelll.keras")

    with open("final_tokenizer.pkl", "rb") as f:
        tokenizer = pickle.load(f)

    with open("final_tfidff.pkl", "rb") as f:
        tfidf = pickle.load(f)

    return model, tokenizer, tfidf

model, tokenizer, tfidf = load_models()
feature_names = tfidf.get_feature_names_out()
MAX_LEN = 200

# ---------------------------------
# Text preprocessing
# ---------------------------------
important_negative_words = {
    'not','no','never','bad','worst','boring','awful','terrible','poor'
}

stop_words = stop_words - important_negative_words

def preprocess_text(text):
    text = text.lower()
    text = re.sub(r'<.*?>', '', text)
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    words = [w for w in text.split() if w not in stop_words]
    return " ".join(words)

# ---------------------------------
# Influential words
# ---------------------------------
def get_influential_words(text, top_n=6):
    processed = preprocess_text(text)
    vec = tfidf.transform([processed])
    scores = vec.toarray()[0]
    top_indices = np.argsort(scores)[-top_n:]
    return [feature_names[i] for i in top_indices if scores[i] > 0][::-1]

# ---------------------------------
# Prediction
# ---------------------------------
def predict_sentiment(text):
    processed = preprocess_text(text)
    seq = tokenizer.texts_to_sequences([processed])
    pad = pad_sequences(seq, maxlen=MAX_LEN, padding='post')
    prob = model.predict(pad, verbose=0)[0][0]

    prob = float(prob)  # FIX
    prob = max(0.0, min(1.0, prob))  # safety clamp

    sentiment = "Positive" if prob >= 0.5 else "Negative"
    confidence = prob if prob >= 0.5 else 1 - prob
    words = get_influential_words(text)

    return sentiment, confidence, words

# ---------------------------------
# Sidebar
# ---------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Choose", ["About", "Analyze Review"])

# ---------------------------------
# ABOUT PAGE
# ---------------------------------
if page == "About":
    st.title("IMDB Sentiment Analysis By Deepali")

    st.markdown("""
    <div class="info-box">
    This app analyzes movie reviews and predicts whether the sentiment is positive or negative using an LSTM model.
    </div>
    """, unsafe_allow_html=True)

# ---------------------------------
# ANALYZER PAGE
# ---------------------------------
else:
    st.title("Analyze Movie Review")

    review = st.text_area("Enter your review", height=180)

    if st.button("Analyze Sentiment"):
        if review.strip() == "":
            st.warning("Please enter a review.")
        else:
            sentiment, confidence, words = predict_sentiment(review)

            css_class = "positive" if sentiment == "Positive" else "negative"

            st.markdown(f"""
            <div class="sentiment-display {css_class}">
                {sentiment}
            </div>
            """, unsafe_allow_html=True)

            st.progress(float(confidence), text=f"{confidence*100:.1f}% confidence")

            if words:
                badges = "".join(
                    [f"<span class='word-badge'>{w}</span>" for w in words]
                )
                st.markdown(badges, unsafe_allow_html=True)
