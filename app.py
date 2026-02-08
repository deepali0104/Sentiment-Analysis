import streamlit as st
import pickle
import re
import numpy as np
import nltk

from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from nltk.corpus import stopwords


nltk.download('stopwords')
# ---------------------------------
# Page configuration
# ---------------------------------
st.set_page_config(
    page_title="IMDB Sentiment Analysis",
    layout="centered",
    initial_sidebar_state="expanded"
)

# ---------------------------------
# Custom CSS for better styling
# ---------------------------------
st.markdown("""
<style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }

    /* Card-like containers */
    .stApp {
        background-color: #f8f9fa;
    }

    /* Custom result cards */
    .result-card {
        padding: 30px;
        border-radius: 15px;
        margin: 20px 0;
        box-shadow: 0 8px 16px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }

    .result-card:hover {
        transform: translateY(-5px);
        box-shadow: 0 12px 24px rgba(0,0,0,0.15);
    }

    .positive-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
    }

    .negative-card {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }

    /* Sentiment display */
    .sentiment-display {
        font-size: 48px;
        font-weight: bold;
        text-align: center;
        padding: 20px;
        margin: 20px 0;
        border-radius: 15px;
        text-transform: uppercase;
        letter-spacing: 3px;
    }

    .positive-sentiment {
        background: linear-gradient(135deg, #11998e 0%, #38ef7d 100%);
        color: white;
        box-shadow: 0 8px 20px rgba(56, 239, 125, 0.3);
    }

    .negative-sentiment {
        background: linear-gradient(135deg, #ee0979 0%, #ff6a00 100%);
        color: white;
        box-shadow: 0 8px 20px rgba(238, 9, 121, 0.3);
    }

    /* Confidence meter styling */
    .confidence-container {
        background: white;
        padding: 25px;
        border-radius: 12px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.08);
        margin: 20px 0;
    }

    .confidence-label {
        font-size: 18px;
        font-weight: 600;
        color: #333;
        margin-bottom: 10px;
    }

    /* Influential words styling */
    .word-badge {
        display: inline-block;
        padding: 10px 20px;
        margin: 8px;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border-radius: 25px;
        font-weight: 600;
        font-size: 16px;
        box-shadow: 0 4px 8px rgba(102, 126, 234, 0.3);
        transition: all 0.3s ease;
    }

    .word-badge:hover {
        transform: scale(1.1);
        box-shadow: 0 6px 12px rgba(102, 126, 234, 0.4);
    }

    /* Button styling */
    .stButton>button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-size: 18px;
        font-weight: bold;
        padding: 15px;
        border-radius: 10px;
        border: none;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 16px rgba(102, 126, 234, 0.5);
    }

    /* Text area styling */
    .stTextArea>div>div>textarea {
        border-radius: 12px;
        border: 2px solid #667eea;
        font-size: 16px;
        padding: 15px;
    }

    /* Info boxes */
    .info-box {
        background: white;
        padding: 20px;
        border-radius: 12px;
        border-left: 5px solid #667eea;
        margin: 15px 0;
        box-shadow: 0 2px 8px rgba(0,0,0,0.05);
    }

    /* Header styling */
    h1 {
        color: #2d3748;
        font-weight: 800;
    }

    h3 {
        color: #4a5568;
        font-weight: 700;
    }

    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }

    /* Hide streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
</style>
""", unsafe_allow_html=True)


# ---------------------------------
# Load model and objects
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
    prob = model.predict(pad, verbose=0)[0][0]
    sentiment = "Positive" if prob >= 0.5 else "Negative"
    confidence = prob if prob >= 0.5 else 1 - prob
    words = get_influential_words(text)
    return sentiment, confidence, words


# ---------------------------------
# Sidebar navigation
# ---------------------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio(
    "Choose a section:",
    ["About IMDB Sentiment Analysis", "Analyze Review"],
    label_visibility="collapsed"
)

st.sidebar.markdown("---")
st.sidebar.markdown("""
<div style='text-align: center; color: black; padding: 20px;'>
    <h4>Quick Guide</h4>
    <p>Enter any movie review and our AI will analyze the sentiment instantly.</p>
</div>
""", unsafe_allow_html=True)

# =================================
# PAGE 1: EXPLANATION
# =================================
if page == "About IMDB Sentiment Analysis":
    st.title("IMDB Movie Review Sentiment Analysis")

    st.markdown("""
    <div class='info-box'>
        <h3>Welcome to the Sentiment Analyzer</h3>
        <p>This application uses advanced machine learning to understand the emotional tone of movie reviews. 
        Whether a review is raving or ranting, our model can detect it with high accuracy.</p>
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("""
        <div class='info-box'>
            <h4>Dataset Highlights</h4>
            <ul>
                <li><strong>50,000</strong> labeled reviews</li>
                <li>Balanced positive and negative samples</li>
                <li>Real IMDB movie reviews</li>
                <li>Diverse vocabulary and expressions</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    with col2:
        st.markdown("""
        <div class='info-box'>
            <h4>Model Architecture</h4>
            <ul>
                <li><strong>LSTM</strong> neural network</li>
                <li>Word embeddings for context</li>
                <li>Sigmoid activation for classification</li>
                <li>Optimized for accuracy</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### How It Works")

    step1, step2, step3, step4 = st.columns(4)

    with step1:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: white; border-radius: 10px; margin: 10px 0;'>
            <h2 style='color: #667eea;'>1</h2>
            <p><strong>Clean Text</strong></p>
            <p style='font-size: 14px;'>Remove noise and stopwords</p>
        </div>
        """, unsafe_allow_html=True)

    with step2:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: white; border-radius: 10px; margin: 10px 0;'>
            <h2 style='color: #667eea;'>2</h2>
            <p><strong>Tokenize</strong></p>
            <p style='font-size: 14px;'>Convert words to numbers</p>
        </div>
        """, unsafe_allow_html=True)

    with step3:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: white; border-radius: 10px; margin: 10px 0;'>
            <h2 style='color: #667eea;'>3</h2>
            <p><strong>LSTM Magic</strong></p>
            <p style='font-size: 14px;'>Neural network analysis</p>
        </div>
        """, unsafe_allow_html=True)

    with step4:
        st.markdown("""
        <div style='text-align: center; padding: 20px; background: white; border-radius: 10px; margin: 10px 0;'>
            <h2 style='color: #667eea;'>4</h2>
            <p><strong>Predict</strong></p>
            <p style='font-size: 14px;'>Get sentiment and confidence</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
        <h3>Why LSTM?</h3>
        <p>Long Short-Term Memory networks excel at understanding sequential data like text. 
        Unlike simple models, LSTMs can:</p>
        <ul>
            <li><strong>Remember context</strong> from earlier in the review</li>
            <li><strong>Understand word order</strong> and how it affects meaning</li>
            <li><strong>Capture nuance</strong> in language and sentiment</li>
            <li><strong>Handle long reviews</strong> without losing important information</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='info-box'>
        <h3>What You'll Get</h3>
        <ul>
            <li><strong>Sentiment Label:</strong> Clear positive or negative classification</li>
            <li><strong>Confidence Score:</strong> How certain the model is about its prediction</li>
            <li><strong>Influential Words:</strong> Key words that drove the decision</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

# =================================
# PAGE 2: WORKING ANALYZER
# =================================
else:
    st.title("Movie Review Sentiment Analyzer")

    st.markdown("""
    <div class='info-box'>
        <p style='margin: 0; font-size: 16px;'>
            Paste your movie review below and discover what the AI thinks about it. 
            The more detailed your review, the better the analysis.
        </p>
    </div>
    """, unsafe_allow_html=True)

    review = st.text_area(
        "Enter your movie review:",
        height=180,
        placeholder="Example: This movie was absolutely incredible! The acting was superb and the plot kept me on the edge of my seat...",
        label_visibility="collapsed"
    )

    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        analyze_button = st.button("Analyze Sentiment", use_container_width=True)

    if analyze_button:
        if review.strip() == "":
            st.warning("Please enter a review to analyze.")
        else:
            with st.spinner("Analyzing your review..."):
                sentiment, confidence, words = predict_sentiment(review)

            # Sentiment display with styled card
            sentiment_class = "positive-sentiment" if sentiment == "Positive" else "negative-sentiment"
            st.markdown(f"""
            <div class='sentiment-display {sentiment_class}'>
                {sentiment}
            </div>
            """, unsafe_allow_html=True)

            # Confidence meter
            st.markdown("""
            <div class='confidence-container'>
                <div class='confidence-label'>Confidence Level</div>
            </div>
            """, unsafe_allow_html=True)

            confidence_percent = float(confidence) * 100
            confidence_value = float(confidence)

            # Color-coded progress bar
            if confidence_value >= 0.8:
                bar_color = "#38ef7d"
            elif confidence_value >= 0.6:
                bar_color = "#667eea"
            else:
                bar_color = "#ff6a00"

            st.progress(confidence_value, text=f"{confidence_percent:.1f}% confident")

            # Visual interpretation
            if confidence_value >= 0.9:
                interpretation = "Extremely confident"
            elif confidence_value >= 0.75:
                interpretation = "Very confident"
            elif confidence_value >= 0.6:
                interpretation = "Moderately confident"
            else:
                interpretation = "Somewhat uncertain"

            st.markdown(f"""
            <div style='text-align: center; padding: 10px; color: #666; font-style: italic;'>
                {interpretation}
            </div>
            """, unsafe_allow_html=True)

            # Influential words section
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("""
            <div class='confidence-container'>
                <div class='confidence-label'>Key Words Driving This Prediction</div>
                <p style='color: #666; font-size: 14px; margin-top: 5px;'>
                    These words had the strongest influence on the sentiment analysis
                </p>
            </div>
            """, unsafe_allow_html=True)

            if words:
                # Display words as styled badges
                words_html = "".join([f"<span class='word-badge'>{w}</span>" for w in words])
                st.markdown(f"""
                <div style='text-align: center; padding: 20px;'>
                    {words_html}
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown("""
                <div style='text-align: center; padding: 20px; color: #999;'>
                    No highly influential words detected
                </div>
                """, unsafe_allow_html=True)

            # Add some space and a retry option
            st.markdown("<br>", unsafe_allow_html=True)
            col1, col2, col3 = st.columns([1, 2, 1])
            with col2:
                if st.button("Analyze Another Review", use_container_width=True):
                    st.rerun()
