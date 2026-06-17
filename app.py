import streamlit as st
import pickle, json, random, time, re
import numpy as np
import pandas as pd
import plotly.express as px
import nltk, spacy
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.preprocessing.text import tokenizer_from_json

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="BrandPulse AI",
    page_icon="🔍",
    layout="wide"
)

# ── Load models (cached so they load only once) ───────────────────────────────
@st.cache_resource
def load_all_models():
    lr    = pickle.load(open('lr_model.pkl',         'rb'))
    nb    = pickle.load(open('nb_model.pkl',         'rb'))
    tfidf = pickle.load(open('tfidf_vectorizer.pkl', 'rb'))
    lstm  = load_model('lstm_model.h5')
    with open('tokenizer.json') as f:
        tok = tokenizer_from_json(json.load(f))
    nlp  = spacy.load('en_core_web_sm', disable=['parser', 'ner'])
    nltk.download('stopwords', quiet=True)
    stops = set(nltk.corpus.stopwords.words('english'))
    return lr, nb, tfidf, lstm, tok, nlp, stops

lr_model, nb_model, tfidf_vec, lstm_model, tokenizer, nlp, stop_words = load_all_models()

# ── Cleaning function ─────────────────────────────────────────────────────────
def clean_tweet(text):
    if not isinstance(text, str):
        return ''
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'@\w+', '', text)
    text = re.sub(r'#(\w+)', r'\1', text)
    text = re.sub(r'[^a-z\s]', '', text)
    doc  = nlp(text)
    return ' '.join(t.lemma_ for t in doc
                    if t.text not in stop_words and not t.is_space)

# ── Prediction function ───────────────────────────────────────────────────────
def predict(raw_text):
    cleaned = clean_tweet(raw_text)
    vec     = tfidf_vec.transform([cleaned])
    labels  = {0: 'Negative', 1: 'Positive'}

    lr_pred   = int(lr_model.predict(vec)[0])
    lr_prob   = float(lr_model.predict_proba(vec)[0].max())

    nb_pred   = int(nb_model.predict(vec)[0])
    nb_prob   = float(nb_model.predict_proba(vec)[0].max())

    seq       = pad_sequences(tokenizer.texts_to_sequences([cleaned]),
                              maxlen=50, padding='post')
    lstm_prob = float(lstm_model.predict(seq, verbose=0)[0][0])
    lstm_pred = int(lstm_prob > 0.5)

    return {
        'lr':   {'label': labels[lr_pred],   'conf': lr_prob},
        'nb':   {'label': labels[nb_pred],   'conf': nb_prob},
        'lstm': {'label': labels[lstm_pred], 'conf': max(lstm_prob, 1 - lstm_prob)},
    }

# ── Sample tweets for simulation ─────────────────────────────────────────────
SAMPLE_TWEETS = [
    "Absolutely love this airline! Staff were incredibly kind.",
    "Delayed 3 hours with zero explanation. Completely unacceptable.",
    "Flight was okay, nothing special honestly.",
    "Best travel experience ever, will definitely fly again!",
    "Lost my luggage. This is the worst airline ever.",
    "On-time departure, comfy seats. Happy traveller here.",
    "The food was terrible and the seats were so uncomfortable.",
    "Thank you for the amazing service today! You made my day.",
    "Stuck on the tarmac for 2 hours. No updates from crew.",
    "Crew was super friendly and helpful throughout the flight!",
]

# ── Header ────────────────────────────────────────────────────────────────────
st.title("BrandPulse AI")
st.markdown("**Real-time Tweet Sentiment Analysis** — Classical NLP vs Deep Learning")
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["Live Predict", "Simulate Feed", "Model Comparison"])

# ═════════════════════════════════════════════════════════════════════════════
# TAB 1 — Live Prediction
# ═════════════════════════════════════════════════════════════════════════════
with tab1:
    st.subheader("Analyse any tweet instantly")
    user_text = st.text_area(
        "Enter a tweet:",
        "The service was absolutely amazing! 10/10 would recommend.",
        height=100
    )

    if st.button("Analyse Sentiment", type="primary"):
        with st.spinner("Analysing..."):
            results = predict(user_text)

        st.markdown("### Results")
        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric("Logistic Regression",
                      results['lr']['label'],
                      f"{results['lr']['conf']:.0%} confidence")
        with col2:
            st.metric("Naïve Bayes",
                      results['nb']['label'],
                      f"{results['nb']['conf']:.0%} confidence")
        with col3:
            st.metric("BiLSTM",
                      results['lstm']['label'],
                      f"{results['lstm']['conf']:.0%} confidence")

        # Agreement check
        preds = [results['lr']['label'],
                 results['nb']['label'],
                 results['lstm']['label']]
        if len(set(preds)) == 1:
            st.success("All 3 models agree!")
        else:
            st.warning("Models disagree — the tweet may be ambiguous.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 2 — Simulate Feed
# ═════════════════════════════════════════════════════════════════════════════
with tab2:
    st.subheader("Simulate an incoming tweet feed")
    n_tweets = st.slider("Number of tweets to simulate", 5, 30, 15)

    if st.button("▶ Run Simulation", type="primary"):
        history   = []
        tweet_box = st.empty()
        chart_box = st.empty()
        prog      = st.progress(0)

        for i in range(n_tweets):
            tweet  = random.choice(SAMPLE_TWEETS)
            result = predict(tweet)
            label  = result['lstm']['label']
            history.append({
                'Tweet #': i + 1,
                'Tweet':   tweet[:60] + '...' if len(tweet) > 60 else tweet,
                'Sentiment': label
            })

            df_hist = pd.DataFrame(history)
            counts  = df_hist['Sentiment'].value_counts().reset_index()
            counts.columns = ['Sentiment', 'Count']

            colors = {'Positive': '#22c55e', 'Negative': '#ef4444'}
            fig = px.pie(
                counts, names='Sentiment', values='Count',
                title=f'Sentiment distribution — {i+1} tweets processed',
                color='Sentiment',
                color_discrete_map=colors,
                hole=0.4
            )
            fig.update_layout(margin=dict(t=40, b=0, l=0, r=0))

            with chart_box.container():
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    st.plotly_chart(fig, use_container_width=True)
                with col_b:
                    st.dataframe(df_hist[['Tweet #','Tweet','Sentiment']],
                                 use_container_width=True, height=300)

            prog.progress((i + 1) / n_tweets)
            time.sleep(0.3)

        st.success(f"Simulation complete! Processed {n_tweets} tweets.")

# ═════════════════════════════════════════════════════════════════════════════
# TAB 3 — Model Comparison
# ═════════════════════════════════════════════════════════════════════════════
with tab3:
    st.subheader("Model performance comparison")

    comp = pd.DataFrame({
        'Model':            ['Logistic Regression', 'Naïve Bayes', 'BiLSTM'],
        'Accuracy':         ['~82%', '~80%', '~76%'],
        'F1-Score':         ['~0.82', '~0.80', '~0.76'],
        'Training Time':    ['< 1 min', '< 30 sec', '~8 min (CPU)'],
        'Interpretable':    ['Yes', 'Yes', 'No'],
        'Handles context':  ['No', 'No', 'Yes'],
    })
    st.dataframe(comp, use_container_width=True, hide_index=True)

    st.markdown("### Key takeaways")
    st.info("💡 **Logistic Regression** wins on this dataset — fast, interpretable, and accurate.")
    st.warning("**BiLSTM** needs more data and GPU training to show its real strength.")
    st.success("For production with millions of tweets, upgrade to a pretrained transformer like **BERTweet**.")

    st.markdown("### Confusion matrices")
    col1, col2 = st.columns(2)
    with col1:
        st.image('confusion_matrices_classical.png',
                 caption='Classical models', use_column_width=True)
    with col2:
        st.image('confusion_matrix_lstm.png',
                 caption='BiLSTM', use_column_width=True)