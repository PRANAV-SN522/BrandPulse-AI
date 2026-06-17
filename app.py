app_code = '''
import streamlit as st
import pickle, re, random, time
import pandas as pd
import plotly.express as px
import nltk, spacy

st.set_page_config(page_title="BrandPulse AI", page_icon="🔍", layout="wide")

@st.cache_resource
def load_models():
    lr    = pickle.load(open("lr_model.pkl",         "rb"))
    nb    = pickle.load(open("nb_model.pkl",         "rb"))
    tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    nlp   = spacy.load("en_core_web_sm", disable=["parser", "ner"])
    nltk.download("stopwords", quiet=True)
    stops = set(nltk.corpus.stopwords.words("english"))
    return lr, nb, tfidf, nlp, stops

lr_model, nb_model, tfidf_vec, nlp, stop_words = load_models()

def clean_tweet(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"http\\S+|www\\S+", "", text)
    text = re.sub(r"@\\w+", "", text)
    text = re.sub(r"#(\\w+)", r"\\1", text)
    text = re.sub(r"[^a-z\\s]", "", text)
    doc  = nlp(text)
    return " ".join(t.lemma_ for t in doc
                    if t.text not in stop_words and not t.is_space)

def predict(raw_text):
    cleaned = clean_tweet(raw_text)
    vec     = tfidf_vec.transform([cleaned])
    labels  = {0: "Negative", 1: "Positive"}
    lr_pred = int(lr_model.predict(vec)[0])
    lr_prob = float(lr_model.predict_proba(vec)[0].max())
    nb_pred = int(nb_model.predict(vec)[0])
    nb_prob = float(nb_model.predict_proba(vec)[0].max())
    return {
        "lr": {"label": labels[lr_pred], "conf": lr_prob},
        "nb": {"label": labels[nb_pred], "conf": nb_prob},
    }

SAMPLE_TWEETS = [
    "Absolutely love this airline! Staff were incredibly kind.",
    "Delayed 3 hours with zero explanation. Completely unacceptable.",
    "Flight was okay, nothing special honestly.",
    "Best travel experience ever, will definitely fly again!",
    "Lost my luggage. This is the worst airline ever.",
    "On-time departure, comfy seats. Happy traveller here.",
    "The food was terrible and the seats were so uncomfortable.",
    "Thank you for the amazing service today!",
    "Stuck on the tarmac for 2 hours. No updates from crew.",
    "Crew was super friendly and helpful throughout the flight!",
]

st.title("🔍 BrandPulse AI")
st.markdown("**Real-time Tweet Sentiment Analysis** — Classical NLP")
st.divider()

tab1, tab2, tab3 = st.tabs(["Live Predict", "Simulate Feed", "Model Comparison"])

with tab1:
    st.subheader("Analyse any tweet instantly")
    user_text = st.text_area("Enter a tweet:",
                             "The service was absolutely amazing! 10/10 would recommend.",
                             height=100)
    if st.button("Analyse Sentiment", type="primary"):
        with st.spinner("Analysing..."):
            results = predict(user_text)
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Logistic Regression",
                      results["lr"]["label"],
                      f"{results['lr']['conf']:.0%} confidence")
        with col2:
            st.metric("Naïve Bayes",
                      results["nb"]["label"],
                      f"{results['nb']['conf']:.0%} confidence")
        if results["lr"]["label"] == results["nb"]["label"]:
            st.success("Both models agree!")
        else:
            st.warning("Models disagree — tweet may be ambiguous.")

with tab2:
    st.subheader("Simulate an incoming tweet feed")
    n_tweets = st.slider("Number of tweets to simulate", 5, 30, 15)
    if st.button("▶ Run Simulation", type="primary"):
        history   = []
        chart_box = st.empty()
        prog      = st.progress(0)
        for i in range(n_tweets):
            tweet  = random.choice(SAMPLE_TWEETS)
            result = predict(tweet)
            label  = result["lr"]["label"]
            history.append({"Tweet #": i+1, "Tweet": tweet[:60], "Sentiment": label})
            df_hist = pd.DataFrame(history)
            counts  = df_hist["Sentiment"].value_counts().reset_index()
            counts.columns = ["Sentiment", "Count"]
            colors = {"Positive": "#22c55e", "Negative": "#ef4444"}
            fig = px.pie(counts, names="Sentiment", values="Count",
                         title=f"Sentiment distribution — {i+1} tweets",
                         color="Sentiment", color_discrete_map=colors, hole=0.4)
            with chart_box.container():
                col_a, col_b = st.columns([1, 1])
                with col_a:
                    st.plotly_chart(fig, use_container_width=True)
                with col_b:
                    st.dataframe(df_hist[["Tweet #","Tweet","Sentiment"]],
                                 use_container_width=True, height=300)
            prog.progress((i+1)/n_tweets)
            time.sleep(0.3)
        st.success(f"Simulation complete!")

with tab3:
    st.subheader("Model performance comparison")
    comp = pd.DataFrame({
        "Model":          ["Logistic Regression", "Naïve Bayes"],
        "Accuracy":       ["~82%", "~80%"],
        "F1-Score":       ["~0.82", "~0.80"],
        "Training Time":  ["< 1 min", "< 30 sec"],
        "Interpretable":  ["Yes", "Yes"],
    })
    st.dataframe(comp, use_container_width=True, hide_index=True)
    st.info("Logistic Regression wins — fast, interpretable, and accurate.")
    st.success("Next step: Fine-tune BERTweet transformer for 90%+ accuracy.")
    st.markdown("### Confusion Matrices")
    col1, col2 = st.columns(2)
    with col1:
        st.image("confusion_matrices_classical.png",
                 caption="Classical Models", use_column_width=True)
    with col2:
        st.image("lstm_training_curves.png",
                 caption="LSTM Training Curves", use_column_width=True)
'''

with open('app.py', 'w') as f:
    f.write(app_code)

print("New app.py created!")
