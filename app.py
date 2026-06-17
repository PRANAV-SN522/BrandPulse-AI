import streamlit as st
import pickle, re, random, time
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import nltk

nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)
nltk.download("omw-1.4", quiet=True)
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

st.set_page_config(
    page_title="BrandPulse AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f172a; }
    .block-container { padding-top: 2rem; }
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 20px;
        text-align: center;
        margin: 8px 0;
    }
    .positive { border-left: 4px solid #22c55e !important; }
    .negative { border-left: 4px solid #ef4444 !important; }
    .hero-title {
        font-size: 3rem;
        font-weight: 800;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
    }
    .hero-sub {
        text-align: center;
        color: #94a3b8;
        font-size: 1.1rem;
        margin-bottom: 2rem;
    }
    .stat-box {
        background: #1e293b;
        border-radius: 10px;
        padding: 15px;
        text-align: center;
        border: 1px solid #334155;
    }
</style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_models():
    lr    = pickle.load(open("lr_model.pkl",         "rb"))
    nb    = pickle.load(open("nb_model.pkl",         "rb"))
    tfidf = pickle.load(open("tfidf_vectorizer.pkl", "rb"))
    return lr, nb, tfidf

lr_model, nb_model, tfidf_vec = load_models()
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words("english"))

def clean_tweet(text):
    if not isinstance(text, str): return ""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#(\w+)", r"\1", text)
    text = re.sub(r"[^a-z\s]", "", text)
    tokens = [lemmatizer.lemmatize(t) for t in text.split()
              if t not in stop_words and len(t) > 1]
    return " ".join(tokens)

def predict(raw_text):
    cleaned = clean_tweet(raw_text)
    vec     = tfidf_vec.transform([cleaned])
    labels  = {0: "😠 Negative", 1: "😊 Positive"}
    lr_pred = int(lr_model.predict(vec)[0])
    lr_prob = float(lr_model.predict_proba(vec)[0].max())
    nb_pred = int(nb_model.predict(vec)[0])
    nb_prob = float(nb_model.predict_proba(vec)[0].max())
    return {
        "lr": {"label": labels[lr_pred], "conf": lr_prob, "pred": lr_pred},
        "nb": {"label": labels[nb_pred], "conf": nb_prob, "pred": nb_pred},
    }

SAMPLE_TWEETS = [
    "Absolutely love this airline! Staff were incredibly kind.",
    "Delayed 3 hours with zero explanation. Completely unacceptable.",
    "Flight was okay, nothing special honestly.",
    "Best travel experience ever, will definitely fly again!",
    "Lost my luggage. This is the worst airline ever.",
    "On-time departure, comfy seats. Happy traveller here.",
    "The food was terrible and seats were so uncomfortable.",
    "Thank you for the amazing service today!",
    "Stuck on the tarmac for 2 hours. No updates from crew.",
    "Crew was super friendly and helpful throughout the flight!",
    "Never flying this airline again. Total disaster.",
    "Smooth landing and excellent in-flight entertainment!",
]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/search.png", width=60)
    st.title("BrandPulse AI")
    st.markdown("---")
    st.markdown("### 📊 About")
    st.info(
        "BrandPulse AI analyses tweet sentiment using "
        "Classical NLP models trained on 100,000 real tweets."
    )
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("""
    - 🐍 Python
    - 📊 Scikit-learn
    - 🌊 Streamlit
    - 📈 Plotly
    - 📝 NLTK
    """)
    st.markdown("### 📈 Model Performance")
    st.markdown("""
    | Model | Accuracy |
    |-------|----------|
    | Logistic Regression | 82% |
    | Naïve Bayes | 80% |
    """)
    st.markdown("---")
    st.caption("Built by PRANAV-SN522 • NLP Internship Project")

# ── Hero Header ───────────────────────────────────────────────────────────────
st.markdown('<p class="hero-title">🔍 BrandPulse AI</p>', unsafe_allow_html=True)
st.markdown(
    '<p class="hero-sub">Real-time Tweet Sentiment Analysis powered by Classical NLP</p>',
    unsafe_allow_html=True)

# ── Stats Row ─────────────────────────────────────────────────────────────────
col1, col2, col3, col4 = st.columns(4)
with col1:
    st.markdown('<div class="stat-box"><h2>100K</h2><p>Tweets Trained</p></div>',
                unsafe_allow_html=True)
with col2:
    st.markdown('<div class="stat-box"><h2>82%</h2><p>Best Accuracy</p></div>',
                unsafe_allow_html=True)
with col3:
    st.markdown('<div class="stat-box"><h2>2</h2><p>ML Models</p></div>',
                unsafe_allow_html=True)
with col4:
    st.markdown('<div class="stat-box"><h2>Live</h2><p>Predictions</p></div>',
                unsafe_allow_html=True)

st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs(["💬 Live Predict", "📡 Simulate Feed", "📊 Model Comparison"])

# ═══════════════════════════════════════════════════════
# TAB 1
# ═══════════════════════════════════════════════════════
with tab1:
    st.subheader("💬 Analyse any tweet instantly")
    st.markdown("Type or paste any tweet below and get instant sentiment predictions.")

    col_input, col_examples = st.columns([2, 1])
    with col_input:
        user_text = st.text_area(
            "Enter a tweet:",
            "The service was absolutely amazing! 10/10 would recommend.",
            height=120)
        analyse_btn = st.button("🔍 Analyse Sentiment", type="primary", use_container_width=True)

    with col_examples:
        st.markdown("**💡 Try these examples:**")
        examples = [
            "Amazing flight experience!",
            "Worst airline ever, avoid!",
            "The flight was delayed again",
            "Staff were so helpful today",
        ]
        for ex in examples:
            if st.button(ex, use_container_width=True):
                user_text = ex

    if analyse_btn:
        with st.spinner("🧠 Analysing sentiment..."):
            results = predict(user_text)
            time.sleep(0.5)

        st.markdown("### 🎯 Results")
        col1, col2 = st.columns(2)

        lr_color = "#22c55e" if results["lr"]["pred"] == 1 else "#ef4444"
        nb_color = "#22c55e" if results["nb"]["pred"] == 1 else "#ef4444"

        with col1:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid {lr_color}">
                <h4 style="color:#94a3b8">Logistic Regression</h4>
                <h2 style="color:{lr_color}">{results['lr']['label']}</h2>
                <p style="color:#64748b">Confidence: {results['lr']['conf']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card" style="border-left: 4px solid {nb_color}">
                <h4 style="color:#94a3b8">Naïve Bayes</h4>
                <h2 style="color:{nb_color}">{results['nb']['label']}</h2>
                <p style="color:#64748b">Confidence: {results['nb']['conf']:.1%}</p>
            </div>
            """, unsafe_allow_html=True)

        if results["lr"]["label"] == results["nb"]["label"]:
            st.success("✅ Both models agree on the sentiment!")
        else:
            st.warning("⚠️ Models disagree — this tweet may be ambiguous or sarcastic.")

        # Confidence bar chart
        fig = go.Figure(go.Bar(
            x=["Logistic Regression", "Naïve Bayes"],
            y=[results["lr"]["conf"] * 100, results["nb"]["conf"] * 100],
            marker_color=[lr_color, nb_color],
            text=[f"{results['lr']['conf']:.1%}", f"{results['nb']['conf']:.1%}"],
            textposition="auto"
        ))
        fig.update_layout(
            title="Model Confidence Comparison",
            yaxis_title="Confidence %",
            yaxis_range=[0, 100],
            plot_bgcolor="#1e293b",
            paper_bgcolor="#0f172a",
            font_color="#94a3b8",
            height=300
        )
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB 2
# ═══════════════════════════════════════════════════════
with tab2:
    st.subheader("📡 Simulate a live tweet feed")
    st.markdown("Simulate incoming tweets and watch the sentiment distribution update in real time.")

    col_a, col_b = st.columns([1, 2])
    with col_a:
        n_tweets = st.slider("Number of tweets", 5, 30, 15)
        speed    = st.selectbox("Speed", ["Slow", "Normal", "Fast"])
        delay    = {"Slow": 0.8, "Normal": 0.3, "Fast": 0.1}[speed]

    with col_b:
        st.info(f"Will simulate **{n_tweets} tweets** at **{speed}** speed. Click Run to start!")

    if st.button("▶ Run Simulation", type="primary"):
        history   = []
        chart_box = st.empty()
        prog      = st.progress(0)
        status    = st.empty()

        for i in range(n_tweets):
            tweet  = random.choice(SAMPLE_TWEETS)
            result = predict(tweet)
            label  = result["lr"]["label"]
            history.append({
                "Tweet #":   i + 1,
                "Tweet":     tweet[:55] + "..." if len(tweet) > 55 else tweet,
                "Sentiment": label,
                "Confidence": f"{result['lr']['conf']:.0%}"
            })
            df_hist = pd.DataFrame(history)
            counts  = df_hist["Sentiment"].value_counts().reset_index()
            counts.columns = ["Sentiment", "Count"]
            colors  = {"😊 Positive": "#22c55e", "😠 Negative": "#ef4444"}

            fig = px.pie(counts, names="Sentiment", values="Count",
                         title=f"Live Sentiment — {i+1}/{n_tweets} tweets",
                         color="Sentiment", color_discrete_map=colors, hole=0.45)
            fig.update_layout(
                plot_bgcolor="#1e293b", paper_bgcolor="#0f172a",
                font_color="#94a3b8", height=350,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2)
            )
            fig.update_traces(textposition="inside", textinfo="percent+label")

            with chart_box.container():
                c1, c2 = st.columns([1, 1])
                with c1:
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    st.dataframe(
                        df_hist[["Tweet #", "Tweet", "Sentiment", "Confidence"]],
                        use_container_width=True, height=350)

            prog.progress((i + 1) / n_tweets)
            status.markdown(f"Processing tweet **{i+1}** of **{n_tweets}**...")
            time.sleep(delay)

        status.empty()
        pos = sum(1 for h in history if h["Sentiment"] == "😊 Positive")
        neg = n_tweets - pos
        st.success(f"✅ Done! **{pos} Positive** and **{neg} Negative** out of {n_tweets} tweets.")

# ═══════════════════════════════════════════════════════
# TAB 3
# ═══════════════════════════════════════════════════════
with tab3:
    st.subheader("📊 Model Performance Comparison")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="stat-box">
            <h3 style="color:#3b82f6">Logistic Regression</h3>
            <h2 style="color:#22c55e">82%</h2>
            <p>Accuracy</p>
            <p style="color:#94a3b8">F1: 0.82 | Time: &lt;1 min</p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="stat-box">
            <h3 style="color:#8b5cf6">Naïve Bayes</h3>
            <h2 style="color:#22c55e">80%</h2>
            <p>Accuracy</p>
            <p style="color:#94a3b8">F1: 0.80 | Time: &lt;30 sec</p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="stat-box">
            <h3 style="color:#f59e0b">BiLSTM</h3>
            <h2 style="color:#f59e0b">76%</h2>
            <p>Accuracy</p>
            <p style="color:#94a3b8">F1: 0.76 | Time: ~8 min</p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("### 📈 Accuracy Comparison")
    fig = go.Figure(go.Bar(
        x=["Logistic Regression", "Naïve Bayes", "BiLSTM (CPU)"],
        y=[82, 80, 76],
        marker_color=["#3b82f6", "#8b5cf6", "#f59e0b"],
        text=["82%", "80%", "76%"],
        textposition="auto"
    ))
    fig.update_layout(
        yaxis_range=[70, 90],
        yaxis_title="Accuracy %",
        plot_bgcolor="#1e293b",
        paper_bgcolor="#0f172a",
        font_color="#94a3b8",
        height=350
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("### 🔍 Detailed Comparison")
    comp = pd.DataFrame({
        "Model":         ["Logistic Regression", "Naïve Bayes", "BiLSTM"],
        "Accuracy":      ["82%", "80%", "76%"],
        "F1-Score":      ["0.82", "0.80", "0.76"],
        "Train Time":    ["< 1 min", "< 30 sec", "~8 min CPU"],
        "Interpretable": ["✅ Yes", "✅ Yes", "❌ No"],
        "Context":       ["❌ No", "❌ No", "✅ Yes"],
    })
    st.dataframe(comp, use_container_width=True, hide_index=True)

    st.markdown("### 📸 Confusion Matrices")
    col1, col2 = st.columns(2)
    with col1:
        st.image("confusion_matrices_classical.png",
                 caption="Classical Models — Logistic Regression & Naïve Bayes",
                 use_column_width=True)
    with col2:
        st.image("lstm_training_curves.png",
                 caption="BiLSTM Training Curves",
                 use_column_width=True)

    st.markdown("### 💡 Key Insights")
    st.info("🏆 **Logistic Regression wins** on 100k tweets — fast, interpretable, and 82% accurate.")
    st.warning("🧠 **BiLSTM needs more data** — it would improve significantly with 1M+ tweets and GPU training.")
    st.success("🚀 **Next step** — Fine-tune BERTweet (transformer pretrained on tweets) to push accuracy above 90%.")

