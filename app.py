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
    .block-container { padding-top: 1rem; }

    .hero-banner {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #3b82f6;
        border-radius: 20px;
        padding: 36px 40px;
        margin-bottom: 24px;
        text-align: center;
    }
    .hero-title {
        font-size: 3rem;
        font-weight: 900;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6, #ec4899);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: block;
        margin-bottom: 10px;
        letter-spacing: -1px;
    }
    .hero-sub {
        color: #94a3b8;
        font-size: 1.1rem;
        margin: 4px 0;
    }
    .section-heading {
        font-size: 1.6rem;
        font-weight: 700;
        background: linear-gradient(90deg, #3b82f6, #8b5cf6);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        display: block;
        margin-bottom: 6px;
    }
    .stat-box {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border-radius: 14px;
        padding: 22px;
        text-align: center;
        border: 1px solid #334155;
    }
    .stat-box h2 {
        font-size: 2.2rem;
        font-weight: 800;
        color: #3b82f6;
        margin: 0;
    }
    .stat-box p {
        color: #94a3b8;
        margin: 6px 0 0 0;
        font-size: 0.9rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #1e293b, #0f172a);
        border: 1px solid #334155;
        border-radius: 14px;
        padding: 28px;
        text-align: center;
        margin: 8px 0;
    }
    .metric-card h4 { color: #94a3b8; margin: 0 0 10px 0; font-size:1rem; }
    .metric-card h2 { margin: 0 0 6px 0; font-size: 1.9rem; }
    .metric-card p  { color: #64748b; margin: 0; font-size: 0.9rem; }
    .model-box {
        background: #1e293b;
        border-radius: 14px;
        padding: 22px;
        text-align: center;
        border: 1px solid #334155;
        height: 100%;
    }
    .model-box h3 { margin: 0 0 10px 0; }
    .model-box h2 { margin: 0 0 6px 0; font-size: 2.2rem; }
    .model-box p  { color: #94a3b8; margin: 4px 0; font-size: 0.88rem; }
    .tab-heading {
        font-size: 1.5rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 4px;
    }
    .tab-sub {
        color: #64748b;
        font-size: 0.95rem;
        margin-bottom: 20px;
    }
</style>
""", unsafe_allow_html=True)

# ── Load models ───────────────────────────────────────────────────────────────
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

EXAMPLES = [
    "Amazing flight, loved every moment! Best airline ever!",
    "Worst airline ever, never flying again. Total disaster!",
    "The flight was delayed by 3 hours with no explanation.",
    "Staff were so kind and helpful throughout the journey!",
]

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("## 🔍 BrandPulse AI")
    st.markdown("---")
    st.markdown("### 📊 About")
    st.info(
        "BrandPulse AI analyses tweet sentiment "
        "using Classical NLP models trained on "
        "100,000 real tweets from Sentiment140."
    )
    st.markdown("### 🛠️ Tech Stack")
    st.markdown("""
    - 🐍 Python 3.11
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
    | BiLSTM (CPU) | 76% |
    """)
    st.markdown("---")
    st.caption("Built by PRANAV-SN522")
    st.caption("NLP Internship Project 2026")

# ── Hero Banner ───────────────────────────────────────────────────────────────
st.markdown("""
<div class="hero-banner">
    <span class="hero-title">🔍 BrandPulse AI</span>
    <p class="hero-sub">Real-time Tweet Sentiment Analysis powered by Classical NLP</p>
    <p class="hero-sub">Trained on 100,000 tweets &nbsp;·&nbsp; Logistic Regression &nbsp;·&nbsp; Naïve Bayes &nbsp;·&nbsp; BiLSTM</p>
</div>
""", unsafe_allow_html=True)

# ── Stats Row ─────────────────────────────────────────────────────────────────
c1, c2, c3, c4 = st.columns(4)
with c1:
    st.markdown('<div class="stat-box"><h2>100K</h2><p>Tweets Trained On</p></div>',
                unsafe_allow_html=True)
with c2:
    st.markdown('<div class="stat-box"><h2>82%</h2><p>Best Accuracy</p></div>',
                unsafe_allow_html=True)
with c3:
    st.markdown('<div class="stat-box"><h2>2</h2><p>Live ML Models</p></div>',
                unsafe_allow_html=True)
with c4:
    st.markdown('<div class="stat-box"><h2>⚡</h2><p>Real-time Predictions</p></div>',
                unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)
st.divider()

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab1, tab2, tab3 = st.tabs([
    "💬  Live Predict",
    "📡  Simulate Feed",
    "📊  Model Comparison"
])

# ═══════════════════════════════════════════════════════
# TAB 1 — Live Predict
# ═══════════════════════════════════════════════════════
with tab1:
    st.markdown('<span class="section-heading">💬 Analyse Any Tweet Instantly</span>',
                unsafe_allow_html=True)
    st.markdown('<p class="tab-sub">Type a tweet or click an example below — '
                'get instant predictions from 2 ML models.</p>',
                unsafe_allow_html=True)

    # ── Example buttons — FIX: use session state ──────────────────────────────
    if "tweet_input" not in st.session_state:
        st.session_state.tweet_input = "The service was absolutely amazing! 10/10 would recommend."

    st.markdown("**💡 Click an example to try it:**")
    ex_cols = st.columns(4)
    ex_labels = ["😊 Amazing flight!", "😠 Worst ever!", "😐 Delayed again", "😊 Kind staff!"]
    for i, (col, ex, label) in enumerate(zip(ex_cols, EXAMPLES, ex_labels)):
        with col:
            if st.button(label, key=f"ex_{i}", use_container_width=True):
                st.session_state.tweet_input = ex

    st.markdown("")

    # ── Text input uses session state ─────────────────────────────────────────
    user_text = st.text_area(
        "✏️ Enter your tweet:",
        value=st.session_state.tweet_input,
        height=120,
        key="tweet_box"
    )

    analyse_btn = st.button(
        "🔍 Analyse Sentiment",
        type="primary",
        use_container_width=True
    )

    if analyse_btn and user_text.strip():
        with st.spinner("🧠 Analysing sentiment..."):
            results = predict(user_text)
            time.sleep(0.4)

        st.markdown("---")
        st.markdown('<span class="section-heading">🎯 Prediction Results</span>',
                    unsafe_allow_html=True)

        lr_color = "#22c55e" if results["lr"]["pred"] == 1 else "#ef4444"
        nb_color = "#22c55e" if results["nb"]["pred"] == 1 else "#ef4444"

        col1, col2 = st.columns(2)
        with col1:
            st.markdown(f"""
            <div class="metric-card" style="border-left:5px solid {lr_color}">
                <h4>🤖 Logistic Regression</h4>
                <h2 style="color:{lr_color}">{results['lr']['label']}</h2>
                <p>Confidence: <strong>{results['lr']['conf']:.1%}</strong></p>
            </div>
            """, unsafe_allow_html=True)
        with col2:
            st.markdown(f"""
            <div class="metric-card" style="border-left:5px solid {nb_color}">
                <h4>📊 Naïve Bayes</h4>
                <h2 style="color:{nb_color}">{results['nb']['label']}</h2>
                <p>Confidence: <strong>{results['nb']['conf']:.1%}</strong></p>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("")
        if results["lr"]["label"] == results["nb"]["label"]:
            st.success("✅ Both models agree on the sentiment!")
        else:
            st.warning("⚠️ Models disagree — this tweet may be sarcastic or ambiguous.")

        fig = go.Figure(go.Bar(
            x=["Logistic Regression", "Naïve Bayes"],
            y=[results["lr"]["conf"]*100, results["nb"]["conf"]*100],
            marker_color=[lr_color, nb_color],
            text=[f"{results['lr']['conf']:.1%}", f"{results['nb']['conf']:.1%}"],
            textposition="auto",
            width=0.4
        ))
        fig.update_layout(
            title="Model Confidence Comparison",
            yaxis_title="Confidence %",
            yaxis_range=[0, 100],
            plot_bgcolor="#1e293b",
            paper_bgcolor="#0f172a",
            font_color="#94a3b8",
            height=320,
            margin=dict(t=40, b=20)
        )
        st.plotly_chart(fig, use_container_width=True)

# ═══════════════════════════════════════════════════════
# TAB 2 — Simulate Feed
# ═══════════════════════════════════════════════════════
with tab2:
    st.markdown('<span class="section-heading">📡 Live Tweet Feed Simulator</span>',
                unsafe_allow_html=True)
    st.markdown('<p class="tab-sub">Watch the sentiment distribution update in real time '
                'as tweets stream in.</p>', unsafe_allow_html=True)

    col_a, col_b = st.columns([1, 2])
    with col_a:
        n_tweets = st.slider("📊 Number of tweets", 5, 30, 15)
        speed    = st.selectbox("⚡ Speed", ["Slow", "Normal", "Fast"])
        delay    = {"Slow": 0.8, "Normal": 0.35, "Fast": 0.1}[speed]
    with col_b:
        st.info(
            f"Will simulate **{n_tweets} tweets** at **{speed}** speed.\n\n"
            "The pie chart and table update live as each tweet is classified!"
        )

    if st.button("▶  Run Simulation", type="primary", use_container_width=True):
        history   = []
        chart_box = st.empty()
        prog      = st.progress(0)
        status    = st.empty()

        for i in range(n_tweets):
            tweet  = random.choice(SAMPLE_TWEETS)
            result = predict(tweet)
            label  = result["lr"]["label"]
            history.append({
                "Tweet #":    i + 1,
                "Tweet":      tweet[:55] + "..." if len(tweet) > 55 else tweet,
                "Sentiment":  label,
                "Confidence": f"{result['lr']['conf']:.0%}"
            })
            df_hist = pd.DataFrame(history)
            counts  = df_hist["Sentiment"].value_counts().reset_index()
            counts.columns = ["Sentiment", "Count"]
            colors  = {"😊 Positive": "#22c55e", "😠 Negative": "#ef4444"}

            fig = px.pie(
                counts, names="Sentiment", values="Count",
                title=f"Live Sentiment — {i+1}/{n_tweets} tweets processed",
                color="Sentiment", color_discrete_map=colors, hole=0.45
            )
            fig.update_layout(
                plot_bgcolor="#1e293b", paper_bgcolor="#0f172a",
                font_color="#94a3b8", height=380,
                legend=dict(orientation="h", yanchor="bottom", y=-0.2),
                margin=dict(t=50, b=60)
            )
            fig.update_traces(
                textposition="inside",
                textinfo="percent+label",
                textfont_size=14
            )

            with chart_box.container():
                ca, cb = st.columns([1, 1])
                with ca:
                    st.plotly_chart(fig, use_container_width=True)
                with cb:
                    st.dataframe(
                        df_hist[["Tweet #", "Tweet", "Sentiment", "Confidence"]],
                        use_container_width=True,
                        height=380
                    )

            prog.progress((i + 1) / n_tweets)
            status.markdown(f"⏳ Processing tweet **{i+1}** of **{n_tweets}**...")
            time.sleep(delay)

        status.empty()
        pos = sum(1 for h in history if h["Sentiment"] == "😊 Positive")
        neg = n_tweets - pos
        st.balloons()
        st.success(
            f"✅ Done! **{pos} Positive 😊** and **{neg} Negative 😠** "
            f"out of {n_tweets} tweets."
        )

# ═══════════════════════════════════════════════════════
# TAB 3 — Model Comparison
# ═══════════════════════════════════════════════════════
with tab3:
    st.markdown('<span class="section-heading">📊 Model Performance Comparison</span>',
                unsafe_allow_html=True)
    st.markdown('<p class="tab-sub">Classical NLP vs Deep Learning — '
                'trade-offs between speed, accuracy and interpretability.</p>',
                unsafe_allow_html=True)

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="model-box" style="border-top:4px solid #3b82f6">
            <h3 style="color:#3b82f6">🤖 Logistic Regression</h3>
            <h2 style="color:#22c55e">82%</h2>
            <p>Accuracy</p><br>
            <p>F1-Score: <strong>0.82</strong></p>
            <p>Train Time: <strong>&lt; 1 min</strong></p>
            <p>Interpretable: <strong>✅ Yes</strong></p>
            <p>Context-aware: <strong>❌ No</strong></p>
        </div>
        """, unsafe_allow_html=True)
    with col2:
        st.markdown("""
        <div class="model-box" style="border-top:4px solid #8b5cf6">
            <h3 style="color:#8b5cf6">📊 Naïve Bayes</h3>
            <h2 style="color:#22c55e">80%</h2>
            <p>Accuracy</p><br>
            <p>F1-Score: <strong>0.80</strong></p>
            <p>Train Time: <strong>&lt; 30 sec</strong></p>
            <p>Interpretable: <strong>✅ Yes</strong></p>
            <p>Context-aware: <strong>❌ No</strong></p>
        </div>
        """, unsafe_allow_html=True)
    with col3:
        st.markdown("""
        <div class="model-box" style="border-top:4px solid #f59e0b">
            <h3 style="color:#f59e0b">🧠 BiLSTM</h3>
            <h2 style="color:#f59e0b">76%</h2>
            <p>Accuracy</p><br>
            <p>F1-Score: <strong>0.76</strong></p>
            <p>Train Time: <strong>~8 min CPU</strong></p>
            <p>Interpretable: <strong>❌ No</strong></p>
            <p>Context-aware: <strong>✅ Yes</strong></p>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<span class="section-heading">📈 Accuracy Bar Chart</span>',
                unsafe_allow_html=True)

    fig = go.Figure(go.Bar(
        x=["Logistic Regression", "Naïve Bayes", "BiLSTM (CPU)"],
        y=[82, 80, 76],
        marker_color=["#3b82f6", "#8b5cf6", "#f59e0b"],
        text=["82%", "80%", "76%"],
        textposition="outside",
        width=0.5
    ))
    fig.update_layout(
        yaxis_range=[70, 90],
        yaxis_title="Accuracy %",
        plot_bgcolor="#1e293b",
        paper_bgcolor="#0f172a",
        font_color="#94a3b8",
        height=380,
        margin=dict(t=30, b=30)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown('<span class="section-heading">🔍 Full Comparison Table</span>',
                unsafe_allow_html=True)
    comp = pd.DataFrame({
        "Model":          ["Logistic Regression", "Naïve Bayes", "BiLSTM"],
        "Accuracy":       ["82%", "80%", "76%"],
        "F1-Score":       ["0.82", "0.80", "0.76"],
        "Training Time":  ["< 1 min", "< 30 sec", "~8 min CPU"],
        "Interpretable":  ["✅ Yes", "✅ Yes", "❌ No"],
        "Context-aware":  ["❌ No", "❌ No", "✅ Yes"],
        "Best for":       ["Production", "Quick baseline", "Large datasets"],
    })
    st.dataframe(comp, use_container_width=True, hide_index=True)

    st.markdown('<span class="section-heading">📸 Confusion Matrices</span>',
                unsafe_allow_html=True)
    ci1, ci2 = st.columns(2)
    with ci1:
        st.image("confusion_matrices_classical.png",
                 caption="Classical Models — Logistic Regression & Naïve Bayes",
                 use_column_width=True)
    with ci2:
        st.image("lstm_training_curves.png",
                 caption="BiLSTM Training Curves — Accuracy & Loss",
                 use_column_width=True)

    st.markdown('<span class="section-heading">💡 Key Insights</span>',
                unsafe_allow_html=True)
    st.info("🏆 **Winner: Logistic Regression** — 82% accuracy, fastest training, fully interpretable.")
    st.warning("🧠 **BiLSTM potential** — Needs 1M+ tweets and GPU to outperform classical models.")
    st.success("🚀 **Future upgrade** — Fine-tune BERTweet transformer to push accuracy above 90%.")
