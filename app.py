import streamlit as st
import joblib
import pandas as pd
from datetime import datetime

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Fake News Detector",
    page_icon="📰",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL
# --------------------------------------------------

@st.cache_resource
def load_model():
    model = joblib.load("fake_news_model.pkl")
    vectorizer = joblib.load("tfidf_vectorizer.pkl")
    return model, vectorizer

model, vectorizer = load_model()

# --------------------------------------------------
# SESSION STATE
# --------------------------------------------------

if "history" not in st.session_state:
    st.session_state.history = []

# --------------------------------------------------
# CUSTOM CSS
# --------------------------------------------------

st.markdown("""
<style>

.stApp {
    background-color: #121212;
}

.main-title {
    text-align: center;
    font-size: 48px;
    font-weight: bold;
    color: white;
}

.result-card {
    text-align: center;
    padding: 25px;
    border-radius: 15px;
    font-size: 30px;
    font-weight: bold;
    margin-top: 20px;
}

.footer {
    text-align: center;
    color: gray;
    margin-top: 50px;
}

</style>
""", unsafe_allow_html=True)

# --------------------------------------------------
# HEADER
# --------------------------------------------------

st.markdown(
    "<div class='main-title'>📰 Fake News Detector</div>",
    unsafe_allow_html=True
)

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

with st.sidebar:

    st.header("📊 Dashboard")

    total = len(st.session_state.history)

    fake_count = len(
        [x for x in st.session_state.history
         if "FAKE" in x["Prediction"]]
    )

    true_count = len(
        [x for x in st.session_state.history
         if "TRUE" in x["Prediction"]]
    )

    st.metric("Total Predictions", total)
    st.metric("Fake News", fake_count)
    st.metric("True News", true_count)

    st.markdown("---")

# --------------------------------------------------
# INPUT SECTION
# --------------------------------------------------

st.markdown("### 📰 News Article Input")

news = st.text_area(
    "",
    height=140,
    placeholder="Paste news article here..."
)

st.caption(
    f"📝 Words: {len(news.split())} | Characters: {len(news)}"
)

# --------------------------------------------------
# BUTTON
# --------------------------------------------------

predict_btn = st.button(
    "🔍 Analyze News",
    use_container_width=True
)

# --------------------------------------------------
# PREDICTION
# --------------------------------------------------

if predict_btn:

    if news.strip():

        news_vector = vectorizer.transform([news])

        prediction = model.predict(news_vector)

        # Debug Information
        with st.expander("Debug Information"):

            st.write(
                "Raw Prediction:",
                prediction[0]
            )

            st.write(
                "Characters:",
                len(news)
            )

            st.write(
                "Words:",
                len(news.split())
            )

        try:

            confidence = (
                model.predict_proba(news_vector)
                .max() * 100
            )

        except:

            confidence = 0

        if prediction[0] == 0:

            result = "🚨 FAKE NEWS"

            st.markdown(
                f"""
                <div class='result-card'
                style='background:#4a1515;
                       color:#ff6b6b;'>
                {result}
                </div>
                """,
                unsafe_allow_html=True
            )

        else:

            result = "✅ TRUE NEWS"

            st.markdown(
                f"""
                <div class='result-card'
                style='background:#153d22;
                       color:#7dff9c;'>
                {result}
                </div>
                """,
                unsafe_allow_html=True
            )

        st.subheader("Confidence Score")

        st.progress(confidence / 100)

        st.write(
            f"### {confidence:.2f}%"
        )

        # Confidence Interpretation

        if confidence > 90:

            st.success(
                "High Confidence"
            )

        elif confidence > 70:

            st.warning(
                "Moderate Confidence"
            )

        else:

            st.error(
                "Low Confidence"
            )

        timestamp = datetime.now().strftime(
            "%d-%m-%Y %H:%M:%S"
        )

        st.session_state.history.append({

            "Time": timestamp,

            "Prediction": result,

            "Confidence":
            f"{confidence:.2f}%",

            "Preview":
            news[:80] + "..."
        })

    else:

        st.warning(
            "Please enter a news article."
        )

# --------------------------------------------------
# HISTORY
# --------------------------------------------------

st.markdown("---")

st.subheader("Prediction History")

if st.session_state.history:

    history_df = pd.DataFrame(
        st.session_state.history[::-1]
    )

    st.dataframe(
        history_df,
        use_container_width=True
    )

    csv = history_df.to_csv(
        index=False
    ).encode("utf-8")

    st.download_button(
        label="📥 Download History",
        data=csv,
        file_name="prediction_history.csv",
        mime="text/csv"
    )

else:

    st.info(
        "No predictions yet."
    )

# --------------------------------------------------
# FOOTER
# --------------------------------------------------

st.markdown("---")

st.markdown(
    """
    <div class='footer'>
    Built with Streamlit, Scikit-Learn & Python
    </div>
    """,
    unsafe_allow_html=True
)