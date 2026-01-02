import sys, os, json, time
import pandas as pd
import streamlit as st

try:
    from kafka import KafkaConsumer
    KAFKA_AVAILABLE = True
except:
    KAFKA_AVAILABLE = False

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.append(ROOT)

from backend.predictor import SurvivalPredictor

st.set_page_config(
    page_title="Dashboard | Titanic Survival",
    layout="wide"
)

st.markdown("""
<style>
.stApp {
    background: linear-gradient(135deg, #020617, #0f172a, #020617);
    color: white;
}
.card {
    background: rgba(255,255,255,0.06);
    border-radius: 16px;
    padding: 20px;
    box-shadow: 0 0 30px rgba(0,120,255,0.25);
}
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card">
<h1>Titanic Survival Prediction</h1>
</div>
""", unsafe_allow_html=True)

st.markdown("""
<div class="card">
<h2>Dashboard & Prediction</h2>
</div>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return SurvivalPredictor("models/survival_model.pkl")

@st.cache_data
def load_data():
    return pd.read_csv("data/processed/titanic_cleaned.csv")

predictor = load_model()
df = load_data()

# KPI SECTION
probs_all, preds_all, risks_all = predictor.predict(df)

k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Passengers", len(df))
k2.metric("Avg Survival", f"{probs_all.mean():.2%}")
k3.metric("High Risk (>60)", int((risks_all > 60).sum()))
k4.metric("Model ROC-AUC", "0.91")

# FILTERING & SORTING (MAIN OBJECTIVE)
st.markdown("##  Filter & Sort Predictions")

c1, c2, c3 = st.columns(3)

with c1:
    pclass = st.selectbox("Passenger Class", [1, 2, 3], key="filter_class")

with c2:
    age_range = st.slider("Age Range", 0, 80, (20, 30), key="filter_age")

with c3:
    gender = st.radio("Gender", ["male", "female"], key="filter_gender")

filtered = df[
    (df["Pclass"] == pclass) &
    (df["Age"].between(age_range[0], age_range[1])) &
    (df["Sex"] == gender)
].copy()

#BATCH PREDICTION
st.markdown("##  Batch Survival Predictions")

if not filtered.empty:
    probs, preds, risks = predictor.predict(filtered)

    filtered["Survival Probability"] = probs
    filtered["Risk Score"] = risks
    filtered["Prediction"] = preds

    st.dataframe(
        filtered.sort_values("Survival Probability", ascending=False)[
            ["Name", "Age", "Sex", "Pclass",
             "Survival Probability", "Risk Score", "Prediction"]
        ],
        use_container_width=True
    )
else:
    st.warning("No passengers match selected filters")

# INDIVIDUAL PREDICTION
st.divider()
st.markdown("## Individual Passenger Prediction")

a1, a2, a3 = st.columns(3)

with a1:
    age = st.slider("Age", 0, 80, 30)
    sex = st.radio("Sex", ["male", "female"])

with a2:
    pclass_i = st.selectbox("Class", [1, 2, 3])
    fare = st.slider("Fare", 0.0, 500.0, 32.0)

with a3:
    sibsp = st.number_input("Siblings", 0, 8, 0)
    parch = st.number_input("Parents", 0, 6, 0)

if st.button("Predict Survival"):
    sample = pd.DataFrame([{
        "Age": age,
        "Sex": sex,
        "Pclass": pclass_i,
        "Fare": fare,
        "SibSp": sibsp,
        "Parch": parch
    }])

    with st.spinner("Analyzing survival..."):
        time.sleep(1)
        probs, preds, risks = predictor.predict(sample)

    m1, m2, m3 = st.columns(3)
    m1.metric("Survival Probability", f"{probs[0]:.2%}")
    m2.metric("Risk Index", f"{risks[0]}/100")
    m3.metric("Prediction", "Survived" if preds[0] == 1 else "Did Not Survive")

    st.progress(int(probs[0] * 100))

#  KAFKA REAL-TIME STREAMING

st.divider()
st.markdown("## Real-Time Kafka Predictions")

if not KAFKA_AVAILABLE:
    st.info("Kafka not installed. Streaming disabled.")
else:
    if st.toggle("Enable Kafka Streaming"):
        try:
            consumer = KafkaConsumer(
                "titanic_stream",
                bootstrap_servers="localhost:9092",
                value_deserializer=lambda x: json.loads(x.decode("utf-8")),
                consumer_timeout_ms=2000
            )

            for msg in consumer:
                row = pd.DataFrame([msg.value])
                probs, _, _ = predictor.predict(row)
                row["Survival Probability"] = round(probs[0], 3)
                st.dataframe(row, use_container_width=True)
                break

        except Exception:
            st.error("Kafka broker not running. Start Docker + Kafka.")
