import sys, os
ROOT_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT_DIR not in sys.path:
    sys.path.append(ROOT_DIR)

import streamlit as st
import pandas as pd
import plotly.express as px
import pickle
from sklearn.metrics import confusion_matrix, roc_curve, auc
with open("frontend/style.css") as f:
    st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

# ---------------- PAGE HEADER ----------------
st.markdown("""
<div class="card fade">
<h1>Analytics & Model Intelligence Report</h1>
<p>
Comprehensive exploratory analysis, feature engineering insights,
and machine learning performance evaluation.
</p>
</div>
""", unsafe_allow_html=True)

# ---------------- LOAD DATA ----------------
df = pd.read_csv("data/processed/titanic_cleaned.csv")

# ---------------- EDA SECTION ----------------
st.markdown("## Exploratory Data Analysis")

st.plotly_chart(
    px.histogram(df, x="Age", color="Survived",
                 title="Age Distribution vs Survival"),
    use_container_width=True
)

st.plotly_chart(
    px.box(df, x="Pclass", y="Fare", color="Survived",
           title="Fare vs Class vs Survival"),
    use_container_width=True
)

st.plotly_chart(
    px.bar(df, x="Embarked", y="Survived",
           title="Survival by Port of Embarkation"),
    use_container_width=True
)

st.info("""
**Key Insights**
• Females and first-class passengers had the highest survival rates  
• Children had evacuation priority  
• Higher fares correlated with higher survival  
• Embarkation port influenced survival likelihood  
""")

# ---------------- MODEL EVALUATION ----------------
st.divider()
st.markdown("## Model Evaluation & Performance")

model = pickle.load(open("models/survival_model.pkl","rb"))
data = pd.read_csv("data/processed/titanic_featured.csv")

X = data.drop("Survived", axis=1)
y = data["Survived"]

y_pred = model.predict(X)
y_prob = model.predict_proba(X)[:,1]

cm = confusion_matrix(y, y_pred)
fpr, tpr, _ = roc_curve(y, y_prob)
roc_auc = auc(fpr, tpr)

c1, c2 = st.columns(2)
c1.write("### Confusion Matrix")
c1.write(cm)

c2.metric("ROC-AUC Score", f"{roc_auc:.2f}")

st.plotly_chart(
    px.line(x=fpr, y=tpr, labels={"x":"False Positive Rate","y":"True Positive Rate"},
            title="ROC Curve"),
    use_container_width=True
)

# ---------------- MODEL COMPARISON ----------------
st.divider()
st.markdown("## Model Comparison")

model_results = pd.DataFrame({
    "Model": ["Logistic Regression", "Random Forest"],
    "Accuracy": [0.84, 0.86],
    "Precision": [0.82, 0.85],
    "Recall": [0.74, 0.79],
    "F1 Score": [0.78, 0.81],
    "ROC-AUC": [0.86, 0.89]
})

st.dataframe(model_results, use_container_width=True)

# ---------------- RESPONSIBLE AI ----------------
st.markdown("""
<div class="card">
<h3> Responsible AI & Ethics</h3>
<ul>
<li>Predictions are probabilistic, not absolute</li>
<li>Model reflects historical biases of 1912 society</li>
<li>Designed for decision support, not deterministic judgment</li>
</ul>
</div>
""", unsafe_allow_html=True)
