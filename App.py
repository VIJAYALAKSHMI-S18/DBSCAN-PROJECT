import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import joblib

# Page config
st.set_page_config(page_title="Wine DBSCAN Predictor", layout="centered")

st.title("🍷 Wine DBSCAN Clustering")
st.write("Give DBSCAN parameters and get cluster distribution")

# ---------------- LOAD DATA & SCALER ----------------
df = pd.read_csv("wine_clustering_data.csv")
scaler = joblib.load("scaler.pkl")

X = df.drop("proline", axis=1)
X_scaled = scaler.transform(X)

# ---------------- INPUTS ----------------
st.subheader("🔧 Input Parameters")

eps = st.slider(
    "eps (neighborhood radius)",
    min_value=0.1,
    max_value=5.0,
    value=2.0,
    step=0.1
)

min_samples = st.slider(
    "min_samples",
    min_value=2,
    max_value=10,
    value=2,
    step=1
)

# ---------------- PREDICT BUTTON ----------------
if st.button("Run DBSCAN"):
    dbscan = DBSCAN(eps=eps, min_samples=min_samples)
    labels = dbscan.fit_predict(X_scaled)

    # Cluster counts
    cluster_counts = pd.Series(labels).value_counts().sort_index()

    # ---------------- OUTPUT ----------------
    st.subheader("📊 Output: Cluster Distribution")

    fig, ax = plt.subplots()
    ax.bar(cluster_counts.index.astype(str), cluster_counts.values)
    ax.set_xlabel("Cluster Label")
    ax.set_ylabel("Number of Wines")

    st.pyplot(fig)
