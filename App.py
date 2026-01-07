import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import DBSCAN
import joblib

# Page config
st.set_page_config(page_title="Wine DBSCAN", layout="centered")

st.title("🍷 Wine DBSCAN Clustering")

# Load data & models
df = pd.read_csv("wine_clustering_data.csv")
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")

# ---------------- INPUTS ----------------
st.sidebar.header("🔧 DBSCAN Parameters")

eps = st.sidebar.slider(
    "eps",
    min_value=0.1,
    max_value=5.0,
    value=0.8,
    step=0.1
)

min_samples = st.sidebar.slider(
    "min_samples",
    min_value=2,
    max_value=10,
    value=5,
    step=1
)

# ---------------- PROCESS ----------------
X = df.drop("proline", axis=1)
X_scaled = scaler.transform(X)
X_pca = pca.transform(X_scaled)

dbscan = DBSCAN(eps=eps, min_samples=min_samples)
labels = dbscan.fit_predict(X_pca)

# ---------------- OUTPUT 1: BAR CHART ----------------
st.subheader("📊 Cluster Count (Bar Chart)")

cluster_counts = pd.Series(labels).value_counts().sort_index()

fig1, ax1 = plt.subplots()
ax1.bar(cluster_counts.index.astype(str), cluster_counts.values)
ax1.set_xlabel("Cluster Label")
ax1.set_ylabel("Number of Samples")

st.pyplot(fig1)

# ---------------- OUTPUT 2: CLUSTER SCATTER ----------------
st.subheader("📈 Cluster Visualization (PCA)")

fig2, ax2 = plt.subplots(figsize=(6, 5))

for cluster in np.unique(labels):
    ax2.scatter(
        X_pca[labels == cluster, 0],
        X_pca[labels == cluster, 1],
        label=f"Cluster {cluster}",
        s=40
    )

ax2.set_xlabel("PCA 1")
ax2.set_ylabel("PCA 2")
ax2.legend()

st
