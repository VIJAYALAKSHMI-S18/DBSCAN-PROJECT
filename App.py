import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import joblib

# Page config
st.set_page_config(page_title="Wine DBSCAN Clustering", layout="wide")

st.title("🍷 Wine Clustering using DBSCAN")
st.write("Unsupervised clustering based on chemical properties of wine")

# Load dataset
@st.cache_data
def load_data():
    return pd.read_csv("wine_clustering_data.csv")

df = load_data()

# Load models
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
dbscan = joblib.load("dbscan_model.pkl")

# Show dataset
st.subheader("📄 Dataset Preview")
st.dataframe(df.head())

st.write("Shape of dataset:", df.shape)

# Prepare data
X = df.drop("proline", axis=1)
X_scaled = scaler.transform(X)

# PCA transform
X_pca = pca.transform(X_scaled)

# DBSCAN clustering
labels = dbscan.fit_predict(X_pca)
df["cluster"] = labels

# Cluster info
st.subheader("🔢 Cluster Distribution")
st.write(pd.Series(labels).value_counts())

# Visualization
st.subheader("📊 Cluster Visualization (PCA)")
fig, ax = plt.subplots(figsize=(7, 5))

for cluster in np.unique(labels):
    ax.scatter(
        X_pca[labels == cluster, 0],
        X_pca[labels == cluster, 1],
        label=f"Cluster {cluster}",
        s=40
    )

ax.set_xlabel("PCA Component 1")
ax.set_ylabel("PCA Component 2")
ax.legend(title="Cluster")
st.pyplot(fig)

# Download clustered data
st.subheader("⬇️ Download Clustered Data")
csv = df.to_csv(index=False).encode("utf-8")
st.download_button(
    label="Download CSV",
    data=csv,
    file_name="wine_clustered_output.csv",
    mime="text/csv"
)
