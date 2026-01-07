import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import joblib
import numpy as np

# Load saved models
scaler = joblib.load("scaler.pkl")
pca = joblib.load("pca.pkl")
dbscan = joblib.load("dbscan_model.pkl")

st.set_page_config(page_title="Wine DBSCAN Clustering", layout="wide")

st.title("🍷 Wine Clustering using DBSCAN")
st.write("Unsupervised clustering based on chemical properties of wine")

# Upload dataset
uploaded_file = st.file_uploader("Upload Wine CSV file", type=["csv"])

if uploaded_file is not None:
    df = pd.read_csv(uploaded_file)

    st.subheader("📄 Dataset Preview")
    st.dataframe(df.head())

    # Scale data
    X = df.drop("proline", axis=1)
    X_scaled = scaler.transform(X)

    # PCA
    X_pca = pca.transform(X_scaled)

    # DBSCAN clustering
    labels = dbscan.fit_predict(X_pca)
    df["cluster"] = labels

    st.subheader("🔢 Cluster Counts")
    st.write(pd.Series(labels).value_counts())

    # Plot
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
    ax.legend()
    st.pyplot(fig)

    # Download clustered data
    st.subheader("⬇️ Download Clustered Data")
    csv = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "Download CSV",
        csv,
        "wine_clustered_output.csv",
        "text/csv"
    )
