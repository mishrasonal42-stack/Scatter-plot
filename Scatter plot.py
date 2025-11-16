import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
import plotly.express as px

st.set_page_config(page_title="Clustering Dashboard", layout="wide")

st.title("📊 Clustering Dashboard (Streamlit-Friendly Version)")

# -----------------------------------------------------
# Upload data
# -----------------------------------------------------
file = st.file_uploader("Lifestyle_and_Health_Risk_Prediction_Synthetic_Dataset (1).csv", type=["csv"])

if file:
    df = pd.read_csv("Lifestyle_and_Health_Risk_Prediction_Synthetic_Dataset (1).csv")
    st.success("File uploaded successfully!")

    st.subheader("Data Preview")
    st.write(df.head())

    # numeric only
    numeric_df = df.select_dtypes(include="number")
    if numeric_df.shape[1] < 2:
        st.error("Not enough numeric columns for clustering.")
        st.stop()

    # -----------------------------------------------------
    # Scaling
    # -----------------------------------------------------
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_df)

    # -----------------------------------------------------
    # Select K
    # -----------------------------------------------------
    st.subheader("Select Number of Clusters (k)")
    k = st.slider("Choose k", 2, 10, 3)

    # -----------------------------------------------------
    # KMeans
    # -----------------------------------------------------
    model = KMeans(n_clusters=k, random_state=42)
    clusters = model.fit_predict(scaled)
    df["cluster"] = clusters

    st.success("Clustering done!")

    # -----------------------------------------------------
    # Metrics
    # -----------------------------------------------------
    inertia = model.inertia_
    sil = silhouette_score(scaled, clusters)

    st.subheader("📈 Clustering Metrics")
    col1, col2 = st.columns(2)
    col1.metric("Inertia", round(inertia, 2))
    col2.metric("Silhouette Score", round(sil, 4))

    # -----------------------------------------------------
    # PCA 2D
    # -----------------------------------------------------
    pca = PCA(n_components=2)
    pca_vals = pca.fit_transform(scaled)
    df["PCA1"] = pca_vals[:, 0]
    df["PCA2"] = pca_vals[:, 1]

    st.subheader("📉 PCA Scatter Plot (Matplotlib)")
    fig, ax = plt.subplots(figsize=(6, 4))
    for c in range(k):
        ax.scatter(
            df[df["cluster"] == c]["PCA1"],
            df[df["cluster"] == c]["PCA2"],
            label=f"Cluster {c}",
            s=40
        )
    ax.set_xlabel("PCA1")
    ax.set_ylabel("PCA2")
    ax.legend()
    st.pyplot(fig)

    # -----------------------------------------------------
    # Plotly interactive
    # -----------------------------------------------------
    st.subheader("🌈 Interactive PCA Plot (Plotly)")
    fig2 = px.scatter(
        df, x="PCA1", y="PCA2",
        color="cluster",
        hover_data=df.columns,
        title="Interactive Cluster Visualization"
    )
    st.plotly_chart(fig2, use_container_width=True)

    # -----------------------------------------------------
    # 3D PCA plot
    # -----------------------------------------------------
    pca3 = PCA(n_components=3)
    pca3_vals = pca3.fit_transform(scaled)
    df["P1"] = pca3_vals[:, 0]
    df["P2"] = pca3_vals[:, 1]
    df["P3"] = pca3_vals[:, 2]

    st.subheader("🧊 3D Interactive Cluster Plot")
    fig3 = px.scatter_3d(
        df, x="P1", y="P2", z="P3",
        color="cluster",
        hover_data=df.columns,
        title="3D PCA Clustering"
    )
    st.plotly_chart(fig3, use_container_width=True)

    # ---------------------------------------------------
