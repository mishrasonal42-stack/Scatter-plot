pip install pandas numpy scikit-learn seaborn matplotlib plotly reportlab streamlit fpdf
"""
generate_report.py
Usage:
    python generate_report.py --input path/to/dataset.csv --k 3 --out report.pdf

Produces:
 - numeric-feature selection + scaling
 - elbow & silhouette plots
 - KMeans clustering
 - PCA projections (2D, 3D)
 - boxplots per cluster
 - heatmap of cluster centers
 - assembles a PDF with the images + textual summary
"""
import argparse
import os
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

# PDF
from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Image as RLImage
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch

sns.set(style="whitegrid")

def ensure_dir(d):
    os.makedirs(d, exist_ok=True)

def save_figure(fig, path):
    fig.savefig(path, bbox_inches="tight")
    plt.close(fig)

def run_clustering_pipeline(df, k=3, outdir="report_assets"):
    ensure_dir(outdir)

    # 1. select numeric
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.shape[1] < 2:
        raise ValueError("Need at least two numeric columns for clustering/visualization.")

    # 2. scale
    scaler = StandardScaler()
    scaled = scaler.fit_transform(numeric_df)

    # 3. elbow & silhouette curves
    inertia_list = []
    silhouette_list = []
    K = range(2, 10)
    for kk in K:
        km = KMeans(n_clusters=kk, random_state=42, n_init=10)
        km.fit(scaled)
        inertia_list.append(km.inertia_)
        silhouette_list.append(silhouette_score(scaled, km.labels_))

    # save elbow plot
    fig = plt.figure(figsize=(7,4))
    plt.plot(list(K), inertia_list, marker="o")
    plt.title("Elbow Curve (Inertia)")
    plt.xlabel("k")
    plt.ylabel("Inertia")
    save_figure(fig, os.path.join(outdir, "elbow.png"))

    # save silhouette plot
    fig = plt.figure(figsize=(7,4))
    plt.plot(list(K), silhouette_list, marker="o")
    plt.title("Silhouette Score by k")
    plt.xlabel("k")
    plt.ylabel("Silhouette Score")
    save_figure(fig, os.path.join(outdir, "silhouette.png"))

    # 4. final KMeans
    model = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = model.fit_predict(scaled)
    df2 = df.copy()
    df2["cluster"] = labels

    # 5. cluster sizes
    cluster_sizes = df2["cluster"].value_counts().sort_index()

    # 6. cluster centers (unscaled)
    centers_scaled = model.cluster_centers_
    centers_unscaled = scaler.inverse_transform(centers_scaled)
    centers_df = pd.DataFrame(centers_unscaled, columns=numeric_df.columns)
    centers_df.index.name = "cluster"

    # 7. PCA 2D
    pca = PCA(n_components=2, random_state=42)
    pca_vals = pca.fit_transform(scaled)
    df2["PC1"] = pca_vals[:,0]
    df2["PC2"] = pca_vals[:,1]
    fig = plt.figure(figsize=(7,6))
    sns.scatterplot(data=df2, x="PC1", y="PC2", hue="cluster", palette="tab10", legend="full")
    plt.title("PCA 2D - clusters")
    save_figure(fig, os.path.join(outdir, "pca2d.png"))

    # 8. PCA 3D (save as 2D projection image for PDF)
    pca3 = PCA(n_components=3, random_state=42)
    pca3_vals = pca3.fit_transform(scaled)
    df2["P3_1"] = pca3_vals[:,0]
    df2["P3_2"] = pca3_vals[:,1]
    df2["P3_3"] = pca3_vals[:,2]
    fig = plt.figure(figsize=(7,6))
    ax = fig.add_subplot(111, projection='3d')
    scatter = ax.scatter(df2["P3_1"], df2["P3_2"], df2["P3_3"], c=df2["cluster"], cmap="tab10", s=20)
    ax.set_title("PCA 3D (static image)")
    save_figure(fig, os.path.join(outdir, "pca3d.png"))

    # 9. boxplots for each numeric feature by cluster
    ncols = 3
    nrows = int(np.ceil(len(numeric_df.columns) / ncols))
    fig = plt.figure(figsize=(ncols*4, nrows*3))
    for i, col in enumerate(numeric_df.columns, 1):
        ax = plt.subplot(nrows, ncols, i)
        sns.boxplot(data=df2, x="cluster", y=col, ax=ax)
        ax.set_title(f"{col} by cluster")
    plt.tight_layout()
    save_figure(fig, os.path.join(outdir, "boxplots_by_cluster.png"))

    # 10. heatmap cluster centers (unscaled)
    fig = plt.figure(figsize=(max(6, 0.6*len(numeric_df.columns)), 4))
    sns.heatmap(centers_df, annot=True, cmap="coolwarm", fmt=".2f")
    plt.title("Cluster Centers (unscaled values)")
    save_figure(fig, os.path.join(outdir, "centers_heatmap.png"))

    # 11. correlation heatmap of numeric features (helps interpret clusters)
    corr = numeric_df.corr()
    fig = plt.figure(figsize=(8,6))
    sns.heatmap(corr, annot=True, cmap="vlag", center=0)
    plt.title("Feature Correlation")
    save_figure(fig, os.path.join(outdir, "correlation_heatmap.png"))

    # 12. pairplot (for smaller feature sets this can be large; we create a small pairplot of up to 6 features)
    pair_cols = list(numeric_df.columns[:6])
    try:
        pp = sns.pairplot(df2[pair_cols + ["cluster"]], hue="cluster", palette="tab10", plot_kws={"s": 10})
        pp.savefig(os.path.join(outdir, "pairplot.png"))
        plt.close()
    except Exception:
        # fallback: skip if too expensive
        pass

    # summary dictionary
    summary = {
        "k": k,
        "cluster_sizes": cluster_sizes.to_dict(),
        "centers_df": centers_df,
        "outdir": outdir,
        "generated_at": datetime.utcnow().isoformat() + "Z"
    }

    return df2, summary

def build_pdf_report(summary, pdf_path):
    outdir = summary["outdir"]
    styles = getSampleStyleSheet()
    doc = SimpleDocTemplate(pdf_path, pagesize=A4)
    story = []

    # Title
    story.append(Paragraph("Clustering Analysis Report", styles["Title"]))
    story.append(Spacer(1, 0.1*inch))
    story.append(Paragraph(f"Generated: {summary['generated_at']}", styles["Normal"]))
    story.append(Spacer(1, 0.2*inch))

    # High level info
    story.append(Paragraph(f"Chosen k: {summary['k']}", styles["Heading2"]))
    story.append(Paragraph("Cluster sizes:", styles["Normal"]))
    for cl, size in summary["cluster_sizes"].items():
        story.append(Paragraph(f" - Cluster {cl} : {size} rows", styles["Normal"]))
    story.append(Spacer(1, 0.2*inch))

    # Add images
    imgs = [
        "elbow.png",
        "silhouette.png",
        "pca2d.png",
        "pca3d.png",
        "centers_heatmap.png",
        "boxplots_by_cluster.png",
        "correlation_heatmap.png",
        "pairplot.png"
    ]
    for img in imgs:
        p = os.path.join(outdir, img)
        if os.path.exists(p):
            story.append(Spacer(1, 0.1*inch))
            # fit width
            story.append(RLImage(p, width=6.5*inch, height=None))
            story.append(Spacer(1, 0.1*inch))

    doc.build(story)
    return pdf_path

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", "-i", required=True, help="path to csv")
    parser.add_argument("--k", "-k", type=int, default=3, help="number of clusters")
    parser.add_argument("--out", "-o", default="clustering_report.pdf", help="output pdf path")
    parser.add_argument("--assets", default="report_assets", help="folder to write images")
    args = parser.parse_args()

    df = pd.read_csv(args.input)
    df2, summary = run_clustering_pipeline(df, k=args.k, outdir=args.assets)
    summary["k"] = args.k
    pdf = build_pdf_report(summary, args.out)
    print("Report written to:", pdf)

if __name__ == "__main__":
    main()
