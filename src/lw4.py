# %%

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial import Voronoi, voronoi_plot_2d
from sklearn.cluster import AgglomerativeClustering, DBSCAN, KMeans
from sklearn.metrics import silhouette_score, silhouette_samples
from sklearn.preprocessing import StandardScaler

from config import DATASETS_DIR, REPORTS_DIR

DATASET_PATHS = {
    "dataset1": DATASETS_DIR / "4" / "lab4_arcs.csv",
    "dataset2": DATASETS_DIR / "4" / "lab4_roll.csv",
}

REPORT_DIR = REPORTS_DIR / "lw4"
IMAGES_DIR = REPORT_DIR / "images"

RANDOM_STATE = 42
K_RANGE_ELBOW = range(1, 16)
K_RANGE_SILHOUETTE = range(2, 16)
LINKAGE_CANDIDATES = ("ward", "average", "complete", "single")

# Same cluster id -> same color on every plot (tab10 for 0..9, tab20 continues 10..14).
# K-Means / hierarchical use k <= 15 -> labels 0..14. DBSCAN can yield more clusters than
# that (e.g. labels 0..15), so we extend colors beyond the K-Means k-range.
_MAX_K_KMEANS = max(K_RANGE_ELBOW.stop - 1, K_RANGE_SILHOUETTE.stop - 1)
_TAB10 = list(sns.color_palette("tab10", n_colors=10))
_TAB20_REST = list(sns.color_palette("tab20", n_colors=20))[10:_MAX_K_KMEANS]
_PRIMARY_CLUSTER_COLORS = _TAB10 + _TAB20_REST if _MAX_K_KMEANS > 10 else _TAB10[:_MAX_K_KMEANS]
_EXTRA_DBSCAN_COLORS = list(sns.color_palette("husl", n_colors=48))

CLUSTER_PALETTE = {}
for i in range(len(_PRIMARY_CLUSTER_COLORS)):
    CLUSTER_PALETTE[str(i)] = _PRIMARY_CLUSTER_COLORS[i]
# Extra ids for DBSCAN (and other methods) when label count exceeds K-Means bounds
_MAX_PALETTE_ID = 64
for i in range(len(_PRIMARY_CLUSTER_COLORS), _MAX_PALETTE_ID):
    CLUSTER_PALETTE[str(i)] = _EXTRA_DBSCAN_COLORS[(i - len(_PRIMARY_CLUSTER_COLORS)) % len(_EXTRA_DBSCAN_COLORS)]

NOISE_COLOR = "lightgray"


def cluster_color_for_id(cid: int):
    """Stable color for integer cluster label; safe for ids beyond K-Means k-range."""
    key = str(int(cid))
    if key in CLUSTER_PALETTE:
        return CLUSTER_PALETTE[key]
    return _EXTRA_DBSCAN_COLORS[int(cid) % len(_EXTRA_DBSCAN_COLORS)]


def cluster_hue_order(k: int):
    return [str(i) for i in range(k)]


def dbscan_scatter_palette_and_hue_order(labels: np.ndarray):
    cluster_ids = sorted(int(c) for c in set(labels.tolist()) if c != -1)
    palette = {str(c): cluster_color_for_id(c) for c in cluster_ids}
    palette["noise"] = NOISE_COLOR
    hue_order = [str(c) for c in cluster_ids] + ["noise"]
    return palette, hue_order


os.makedirs(IMAGES_DIR, exist_ok=True)


# %%
# 1.1-1.2 Data loading and validation

loaded = {}
for name, path in DATASET_PATHS.items():
    df = pd.read_csv(path, index_col=0)
    assert not df.empty, f"{name}: empty dataframe"
    assert not df.isna().any().any(), f"{name}: contains NaN"
    print(f"\n{name}: shape={df.shape}")
    print(df.head())
    print(df.describe())
    loaded[name] = df


# %%
# 1.3 Scatter plots of original data

for name, df in loaded.items():
    sns.scatterplot(data=df, x=df.columns[0], y=df.columns[1], s=30, edgecolor="k", alpha=0.8)
    plt.title(f"{name}: original data")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / f"{name}_viz.png")
    plt.show()


# %%
# 1.4 Standardization
# StandardScaler is chosen because all clustering algorithms here are distance-based,
# so features must be on the same scale.

scaled = {}
for name, df in loaded.items():
    scaler = StandardScaler()
    scaled[name] = scaler.fit_transform(df)


# %%
# 2.1 K-Means: elbow method

for name in loaded:
    X = scaled[name]
    inertias = []
    for k in K_RANGE_ELBOW:
        km = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        km.fit(X)
        inertias.append(km.inertia_)

    elbow_df = pd.DataFrame({"k": list(K_RANGE_ELBOW), "inertia": inertias})
    sns.lineplot(data=elbow_df, x="k", y="inertia", marker="o")
    plt.title(f"{name}: elbow method")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / f"{name}_kmeans_elbow.png")
    plt.show()


# %%
# 2.2 K-Means: silhouette method

best_k = {}
for name in loaded:
    X = scaled[name]
    scores = []
    for k in K_RANGE_SILHOUETTE:
        labels = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10).fit_predict(X)
        scores.append(silhouette_score(X, labels))

    best_k[name] = K_RANGE_SILHOUETTE.start + int(np.argmax(scores))
    print(f"{name}: best k={best_k[name]}, silhouette={max(scores):.4f}")

    sil_df = pd.DataFrame({"k": list(K_RANGE_SILHOUETTE), "silhouette": scores})
    sns.lineplot(data=sil_df, x="k", y="silhouette", marker="o")
    plt.title(f"{name}: silhouette method")
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / f"{name}_kmeans_silhouette.png")
    plt.show()

# %%
# 2.3 Hardcode optimal k

# best_k = {"dataset1": 4, "dataset2": 3}

# %%
# 2.3 K-Means: fit with optimal k

kmeans_results = {}
for name in loaded:
    X = scaled[name]
    km = KMeans(n_clusters=best_k[name], random_state=RANDOM_STATE, n_init=10)
    labels = km.fit_predict(X)
    kmeans_results[name] = {"labels": labels, "centers": km.cluster_centers_}

# %%
# 2.3b K-Means: silhouette plot for the best k (per-sample silhouette coefficients)
for name in loaded:
    X = scaled[name]
    labels = kmeans_results[name]["labels"]
    k = best_k[name]
    sil_samples = silhouette_samples(X, labels)
    sil_avg = silhouette_score(X, labels)
    fig, ax = plt.subplots(figsize=(8, 6))
    y_lower = 10
    for cluster_id in range(k):
        vals = sil_samples[labels == cluster_id]
        vals.sort()
        n = vals.shape[0]
        y_upper = y_lower + n
        color = CLUSTER_PALETTE[str(cluster_id)]
        ax.fill_betweenx(
            np.arange(y_lower, y_upper),
            0,
            vals,
            facecolor=color,
            edgecolor=color,
            alpha=0.75,
        )

        ax.text(-0.05, y_lower + 0.5 * n, str(cluster_id))
        y_lower = y_upper + 10

    ax.axvline(x=sil_avg, color="crimson", linestyle="--", linewidth=1.2, label=f"mean silhouette = {sil_avg:.3f}")
    ax.set_title(f"{name}: silhouette plot (k={k})")
    ax.set_xlabel("Silhouette coefficient")
    ax.set_ylabel("Sample index (sorted within each cluster)")
    ax.set_xlim(-0.25, 1.05)
    ax.legend(loc="lower right")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / f"{name}_kmeans_silhouette_plot.png")
    plt.show()

# %%
# 2.4 K-Means: cluster scatter plots

for name in loaded:
    X = scaled[name]
    labels = kmeans_results[name]["labels"]
    cols = list(loaded[name].columns)

    scatter_df = pd.DataFrame(X, columns=cols)
    scatter_df["cluster"] = labels.astype(str)

    sns.scatterplot(
        data=scatter_df,
        x=cols[0],
        y=cols[1],
        hue="cluster",
        hue_order=cluster_hue_order(best_k[name]),
        palette=CLUSTER_PALETTE,
        s=30,
        edgecolor="k",
        alpha=0.8,
    )
    plt.title(f"{name}: K-Means clusters (k={best_k[name]})")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / f"{name}_kmeans_scatter.png")
    plt.show()


# %%
# 2.5 K-Means: Voronoi diagrams with centroids
# Voronoi has no seaborn equivalent, so matplotlib is used for region lines.

for name in loaded:
    X = scaled[name]
    labels = kmeans_results[name]["labels"]
    centers = kmeans_results[name]["centers"]
    cols = list(loaded[name].columns)

    vor = Voronoi(centers)
    fig, ax = plt.subplots(figsize=(8, 6))
    voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors="gray", line_width=1.2, line_alpha=0.7)

    scatter_df = pd.DataFrame(X, columns=cols)
    scatter_df["cluster"] = labels.astype(str)
    sns.scatterplot(
        data=scatter_df,
        x=cols[0],
        y=cols[1],
        hue="cluster",
        hue_order=cluster_hue_order(best_k[name]),
        palette=CLUSTER_PALETTE,
        s=25,
        edgecolor="k",
        alpha=0.85,
        ax=ax,
        legend=False,
    )

    ax.scatter(centers[:, 0], centers[:, 1], c="red", marker="*", s=300, edgecolors="k", zorder=6, label="centroids")
    ax.set_title(f"{name}: Voronoi diagram with centroids")
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / f"{name}_kmeans_voronoi.png")
    plt.show()


# %%
# 2.6 K-Means: violin plots per feature by cluster

for name in loaded:
    df = loaded[name].copy()
    df["cluster"] = kmeans_results[name]["labels"].astype(str)
    cols = list(loaded[name].columns)

    melted = df.melt(id_vars=["cluster"], value_vars=cols, var_name="feature", value_name="value")
    sns.violinplot(
        data=melted,
        x="feature",
        y="value",
        hue="cluster",
        hue_order=cluster_hue_order(best_k[name]),
        palette=CLUSTER_PALETTE,
        cut=0,
        inner="quart",
    )
    plt.title(f"{name}: feature distributions by K-Means clusters")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / f"{name}_kmeans_violin.png")
    plt.show()


# %%
# 2.7 K-Means: cluster statistics (count, mean, std, min, max)

for name in loaded:
    df = loaded[name].copy()
    df["cluster"] = kmeans_results[name]["labels"].astype(int)
    print(f"\n{name}: K-Means cluster statistics")
    print(df.groupby("cluster").agg(["count", "mean", "std", "min", "max"]).round(4))

# %%
# 3.1 DBSCAN: parameter tuning
# Strategy: grid search over eps in [0.10, 1.60] and min_samples in {3, 5, 8, 12},
# selecting the combination that maximizes silhouette score on non-noise points.

def tune_dbscan(X: np.ndarray):
    best = {"silhouette": -1.0, "eps": None, "min_samples": None, "labels": None}
    for eps in np.linspace(0.2, 1.00, 31):
        for min_samples in range(3, 15):
            model = DBSCAN(eps=float(eps), min_samples=min_samples)
            lab = model.fit_predict(X)
            n_clu = len(set(lab)) - (1 if -1 in lab else 0)
            if n_clu < 2:
                continue
            core = lab != -1
            if core.sum() < max(min_samples * 3, 20):
                continue
            sil = silhouette_score(X[core], lab[core])
            if sil > best["silhouette"]:
                best = {"silhouette": sil, "eps": float(eps), "min_samples": min_samples, "labels": lab.copy()}
    if best["labels"] is None:
        lab = DBSCAN(eps=0.35, min_samples=5).fit_predict(X)
        best = {"silhouette": np.nan, "eps": 0.35, "min_samples": 5, "labels": lab}
    return best

# %%

# Automatic method
# dbscan_results = {}
# for name in loaded:
#     result = tune_dbscan(scaled[name])
#     dbscan_results[name] = result
#     n_clusters = len(set(result["labels"])) - (1 if -1 in result["labels"] else 0)
#     n_noise = int((result["labels"] == -1).sum())
#     sil = result["silhouette"]
#     sil_str = f"{sil:.4f}" if not (isinstance(sil, float) and np.isnan(sil)) else "nan"
#     print(f"\n{name}: DBSCAN best params")
#     print(f"  eps={result['eps']:.3f}, min_samples={result['min_samples']}")
#     print(f"  clusters={n_clusters}, noise points={n_noise}, silhouette={sil_str}")


# %%
# Hardcoded model parameters

dbscan_results = {
    "dataset1": {"eps": None, "min_samples": None, "labels": None},
    "dataset2": {"eps": None, "min_samples": None, "labels": None}
}
dbscan_results["dataset1"]["eps"]         = 0.2
dbscan_results["dataset1"]["min_samples"] = 10
dbscan_results["dataset1"]["labels"]      = DBSCAN(eps=dbscan_results["dataset1"]["eps"], min_samples=dbscan_results["dataset1"]["min_samples"]).fit_predict(scaled["dataset1"])
dbscan_results["dataset2"]["eps"]         = 0.2
dbscan_results["dataset2"]["min_samples"] = 14
dbscan_results["dataset2"]["labels"]      = DBSCAN(eps=dbscan_results["dataset2"]["eps"], min_samples=dbscan_results["dataset2"]["min_samples"]).fit_predict(scaled["dataset2"])

# %%
# 3.2 DBSCAN: cluster scatter plots

for name in loaded:
    X = scaled[name]
    labels = dbscan_results[name]["labels"]
    cols = list(loaded[name].columns)

    scatter_df = pd.DataFrame(X, columns=cols)
    scatter_df["label"] = np.where(labels == -1, "noise", labels.astype(str))

    palette, hue_order = dbscan_scatter_palette_and_hue_order(labels)

    sns.scatterplot(
        data=scatter_df,
        x=cols[0],
        y=cols[1],
        hue="label",
        hue_order=hue_order,
        palette=palette,
        s=30,
        edgecolor="k",
        alpha=0.8,
    )
    plt.title(f"{name}: DBSCAN clustering")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / f"{name}_dbscan_scatter.png")
    plt.show()


# %%
# 4.1 Hierarchical clustering: linkage selection

def choose_best_linkage(X: np.ndarray, n_clusters: int):
    best_method, best_score, best_labels = None, -1.0, None
    # for method in LINKAGE_CANDIDATES:
    for method in ["single"]:
        model = AgglomerativeClustering(n_clusters=n_clusters, linkage=method)
        lab = model.fit_predict(X)
        score = silhouette_score(X, lab)
        if score > best_score:
            best_method, best_score, best_labels = method, score, lab
    return best_method, best_score, best_labels


hier_results = {}
for name in loaded:
    X = scaled[name]
    k = best_k[name]
    method, sil, labels = choose_best_linkage(X, k)
    print(f"\n{name}: hierarchical clustering (k={k})")
    print(f"  Best linkage: {method}, silhouette: {sil:.4f}")

    final_labels, final_k = labels, k
    if k > 2:
        _, alt_sil, alt_labels = choose_best_linkage(X, k - 1)
        if alt_sil > sil:
            final_labels, final_k = alt_labels, k - 1
            print(f"  Improved with k={final_k}, silhouette: {alt_sil:.4f}")

    hier_results[name] = {"labels": final_labels, "k": final_k, "linkage": method}

# %%
# Manual tuning

hier_results = {
    "dataset1": {"labels": None, "linkage": None, "k": None},
    "dataset2": {"labels": None, "linkage": None, "k": None}
}
hier_results["dataset1"]["linkage"] = "single"
model1 = AgglomerativeClustering(
    n_clusters=None,
    linkage=hier_results["dataset1"]["linkage"],
    distance_threshold=0.32
)
hier_results["dataset1"]["labels"] = model1.fit_predict(scaled["dataset1"])
hier_results["dataset1"]["k"]      = model1.n_clusters_


hier_results["dataset2"]["linkage"] = "single"
hier_results["dataset2"]["k"]       = 3
model2 = AgglomerativeClustering(
    n_clusters=hier_results["dataset2"]["k"],
    linkage=hier_results["dataset2"]["linkage"]
)
hier_results["dataset2"]["labels"] = model2.fit_predict(scaled["dataset2"])

# %%
# 4.1 Hierarchical clustering: dendrograms

for name in loaded:
    X = scaled[name]
    method = hier_results[name]["linkage"]

    Z = linkage(X, method=method)
    plt.figure(figsize=(12, 4))
    dendrogram(Z, truncate_mode="lastp", p=40, leaf_rotation=90, leaf_font_size=8)
    plt.title(f"{name}: dendrogram ({method} linkage)")
    plt.xlabel("Merged cluster index")
    plt.ylabel("Linkage distance")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / f"{name}_hier_dendrogram.png")
    plt.show()


# %%
# 4.2 Hierarchical clustering: scatter plots

for name in loaded:
    X = scaled[name]
    labels = hier_results[name]["labels"]
    k = hier_results[name]["k"]
    method = hier_results[name]["linkage"]
    cols = list(loaded[name].columns)

    scatter_df = pd.DataFrame(X, columns=cols)
    scatter_df["cluster"] = labels.astype(str)

    sns.scatterplot(
        data=scatter_df,
        x=cols[0],
        y=cols[1],
        hue="cluster",
        hue_order=cluster_hue_order(k),
        palette=CLUSTER_PALETTE,
        s=30,
        edgecolor="k",
        alpha=0.8,
    )
    plt.title(f"{name}: hierarchical clusters (k={k}, linkage={method})")
    plt.tight_layout()
    plt.savefig(IMAGES_DIR / f"{name}_hier_scatter.png")
    plt.show()
