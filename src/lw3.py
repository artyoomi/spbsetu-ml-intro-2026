# %%

import os

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler, label_binarize
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier, plot_tree

from config import DATASETS_DIR, REPORTS_DIR


DATASET_FILENAME = "lab3_2.csv"
DATASET_PATH = Path(DATASETS_DIR) / "3" / DATASET_FILENAME
REPORT_DIR = REPORTS_DIR / "lw3"
IMAGES_DIR = REPORT_DIR / "images"

RANDOM_STATE = 42
TEST_SIZE = 0.2


# %%
# 1. Data loading and primary analysis

df = pd.read_csv(DATASET_PATH)
df.drop(columns=df.columns[0], inplace=True)

target_col = "Quality"
feature_cols = [c for c in df.columns if c != target_col]

print("Dataset shape:", df.shape)
print("Class distribution:\n", df[target_col].value_counts())
print(df.head())

sns.scatterplot(data=df, x=feature_cols[0], y=feature_cols[1], hue=target_col)
plt.title("Original dataset by class")
os.makedirs(IMAGES_DIR, exist_ok=True)
plt.savefig(IMAGES_DIR / "dataset_viz.png")
plt.show()

# %%
# 1. Data preprocessing and split

X = df[feature_cols].copy()
y_text = df[target_col].copy()

label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y_text)
class_names = label_encoder.classes_

X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=TEST_SIZE,
    random_state=RANDOM_STATE,
    stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

y_test_bin = label_binarize(y_test, classes=np.arange(len(class_names)))


# %%
# Utility functions

def plot_decision_boundaries(model, X_data, y_data, labels, title, image_name):
    """Plot decision regions and points for 2D data."""
    x_min, x_max = X_data[:, 0].min() - 1.0, X_data[:, 0].max() + 1.0
    y_min, y_max = X_data[:, 1].min() - 1.0, X_data[:, 1].max() + 1.0
    xx, yy = np.meshgrid(
        np.linspace(x_min, x_max, 300),
        np.linspace(y_min, y_max, 300)
    )

    grid = np.c_[xx.ravel(), yy.ravel()]
    preds = model.predict(grid).reshape(xx.shape)

    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, preds, alpha=0.25, cmap="Set1")
    sns.scatterplot(
        x=X_data[:, 0],
        y=X_data[:, 1],
        hue=[labels[v] for v in y_data],
        palette="Set1",
        edgecolor="k"
    )
    plt.title(title)
    plt.xlabel(feature_cols[0])
    plt.ylabel(feature_cols[1])
    plt.savefig(IMAGES_DIR / image_name)
    plt.show()


def plot_confusion(y_true, y_pred, labels, title, image_name):
    disp = ConfusionMatrixDisplay.from_predictions(
        y_true,
        y_pred,
        display_labels=labels,
        cmap="Blues",
        xticks_rotation=30
    )
    disp.ax_.set_title(title)
    plt.savefig(IMAGES_DIR / image_name)
    plt.show()


def plot_multiclass_roc(y_true_bin, y_score, labels, title, image_name):
    plt.figure(figsize=(8, 6))
    for idx, label_name in enumerate(labels):
        fpr, tpr, _ = roc_curve(y_true_bin[:, idx], y_score[:, idx])
        auc_value = roc_auc_score(y_true_bin[:, idx], y_score[:, idx])
        plt.plot(fpr, tpr, label=f"{label_name} (AUC={auc_value:.3f})")
    plt.plot([0, 1], [0, 1], "k--", label="Random")
    plt.xlabel("False positive rate")
    plt.ylabel("True positive rate")
    plt.title(title)
    plt.legend()
    plt.savefig(IMAGES_DIR / image_name)
    plt.show()


def model_metrics(y_true, y_pred, y_score):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(y_true, y_pred, average="macro"),
        "recall_macro": recall_score(y_true, y_pred, average="macro"),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "auc_ovr_macro": roc_auc_score(
            y_test_bin, y_score, multi_class="ovr", average="macro"
        )
    }


all_results = {}


# %%
# 2. kNN

k_values = list(range(1, 31))
knn_unweighted_acc = []
knn_weighted_acc = []

for k in k_values:
    knn_u = KNeighborsClassifier(n_neighbors=k, weights="uniform")
    knn_w = KNeighborsClassifier(n_neighbors=k, weights="distance")
    knn_u.fit(X_train_scaled, y_train)
    knn_w.fit(X_train_scaled, y_train)
    knn_unweighted_acc.append(accuracy_score(y_test, knn_u.predict(X_test_scaled)))
    knn_weighted_acc.append(accuracy_score(y_test, knn_w.predict(X_test_scaled)))

plt.figure(figsize=(9, 5))
plt.plot(k_values, knn_unweighted_acc, label="uniform")
plt.plot(k_values, knn_weighted_acc, label="distance")
plt.xlabel("k")
plt.ylabel("Accuracy")
plt.title("kNN accuracy by neighbors")
plt.grid(alpha=0.3)
plt.legend()
plt.savefig(IMAGES_DIR / "knn_accuracy.png")
plt.show()

# %%

best_knn_mode = "uniform"
best_knn_k = k_values[int(np.argmax(knn_unweighted_acc))]
best_knn_acc = max(knn_unweighted_acc)
if max(knn_weighted_acc) > best_knn_acc:
    best_knn_mode = "distance"
    best_knn_k = k_values[int(np.argmax(knn_weighted_acc))]
    best_knn_acc = max(knn_weighted_acc)

best_knn = KNeighborsClassifier(n_neighbors=best_knn_k, weights=best_knn_mode)
best_knn.fit(X_train_scaled, y_train)
y_pred_knn = best_knn.predict(X_test_scaled)
y_score_knn = best_knn.predict_proba(X_test_scaled)

print(f"Best kNN params: k={best_knn_k}, weights={best_knn_mode}")
print("kNN metrics:", model_metrics(y_test, y_pred_knn, y_score_knn))

plot_decision_boundaries(
    best_knn,
    X_test_scaled,
    y_test,
    class_names,
    f"kNN decision boundaries (k={best_knn_k}, weights={best_knn_mode})",
    "knn_boundaries"
)

# %%

plot_confusion(y_test, y_pred_knn, class_names, "kNN confusion matrix", "knn_confusion.png")

# %%

plot_multiclass_roc(y_test_bin, y_score_knn, class_names, "kNN ROC curves", "knn_roc.png")

all_results["kNN"] = model_metrics(y_test, y_pred_knn, y_score_knn)


# %%
# 3. Logistic regression

logit_configs = [
    ("no_regularization", {"penalty": None, "solver": "lbfgs"}),
    ("l1", {"penalty": "l1", "solver": "saga", "C": 1.0}),
    ("l2", {"penalty": "l2", "solver": "lbfgs", "C": 1.0}),
]

logit_scores = {}
logit_models = {}

for name, params in logit_configs:
    model = LogisticRegression(
        random_state=RANDOM_STATE,
        max_iter=5000,
        **params
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    logit_scores[name] = acc
    logit_models[name] = model

plt.figure(figsize=(8, 5))
sns.barplot(x=list(logit_scores.keys()), y=list(logit_scores.values()))
plt.ylabel("Accuracy")
plt.title("Logistic regression: accuracy by regularization")
plt.ylim(0, 1)
plt.savefig(IMAGES_DIR / "logreg_accur_by_regular.png")
plt.show()

best_logit_name = max(logit_scores, key=logit_scores.get)
best_logit = logit_models[best_logit_name]
y_pred_logit = best_logit.predict(X_test_scaled)
y_score_logit = best_logit.predict_proba(X_test_scaled)

print(f"Best logistic config: {best_logit_name}")
print("Logistic metrics:", model_metrics(y_test, y_pred_logit, y_score_logit))

# %%

plot_decision_boundaries(
    best_logit,
    X_test_scaled,
    y_test,
    class_names,
    f"Logistic decision boundaries ({best_logit_name})",
    "logreg_boundaries.png"
)

# %%

plot_confusion(y_test, y_pred_logit, class_names, "Logistic confusion matrix", "logreg_confusion.png")

# %%

plot_multiclass_roc(y_test_bin, y_score_logit, class_names, "Logistic ROC curves", "logreg_roc.png")

all_results["LogisticRegression"] = model_metrics(y_test, y_pred_logit, y_score_logit)


# %%
# 4. SVM

svm_configs = [
    ("linear", {"kernel": "linear"}),
    ("poly_deg2", {"kernel": "poly", "degree": 2}),
    ("poly_deg3", {"kernel": "poly", "degree": 3}),
    ("poly_deg4", {"kernel": "poly", "degree": 4}),
    ("poly_deg5", {"kernel": "poly", "degree": 5}),
    ("rbf", {"kernel": "rbf"}),
]

svm_scores = {}
svm_models = {}

for name, params in svm_configs:
    model = SVC(
        probability=True,
        random_state=RANDOM_STATE,
        **params
    )
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    acc = accuracy_score(y_test, y_pred)
    svm_scores[name] = acc
    svm_models[name] = model

plt.figure(figsize=(8, 5))
sns.barplot(x=list(svm_scores.keys()), y=list(svm_scores.values()))
plt.ylabel("Accuracy")
plt.title("SVM: accuracy by kernel")
plt.ylim(0, 1)
plt.savefig(IMAGES_DIR / "svm_acc_by_kernel.png")
plt.show()

best_svm_name = max(svm_scores, key=svm_scores.get)
best_svm = svm_models[best_svm_name]
y_pred_svm = best_svm.predict(X_test_scaled)
y_score_svm = best_svm.predict_proba(X_test_scaled)

print(f"Best SVM config: {best_svm_name}")
print("SVM metrics:", model_metrics(y_test, y_pred_svm, y_score_svm))

# %%

plot_decision_boundaries(
    best_svm,
    X_test_scaled,
    y_test,
    class_names,
    f"SVM decision boundaries ({best_svm_name})",
    "svm_boundaries.png"
)

# %%

plot_confusion(y_test, y_pred_svm, class_names, "SVM confusion matrix", "svm_confusion.png")

# %%

plot_multiclass_roc(y_test_bin, y_score_svm, class_names, "SVM ROC curves", "svm_roc.png")

all_results["SVM"] = model_metrics(y_test, y_pred_svm, y_score_svm)


# %%
# 5. Decision tree

tree_grid = [
    {"criterion": "gini", "max_depth": 2, "max_leaf_nodes": 4},
    {"criterion": "gini", "max_depth": 3, "max_leaf_nodes": 6},
    {"criterion": "gini", "max_depth": 4, "max_leaf_nodes": 8},
    {"criterion": "entropy", "max_depth": 3, "max_leaf_nodes": 6},
    {"criterion": "entropy", "max_depth": 4, "max_leaf_nodes": 8},
]

best_tree = None
best_tree_cfg = None
best_tree_acc = -1.0

for cfg in tree_grid:
    model = DecisionTreeClassifier(random_state=RANDOM_STATE, **cfg)
    model.fit(X_train_scaled, y_train)
    acc = accuracy_score(y_test, model.predict(X_test_scaled))
    if acc > best_tree_acc:
        best_tree_acc = acc
        best_tree_cfg = cfg
        best_tree = model

y_pred_tree = best_tree.predict(X_test_scaled)
y_score_tree = best_tree.predict_proba(X_test_scaled)

print(f"Best tree config: {best_tree_cfg}")
print("Decision tree metrics:", model_metrics(y_test, y_pred_tree, y_score_tree))

# %%

plot_decision_boundaries(
    best_tree,
    X_test_scaled,
    y_test,
    class_names,
    "Decision tree boundaries",
    "dectree_boundaries.png"
)

# %%

plot_confusion(y_test, y_pred_tree, class_names, "Decision tree confusion matrix", "dectree_confusion.png")

# %%

plot_multiclass_roc(y_test_bin, y_score_tree, class_names, "Decision tree ROC curves", "dectree_roc.png")

# %%

plt.figure(figsize=(14, 8))
plot_tree(
    best_tree,
    feature_names=feature_cols,
    class_names=list(class_names),
    filled=True,
    rounded=True
)
plt.title("Best decision tree")
plt.savefig(IMAGES_DIR / "dectree_viz.png")
plt.show()

all_results["DecisionTree"] = model_metrics(y_test, y_pred_tree, y_score_tree)


# %%
# 6. Final classifier comparison

results_df = pd.DataFrame(all_results).T
results_df = results_df[["precision_macro", "recall_macro", "auc_ovr_macro", "accuracy", "f1_macro"]]
print(results_df.sort_values("auc_ovr_macro", ascending=False))

sns.heatmap(results_df, annot=True, fmt=".3f", cmap="YlGnBu")
plt.title("Classifier comparison")
plt.savefig(IMAGES_DIR / "classif_cmp.png", bbox_inches='tight')
plt.show()
