"""Full test-set evaluation for the models in construction_ml.ipynb.

Uses the same data, split and model settings as the notebook, then reports
accuracy, macro precision/recall/F1, ROC-AUC, a majority-class baseline,
confusion matrices and XGBoost feature importance. It also saves
confusion_matrix_xgboost.png.

Run from the repo root:  python evaluate.py
"""
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from lightgbm import LGBMClassifier
from matplotlib.colors import LinearSegmentedColormap
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (accuracy_score, confusion_matrix,
                             precision_recall_fscore_support, roc_auc_score)
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")
LABELS = ["Low", "Medium", "High"]  # Risk_Category_Encoded 0, 1, 2

# Same data and split as construction_ml.ipynb
df = pd.read_csv("./data/clean_data.csv")
y = df["Risk_Category_Encoded"]
X = df.drop(columns=["Risk_Category_Encoded"])
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
print(f"Data: {df.shape[0]} rows, {X.shape[1]} features")
print(f"Train: {len(X_train)}  Test: {len(X_test)}")
print("Test class counts:", dict(zip(LABELS, y_test.value_counts().sort_index())))

# Same model settings as the notebook (cells 3 and 6).
# SVC gets random_state so its probability estimates (used for ROC-AUC) are repeatable.
models = {
    "XGBoost": XGBClassifier(eval_metric="mlogloss", learning_rate=0.1, max_depth=6, n_estimators=300),
    "LightGBM": LGBMClassifier(learning_rate=0.05, num_leaves=31, n_estimators=300, verbose=-1),
    "Random Forest (100 trees)": RandomForestClassifier(n_estimators=100, random_state=42),
    "Random Forest (200 trees, depth 20)": RandomForestClassifier(n_estimators=200, max_depth=20, random_state=42),
    "SVM (RBF)": SVC(kernel="rbf", probability=True, random_state=42),
}

rows, matrices = [], {}
for name, model in models.items():
    model.fit(X_train, y_train)
    pred = model.predict(X_test)
    p, r, f, _ = precision_recall_fscore_support(y_test, pred, average="macro", zero_division=0)
    auc = roc_auc_score(y_test, model.predict_proba(X_test), multi_class="ovr", average="macro")
    rows.append({"Model": name, "Accuracy": accuracy_score(y_test, pred),
                 "Precision": p, "Recall": r, "F1": f, "ROC-AUC": auc})
    matrices[name] = confusion_matrix(y_test, pred)

majority = y_train.value_counts().idxmax()
rows.append({"Model": f'Baseline: always "{LABELS[majority]}"',
             "Accuracy": accuracy_score(y_test, np.full(len(y_test), majority))})

print("\nTest-set metrics (precision/recall/F1 macro, ROC-AUC one-vs-rest macro):")
print(pd.DataFrame(rows).round(3).to_string(index=False))

for name, cm in matrices.items():
    print(f"\nConfusion matrix: {name} (rows = actual, columns = predicted)")
    print(pd.DataFrame(cm, index=LABELS, columns=LABELS).to_string())

booster = models["XGBoost"].get_booster()
weight = pd.Series(booster.get_score(importance_type="weight")).sort_values(ascending=False)
print("\nXGBoost top 5 features (split count, same as chart2.png):")
print(weight.head(5).to_string())

# Confusion matrix image for the README
cm = matrices["XGBoost"]
cmap = LinearSegmentedColormap.from_list(
    "blue", ["#f7fafe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
fig, ax = plt.subplots(figsize=(5.6, 4.6), dpi=150)
ax.imshow(cm, cmap=cmap, vmin=0, vmax=cm.max())
for i in range(3):
    for j in range(3):
        v = cm[i, j]
        ax.text(j, i, str(v), ha="center", va="center", fontsize=15,
                color="#ffffff" if v > cm.max() * 0.45 else "#1f2328",
                fontweight="bold" if i == j else "normal")
ax.set_xticks(range(3), LABELS)
ax.set_yticks(range(3), LABELS)
ax.set_xlabel("Predicted risk level", color="#57606a")
ax.set_ylabel("Actual risk level", color="#57606a")
ax.set_title(f"XGBoost confusion matrix (test set, {len(y_test)} projects)",
             fontsize=11, color="#1f2328", loc="left")
ax.tick_params(length=0, colors="#1f2328")
for spine in ax.spines.values():
    spine.set_visible(False)
ax.set_xticks(np.arange(-0.5, 3), minor=True)
ax.set_yticks(np.arange(-0.5, 3), minor=True)
ax.grid(which="minor", color="#ffffff", linewidth=2)
ax.tick_params(which="minor", length=0)
fig.patch.set_facecolor("#ffffff")
fig.tight_layout()
fig.savefig("confusion_matrix_xgboost.png", facecolor="#ffffff")
print("\nSaved confusion_matrix_xgboost.png")
