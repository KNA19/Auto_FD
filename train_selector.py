# train_selector.py
# Step-3: Train an algorithm selector for {Eades, Adaptive-Eades, KK}
# (Fix 2: drop FR as a classification label, but keep FR in score evaluation)
#
# Uses graph features (feat_*) + family one-hot.
# Uses family-balanced GROUP split by instance_id inside each family.

import argparse
import os
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)
import joblib


SCORE_COL = {
    "Eades": "score_Eades",
    "Adaptive-Eades": "score_Adaptive-Eades",
    "FR": "score_FR",
    "KK": "score_KK",
}


def family_balanced_group_split(
    df: pd.DataFrame,
    y: pd.Series,
    group_col: str = "instance_id",
    family_col: str = "family",
    test_size: float = 0.2,
    seed: int = 7,
):
    train_indices = []
    test_indices = []

    for fam in sorted(df[family_col].dropna().unique()):
        sub = df[df[family_col] == fam]
        if len(sub) == 0:
            continue

        groups = sub[group_col].astype(str)
        n_groups = groups.nunique()

        # If a family has only 1 unique group, we can't split it; keep it in train.
        if n_groups < 2:
            train_indices.append(sub.index.values)
            continue

        eff_test_size = test_size
        if n_groups * test_size < 1:
            eff_test_size = 1.0 / n_groups

        gss = GroupShuffleSplit(n_splits=1, test_size=eff_test_size, random_state=seed)
        tr_idx, te_idx = next(gss.split(sub, y.loc[sub.index], groups=groups))

        train_indices.append(sub.index.values[tr_idx])
        test_indices.append(sub.index.values[te_idx])

    train_idx = np.concatenate(train_indices) if train_indices else np.array([], dtype=int)
    test_idx = np.concatenate(test_indices) if test_indices else np.array([], dtype=int)

    # Safety: remove overlap
    train_set = set(train_idx.tolist())
    test_idx = np.array([i for i in test_idx.tolist() if i not in train_set], dtype=int)

    return train_idx, test_idx


def relabel_fr_to_best_of_three(df: pd.DataFrame) -> pd.Series:
    """
    Fix 2:
    If best_algo == 'FR', replace it with best among {Eades, Adaptive-Eades, KK}
    based on the *true* scores in the row. Keeps the training target learnable.
    """
    required = ["best_algo", "score_Eades", "score_Adaptive-Eades", "score_KK"]
    for c in required:
        if c not in df.columns:
            raise ValueError(f"Missing required column for relabeling: {c}")

    def best_of_three(row):
        scores = {
            "Eades": float(row["score_Eades"]),
            "Adaptive-Eades": float(row["score_Adaptive-Eades"]),
            "KK": float(row["score_KK"]),
        }
        return min(scores.keys(), key=lambda k: scores[k])

    y = df["best_algo"].astype(str).copy()
    mask = (y == "FR")
    if mask.any():
        y.loc[mask] = df.loc[mask].apply(best_of_three, axis=1)

    return y


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="out/summary_wide.csv")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--outdir", type=str, default="out")
    ap.add_argument("--debug_features", action="store_true")
    args = ap.parse_args()

    df = pd.read_csv(args.data)

    # --- Basic checks ---
    for col in ["best_algo", "instance_id", "family"]:
        if col not in df.columns:
            raise ValueError(f"{col} column missing.")

    # Features: everything that starts with feat_ plus family one-hot
    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    if not feat_cols:
        raise ValueError("No feat_* columns found. Did you patch the generator to write features?")

    # ---- FIX 2: relabel FR winners for training target ----
    # y_train_target will be in {Eades, Adaptive-Eades, KK}
    y_target = relabel_fr_to_best_of_three(df)

    # Groups for leakage-free splitting
    groups = df["instance_id"].astype(str)

    # -------------------------
    # Fix 1 (already): Family-balanced group split
    # -------------------------
    train_idx, test_idx = family_balanced_group_split(
        df=df,
        y=y_target,
        group_col="instance_id",
        family_col="family",
        test_size=args.test_size,
        seed=args.seed,
    )

    if len(test_idx) == 0:
        raise RuntimeError(
            "Test split is empty. Try lowering --test_size (e.g., 0.1) or generating more graph instances."
        )

    df_train = df.loc[train_idx].reset_index(drop=True)
    df_test = df.loc[test_idx].reset_index(drop=True)

    # Training labels (3-class)
    y_train = relabel_fr_to_best_of_three(df_train)
    y_test = relabel_fr_to_best_of_three(df_test)

    # -------------------------
    # Diagnostics
    # -------------------------
    print("\n=== Diagnostics: Original label distributions (before FR drop) ===")
    print("Train original label counts:")
    print(df_train["best_algo"].value_counts(dropna=False))
    print("\nTest original label counts:")
    print(df_test["best_algo"].value_counts(dropna=False))

    print("\n=== Diagnostics: Training label distributions (after FR relabel) ===")
    print("Train target label counts:")
    print(y_train.value_counts(dropna=False))
    print("\nTest target label counts:")
    print(y_test.value_counts(dropna=False))

    print("\n=== Diagnostics: Wins by family (train, original) ===")
    print(pd.crosstab(df_train["family"], df_train["best_algo"], dropna=False))

    print("\n=== Diagnostics: Wins by family (test, original) ===")
    print(pd.crosstab(df_test["family"], df_test["best_algo"], dropna=False))

    # --- Build X with one-hot family ---
    X_train = pd.get_dummies(df_train[feat_cols + ["family"]], columns=["family"])
    X_test = pd.get_dummies(df_test[feat_cols + ["family"]], columns=["family"])
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    if args.debug_features:
        print("\n[DEBUG] Feature columns used:")
        for c in X_train.columns:
            print(" -", c)

    # Model
    clf = RandomForestClassifier(
        n_estimators=500,
        random_state=args.seed,
        n_jobs=-1,
        class_weight="balanced",
        max_depth=None,
        min_samples_leaf=2,
    )
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    print("\n=== Diagnostics: Predicted label counts (3-class) ===")
    print(pd.Series(y_pred).value_counts(dropna=False))

    # -------------------------
    # Classification metrics (3-class)
    # -------------------------
    acc = accuracy_score(y_test, y_pred)
    bacc = balanced_accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average="macro")

    print("\n=== Selector Classification Metrics (3-class, family-balanced group split) ===")
    print(f"Accuracy:          {acc:.4f}")
    print(f"Balanced Accuracy: {bacc:.4f}")
    print(f"Macro F1:          {macro_f1:.4f}")

    labels_sorted = sorted(set(y_test) | set(y_pred))
    cm = confusion_matrix(y_test, y_pred, labels=labels_sorted)
    print("\nConfusion Matrix (rows=true, cols=pred):")
    print(
        pd.DataFrame(
            cm,
            index=[f"true_{l}" for l in labels_sorted],
            columns=[f"pred_{l}" for l in labels_sorted],
        )
    )

    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, digits=4, zero_division=0))

    # -------------------------
    # Score-based evaluation (still uses all 4 algorithms)
    # -------------------------
    score_cols = [SCORE_COL[a] for a in SCORE_COL]
    for col in score_cols:
        if col not in df.columns:
            raise ValueError(f"Missing score column: {col}")

    # Best fixed baseline from TRAIN set (still among all 4)
    train_means = {algo: float(df_train[SCORE_COL[algo]].mean()) for algo in SCORE_COL}
    best_fixed_algo = min(train_means.keys(), key=lambda a: train_means[a])

    df_test_pred = df_test.copy()
    df_test_pred["pred_algo"] = y_pred  # predicted among 3 classes

    pred_scores = []
    oracle_scores = []
    fixed_scores = []

    for _, row in df_test_pred.iterrows():
        pred_algo = row["pred_algo"]
        if pred_algo not in SCORE_COL:
            pred_algo = best_fixed_algo

        pred_scores.append(float(row[SCORE_COL[pred_algo]]))
        oracle_scores.append(min(float(row[c]) for c in score_cols))
        fixed_scores.append(float(row[SCORE_COL[best_fixed_algo]]))

    mean_pred = sum(pred_scores) / len(pred_scores)
    mean_oracle = sum(oracle_scores) / len(oracle_scores)
    mean_fixed = sum(fixed_scores) / len(fixed_scores)

    print("\n=== Score-based Evaluation (lower is better; still 4-algo oracle) ===")
    print(f"Train mean scores: Eades={train_means['Eades']:.4f}, "
          f"Adaptive={train_means['Adaptive-Eades']:.4f}, FR={train_means['FR']:.4f}, KK={train_means['KK']:.4f}")
    print(f"Chosen best fixed baseline: {best_fixed_algo}")
    print(f"Avg score (predicted):           {mean_pred:.4f}")
    print(f"Avg score (best fixed baseline): {mean_fixed:.4f}")
    print(f"Avg score (oracle min of 4):     {mean_oracle:.4f}")
    print(f"Gap to oracle (pred - oracle):   {(mean_pred - mean_oracle):.4f}")
    print(f"Improvement vs fixed (fixed - pred): {(mean_fixed - mean_pred):.4f}")

    # -------------------------
    # Feature importances
    # -------------------------
    importances = list(zip(list(X_train.columns), clf.feature_importances_))
    importances.sort(key=lambda x: x[1], reverse=True)

    print("\nTop 15 Feature Importances:")
    for k, v in importances[:15]:
        print(f" - {k}: {v:.4f}")

    # -------------------------
    # Save model + metadata
    # -------------------------
    os.makedirs(args.outdir, exist_ok=True)
    model_path = os.path.join(args.outdir, "selector_rf_3class.joblib")
    joblib.dump(clf, model_path)

    meta = {
        "feature_columns": list(X_train.columns),
        "score_columns": SCORE_COL,
        "best_fixed_algo_train": best_fixed_algo,
        "train_score_means": train_means,
        "test_size": args.test_size,
        "seed": args.seed,
        "split": "family_balanced_group_split(instance_id within each family)",
        "target": "3-class (FR relabeled to best of {Eades, Adaptive-Eades, KK})",
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
    }
    meta_path = os.path.join(args.outdir, "selector_rf_3class_meta.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)

    print("\nSaved:")
    print(" - model:", os.path.abspath(model_path))
    print(" - meta: ", os.path.abspath(meta_path))


if __name__ == "__main__":
    main()