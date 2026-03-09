import os
import math
import argparse
import json

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import GroupShuffleSplit
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
)

SCORE_COL = {
    "Eades": "score_Eades",
    "Adaptive-Eades": "score_Adaptive-Eades",
    "FR": "score_FR",
    "KK": "score_KK",
}

ORDER4 = ["Eades", "Adaptive-Eades", "FR", "KK"]
ORDER3 = ["Eades", "Adaptive-Eades", "KK"]


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


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

        # if not enough groups, keep in train (can't split)
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
    If best_algo == 'FR', relabel to best among {Eades, Adaptive-Eades, KK}
    using the *true* scores in that row.
    """
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


def plot_bar_counts(title, counts_a, label_a, counts_b, label_b, outpath):
    # align indices
    all_idx = sorted(set(counts_a.index) | set(counts_b.index))
    a = np.array([counts_a.get(k, 0) for k in all_idx], dtype=float)
    b = np.array([counts_b.get(k, 0) for k in all_idx], dtype=float)

    x = np.arange(len(all_idx))
    w = 0.38

    plt.figure(figsize=(10, 5))
    plt.bar(x - w/2, a, width=w, label=label_a)
    plt.bar(x + w/2, b, width=w, label=label_b)
    plt.xticks(x, all_idx, rotation=0)
    plt.ylabel("Count")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_crosstab_heatmap(ct: pd.DataFrame, title: str, outpath: str):
    # simple heatmap via imshow
    data = ct.values.astype(float)
    plt.figure(figsize=(10, 4.8))
    plt.imshow(data, aspect="auto")
    plt.colorbar(label="Count")
    plt.xticks(np.arange(ct.shape[1]), ct.columns.tolist(), rotation=0)
    plt.yticks(np.arange(ct.shape[0]), ct.index.tolist())
    plt.title(title)
    # annotate cells
    for i in range(ct.shape[0]):
        for j in range(ct.shape[1]):
            plt.text(j, i, str(int(data[i, j])), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_confusion(cm: np.ndarray, labels: list[str], title: str, outpath: str):
    plt.figure(figsize=(6.8, 5.8))
    plt.imshow(cm.astype(float), aspect="auto")
    plt.colorbar(label="Count")
    plt.xticks(np.arange(len(labels)), labels, rotation=0)
    plt.yticks(np.arange(len(labels)), labels)
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, str(int(cm[i, j])), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_feature_importance(names, vals, title, outpath, topk=15):
    # barh top-k
    order = np.argsort(vals)[::-1][:topk]
    sel_names = [names[i] for i in order][::-1]
    sel_vals = [vals[i] for i in order][::-1]

    plt.figure(figsize=(9, 6))
    plt.barh(sel_names, sel_vals)
    plt.xlabel("Importance")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_score_bars(avg_pred, avg_fixed, avg_oracle, title, outpath):
    labels = ["Predicted", "Best Fixed", "Oracle"]
    vals = [avg_pred, avg_fixed, avg_oracle]
    x = np.arange(len(labels))

    plt.figure(figsize=(7, 4.5))
    plt.bar(x, vals)
    plt.xticks(x, labels)
    plt.ylabel("Avg score (lower is better)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_hist(data, title, xlabel, outpath, bins=30):
    plt.figure(figsize=(8, 4.5))
    plt.hist(data, bins=bins)
    plt.title(title)
    plt.xlabel(xlabel)
    plt.ylabel("Count")
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_box_by_family(df_test_pred: pd.DataFrame, value_col: str, title: str, outpath: str):
    fams = sorted(df_test_pred["family"].unique())
    data = [df_test_pred[df_test_pred["family"] == f][value_col].values for f in fams]

    plt.figure(figsize=(9, 5))
    plt.boxplot(data, showfliers=True)
    plt.ylabel(value_col)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def plot_scatter(df: pd.DataFrame, xcol: str, ycol: str, title: str, outpath: str):
    x = pd.to_numeric(df[xcol], errors="coerce").to_numpy(dtype=float)
    y = pd.to_numeric(df[ycol], errors="coerce").to_numpy(dtype=float)
    mask = np.isfinite(x) & np.isfinite(y)

    plt.figure(figsize=(8, 4.8))
    plt.scatter(x[mask], y[mask], s=18)
    plt.xlabel(xcol)
    plt.ylabel(ycol)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath, dpi=160)
    plt.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", type=str, default="out/summary_wide.csv")
    ap.add_argument("--outdir", type=str, default="out/selector_viz")
    ap.add_argument("--test_size", type=float, default=0.2)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--stress_pairs_note", type=str, default="")
    args = ap.parse_args()

    ensure_dir(args.outdir)

    df = pd.read_csv(args.data)

    # checks
    for c in ["best_algo", "instance_id", "family"]:
        if c not in df.columns:
            raise ValueError(f"Missing required column: {c}")
    for c in SCORE_COL.values():
        if c not in df.columns:
            raise ValueError(f"Missing score column: {c}")

    feat_cols = [c for c in df.columns if c.startswith("feat_")]
    if not feat_cols:
        raise ValueError("No feat_* columns found. Regenerate summary_wide.csv with features.")

    # Fix 2 labels
    y_original = df["best_algo"].astype(str)
    y_target = relabel_fr_to_best_of_three(df)  # 3-class labels

    # Fix 1 split (family-balanced, group by instance_id)
    train_idx, test_idx = family_balanced_group_split(
        df=df,
        y=y_target,
        group_col="instance_id",
        family_col="family",
        test_size=args.test_size,
        seed=args.seed,
    )
    if len(test_idx) == 0:
        raise RuntimeError("Empty test split. Increase benchmark size or reduce --test_size.")

    df_train = df.loc[train_idx].reset_index(drop=True)
    df_test = df.loc[test_idx].reset_index(drop=True)

    # distributions (original)
    train_counts_orig = df_train["best_algo"].value_counts()
    test_counts_orig = df_test["best_algo"].value_counts()

    # distributions (3-class after FR relabel)
    y_train_3 = relabel_fr_to_best_of_three(df_train)
    y_test_3 = relabel_fr_to_best_of_three(df_test)
    train_counts_3 = y_train_3.value_counts()
    test_counts_3 = y_test_3.value_counts()

    # wins by family (original)
    ct_train = pd.crosstab(df_train["family"], df_train["best_algo"])
    ct_test = pd.crosstab(df_test["family"], df_test["best_algo"])

    # build X
    X_train = pd.get_dummies(df_train[feat_cols + ["family"]], columns=["family"])
    X_test = pd.get_dummies(df_test[feat_cols + ["family"]], columns=["family"])
    X_test = X_test.reindex(columns=X_train.columns, fill_value=0)

    # train RF (3-class)
    clf = RandomForestClassifier(
        n_estimators=500,
        random_state=args.seed,
        n_jobs=-1,
        class_weight="balanced",
        max_depth=None,
        min_samples_leaf=2,
    )
    clf.fit(X_train, y_train_3)
    y_pred = clf.predict(X_test)

    # confusion (3-class)
    labels3 = ORDER3
    cm3 = confusion_matrix(y_test_3, y_pred, labels=labels3)

    # classification metrics
    acc = accuracy_score(y_test_3, y_pred)
    bacc = balanced_accuracy_score(y_test_3, y_pred)
    mf1 = f1_score(y_test_3, y_pred, average="macro")

    # score-based eval (still uses all 4 scores for oracle & fixed)
    train_means = {a: float(df_train[SCORE_COL[a]].mean()) for a in ORDER4}
    best_fixed_algo = min(train_means.keys(), key=lambda k: train_means[k])

    df_test_pred = df_test.copy()
    df_test_pred["pred_algo"] = y_pred

    pred_scores = []
    oracle_scores = []
    fixed_scores = []

    for _, row in df_test_pred.iterrows():
        pa = row["pred_algo"]
        pred_scores.append(float(row[SCORE_COL[pa]]))
        oracle_scores.append(min(float(row[SCORE_COL[a]]) for a in ORDER4))
        fixed_scores.append(float(row[SCORE_COL[best_fixed_algo]]))

    pred_scores = np.array(pred_scores, dtype=float)
    oracle_scores = np.array(oracle_scores, dtype=float)
    fixed_scores = np.array(fixed_scores, dtype=float)

    regret = pred_scores - oracle_scores
    improvement = fixed_scores - pred_scores

    # ---- SAVE FIGURES ----
    plot_bar_counts(
        "Original best_algo counts (train vs test)",
        train_counts_orig, "Train",
        test_counts_orig, "Test",
        os.path.join(args.outdir, "01_label_counts_original.png"),
    )

    plot_bar_counts(
        "3-class target counts (after FR relabel) (train vs test)",
        train_counts_3, "Train",
        test_counts_3, "Test",
        os.path.join(args.outdir, "02_label_counts_3class.png"),
    )

    plot_crosstab_heatmap(
        ct_train.reindex(index=sorted(ct_train.index)),
        "Wins by family (train, original labels)",
        os.path.join(args.outdir, "03_wins_by_family_train.png"),
    )

    plot_crosstab_heatmap(
        ct_test.reindex(index=sorted(ct_test.index)),
        "Wins by family (test, original labels)",
        os.path.join(args.outdir, "04_wins_by_family_test.png"),
    )

    plot_bar_counts(
        "Predicted label counts (3-class)",
        pd.Series(y_pred).value_counts(), "Predicted",
        y_test_3.value_counts(), "True",
        os.path.join(args.outdir, "05_predicted_vs_true_counts.png"),
    )

    plot_confusion(
        cm3,
        labels3,
        "Confusion matrix (3-class selector)",
        os.path.join(args.outdir, "06_confusion_matrix_3class.png"),
    )

    plot_feature_importance(
        list(X_train.columns),
        clf.feature_importances_,
        "Top feature importances (RandomForest selector)",
        os.path.join(args.outdir, "07_feature_importance.png"),
        topk=15,
    )

    plot_score_bars(
        float(pred_scores.mean()),
        float(fixed_scores.mean()),
        float(oracle_scores.mean()),
        "Avg score comparison (predicted vs fixed vs oracle)",
        os.path.join(args.outdir, "08_score_comparison.png"),
    )

    plot_hist(
        regret,
        "Regret distribution (predicted score - oracle score)",
        "Regret (lower is better; 0 is perfect)",
        os.path.join(args.outdir, "09_regret_hist.png"),
        bins=30,
    )

    plot_box_by_family(
        df_test_pred.assign(regret=regret),
        "regret",
        "Regret by family (why we need probing/dynamics)",
        os.path.join(args.outdir, "10_regret_by_family_boxplot.png"),
    )

    # show the core confusion driver: score difference between Eades and Adaptive
    df_test_pred = df_test_pred.assign(
        score_diff_E_minus_A=(df_test_pred["score_Eades"] - df_test_pred["score_Adaptive-Eades"])
    )
    # scatter against density and n (static features may not separate well)
    plot_scatter(
        df_test_pred,
        "feat_density",
        "score_diff_E_minus_A",
        "Eades vs Adaptive score difference vs density (overlap ⇒ need probe features)",
        os.path.join(args.outdir, "11_score_diff_vs_density.png"),
    )

    plot_scatter(
        df_test_pred,
        "feat_n",
        "score_diff_E_minus_A",
        "Eades vs Adaptive score difference vs n (overlap ⇒ need probe features)",
        os.path.join(args.outdir, "12_score_diff_vs_n.png"),
    )

    # ---- Print a short summary for your professor ----
    print("\n=== Visualization Summary ===")
    print("Train/Test sizes:", len(df_train), len(df_test))
    print(f"3-class Accuracy={acc:.4f}, BalancedAcc={bacc:.4f}, MacroF1={mf1:.4f}")
    print("\nScore-based (lower is better):")
    print(f" - Best fixed algo (train): {best_fixed_algo}  (means: {train_means})")
    print(f" - Avg predicted score: {pred_scores.mean():.4f}")
    print(f" - Avg oracle score:    {oracle_scores.mean():.4f}")
    print(f" - Avg regret:          {regret.mean():.4f}")
    print(f" - Avg improvement vs fixed: {improvement.mean():.4f}")
    print("\nSaved figures to:", os.path.abspath(args.outdir))

    # save a small JSON report too
    report = {
        "n_train": int(len(df_train)),
        "n_test": int(len(df_test)),
        "acc": float(acc),
        "balanced_acc": float(bacc),
        "macro_f1": float(mf1),
        "best_fixed_algo": best_fixed_algo,
        "train_mean_scores": train_means,
        "avg_pred_score": float(pred_scores.mean()),
        "avg_oracle_score": float(oracle_scores.mean()),
        "avg_regret": float(regret.mean()),
        "avg_improvement_vs_fixed": float(improvement.mean()),
        "notes": "If Eades vs Adaptive score differences overlap heavily vs static features, add probe/dynamic features.",
    }
    with open(os.path.join(args.outdir, "viz_report.json"), "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)


if __name__ == "__main__":
    main()