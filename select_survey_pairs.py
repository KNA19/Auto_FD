import os
import argparse
import shutil
from typing import Dict, List

import numpy as np
import pandas as pd

# Pillow is used only to swap left/right halves (fast & simple).
# If you don't have it: pip install pillow
from PIL import Image


EXPECTED_PAIR_TYPES = [
    "Adaptive-Eades__Eades",
    "Eades__FR",
    "Eades__KK",
    "Adaptive-Eades__FR",
    "Adaptive-Eades__KK",
    "FR__KK",
]

ALGOS = ["Eades", "Adaptive-Eades", "FR", "KK"]


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def pair_type(a: str, b: str) -> str:
    x, y = sorted([str(a), str(b)])
    return f"{x}__{y}"


def allocate_counts(total: int, keys: List[str]) -> Dict[str, int]:
    keys = list(keys)
    if not keys:
        return {}
    base = total // len(keys)
    rem = total % len(keys)
    out = {k: base for k in keys}
    for i in range(rem):
        out[keys[i]] += 1
    return out


def swap_pair_fields(row: Dict) -> Dict:
    """
    Swap left/right metadata fields in the manifest row dict.
    """
    swap_keys = [
        ("left_algo", "right_algo"),
        ("left_score", "right_score"),
        ("left_overlap", "right_overlap"),
        ("left_edge_cv", "right_edge_cv"),
        ("left_stress", "right_stress"),
        ("left_aspect", "right_aspect"),
        ("left_label", "right_label"),
    ]
    for a, b in swap_keys:
        if a in row and b in row:
            row[a], row[b] = row[b], row[a]
    return row


def swap_image_halves(src_path: str, dst_path: str) -> None:
    """
    Swap the left/right halves of an A/B side-by-side PNG.
    Assumes a vertical split at width//2.
    """
    img = Image.open(src_path)
    w, h = img.size
    mid = w // 2
    left = img.crop((0, 0, mid, h))
    right = img.crop((mid, 0, w, h))

    out = Image.new(img.mode, (w, h))
    out.paste(right, (0, 0))
    out.paste(left, (mid, 0))
    out.save(dst_path)


def select_balanced_from_subset(sub: pd.DataFrame, n: int, hard_frac: float, seed: int) -> pd.DataFrame:
    """
    Select n rows from sub with a mix of hard (small score_gap) and anchor (large score_gap),
    attempting to balance families.
    """
    if n <= 0 or len(sub) == 0:
        return sub.iloc[0:0].copy()

    n_hard = int(round(n * hard_frac))
    n_anchor = max(0, n - n_hard)

    fams = sorted(sub["family"].dropna().unique().tolist())
    if not fams:
        fams = ["unknown"]

    def pick_per_family(cand: pd.DataFrame, need: int, ascending: bool) -> pd.DataFrame:
        if need <= 0 or len(cand) == 0:
            return cand.iloc[0:0].copy()

        cand_sorted = cand.sort_values("score_gap", ascending=ascending)
        per_fam = allocate_counts(need, fams)

        picks = []
        used = set()
        for fam in fams:
            take = per_fam.get(fam, 0)
            if take <= 0:
                continue
            fam_rows = cand_sorted[cand_sorted["family"] == fam]
            if len(fam_rows) == 0:
                continue
            part = fam_rows.head(take)
            picks.append(part)
            used.update(part.index.tolist())

        picked = pd.concat(picks, axis=0) if picks else cand_sorted.iloc[0:0].copy()

        rem_need = need - len(picked)
        if rem_need > 0:
            leftover = cand_sorted.drop(index=list(used), errors="ignore")
            topup = leftover.head(rem_need)
            picked = pd.concat([picked, topup], axis=0)

        return picked

    hard_pick = pick_per_family(sub, n_hard, ascending=True)
    remaining = sub.drop(index=hard_pick.index, errors="ignore")
    anchor_pick = pick_per_family(remaining, n_anchor, ascending=False)

    picked = pd.concat([hard_pick, anchor_pick], axis=0)
    picked = picked.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    if len(picked) < n:
        need = n - len(picked)
        leftover = sub.drop(index=picked.index, errors="ignore")
        topup = leftover.sort_values("score_gap", ascending=True).head(need)
        picked = pd.concat([picked, topup], axis=0)

    if len(picked) > n:
        picked = picked.sample(n=n, random_state=seed).reset_index(drop=True)

    return picked


def compute_targets_left(selected: pd.DataFrame) -> Dict[str, int]:
    """
    Target left counts = roughly half of total appearances per algo in the selected set.
    """
    appearances = {a: 0 for a in ALGOS}
    for _, r in selected.iterrows():
        appearances[str(r["left_algo"])] += 1
        appearances[str(r["right_algo"])] += 1

    # Target = round(appearances/2). Use floor for stability; remainder will be handled greedily.
    target_left = {a: appearances[a] // 2 for a in ALGOS}
    return target_left


def choose_flip(la: str, ra: str, cur_left: Dict[str, int], target_left: Dict[str, int]) -> bool:
    """
    Decide whether to flip (swap left/right) to improve left-balance.
    Greedy decision that minimizes deviation from target.
    """
    algos = ALGOS

    def deviation(cur):
        return sum(abs(cur[a] - target_left[a]) for a in algos)

    # Option 1: keep
    cur_keep = cur_left.copy()
    cur_keep[la] += 1

    # Option 2: flip
    cur_flip = cur_left.copy()
    cur_flip[ra] += 1

    dev_keep = deviation(cur_keep)
    dev_flip = deviation(cur_flip)

    if dev_flip < dev_keep:
        return True
    if dev_keep < dev_flip:
        return False

    # tie-break: if left algo already above target and right below, flip
    if cur_left[la] >= target_left[la] and cur_left[ra] < target_left[ra]:
        return True
    return False


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", default="out/human_eval_pairs/pairs_manifest.csv")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--hard_frac", type=float, default=0.6)
    ap.add_argument("--out_dir", default="out/human_eval_pairs/survey_set")
    ap.add_argument("--copy_images", action="store_true", default=True)
    ap.add_argument("--pool_dir", default=None, help="Folder containing pool images; default = manifest folder")
    ap.add_argument("--counterbalance_lr", action="store_true", default=True)
    args = ap.parse_args()

    rng_seed = args.seed
    df = pd.read_csv(args.manifest)

    required_cols = [
        "pair_id", "image_file", "family",
        "left_algo", "right_algo",
        "left_score", "right_score",
    ]
    for c in required_cols:
        if c not in df.columns:
            raise ValueError(f"Missing required column in manifest: {c}")

    df["pair_type"] = df.apply(lambda r: pair_type(r["left_algo"], r["right_algo"]), axis=1)
    df["score_gap"] = (df["left_score"] - df["right_score"]).abs()
    df = df.dropna(subset=["family", "image_file", "pair_type", "score_gap"]).copy()

    avail_pair_types = sorted(df["pair_type"].unique().tolist())
    ordered_pair_types = [p for p in EXPECTED_PAIR_TYPES if p in avail_pair_types]
    for p in avail_pair_types:
        if p not in ordered_pair_types:
            ordered_pair_types.append(p)

    target_per_pair = allocate_counts(args.n, ordered_pair_types)

    selected_chunks = []
    used_pair_ids = set()

    for pt in ordered_pair_types:
        target = target_per_pair.get(pt, 0)
        if target <= 0:
            continue
        sub = df[df["pair_type"] == pt].copy()
        sub = sub[~sub["pair_id"].isin(used_pair_ids)]
        if len(sub) == 0:
            continue

        picked = select_balanced_from_subset(sub, target, args.hard_frac, seed=rng_seed)
        picked = picked[~picked["pair_id"].isin(used_pair_ids)]
        used_pair_ids.update(picked["pair_id"].tolist())
        selected_chunks.append(picked)

    selected = pd.concat(selected_chunks, axis=0) if selected_chunks else df.iloc[0:0].copy()

    # Top up or trim to exactly N
    if len(selected) < args.n:
        need = args.n - len(selected)
        remaining = df[~df["pair_id"].isin(used_pair_ids)].sort_values("score_gap", ascending=True)
        topup = remaining.head(need)
        selected = pd.concat([selected, topup], axis=0)
        used_pair_ids.update(topup["pair_id"].tolist())

    if len(selected) > args.n:
        selected = selected.sample(n=args.n, random_state=rng_seed).reset_index(drop=True)

    # Final shuffle for survey order
    selected = selected.sample(frac=1.0, random_state=rng_seed).reset_index(drop=True)

    # Output dirs
    out_root = args.out_dir
    ensure_dir(out_root)
    out_img_dir = os.path.join(out_root, "images")
    ensure_dir(out_img_dir)

    pool_dir = args.pool_dir if args.pool_dir is not None else os.path.dirname(os.path.abspath(args.manifest))

    # Counterbalance setup
    target_left = compute_targets_left(selected)
    cur_left = {a: 0 for a in ALGOS}

    # Copy/rename images -> q0001.png etc, with optional flipping
    survey_rows = []

    for i, (_, row) in enumerate(selected.iterrows()):
        qid = i + 1
        src = os.path.join(pool_dir, str(row["image_file"]))
        dst_name = f"q{qid:04d}.png"
        dst = os.path.join(out_img_dir, dst_name)

        if not os.path.exists(src):
            raise FileNotFoundError(f"Missing source image: {src}")

        rdict = row.to_dict()
        rdict["question_id"] = qid
        rdict["survey_image"] = dst_name
        rdict["left_label"] = "A"
        rdict["right_label"] = "B"

        la = str(rdict["left_algo"])
        ra = str(rdict["right_algo"])

        do_flip = False
        if args.counterbalance_lr:
            do_flip = choose_flip(la, ra, cur_left, target_left)

        if args.copy_images:
            if do_flip:
                # swap image halves and swap row metadata
                swap_image_halves(src, dst)
                rdict = swap_pair_fields(rdict)
                # after flip, left algo is what was previously right
                la_final = str(rdict["left_algo"])
                cur_left[la_final] += 1
            else:
                shutil.copy2(src, dst)
                cur_left[la] += 1
        else:
            # no copy mode: still update balance counts assuming no flip
            cur_left[la] += 1

        survey_rows.append(rdict)

    selected_key = pd.DataFrame(survey_rows)

    # Save full key (private: includes algorithms/scores)
    key_path = os.path.join(out_root, "survey_pairs_key.csv")
    selected_key.to_csv(key_path, index=False)

    # Save blinded file (safe to share)
    blinded_cols = ["question_id", "survey_image"]
    blinded_path = os.path.join(out_root, "survey_pairs_blinded.csv")
    selected_key[blinded_cols].to_csv(blinded_path, index=False)

    # Save image list
    img_list_path = os.path.join(out_root, "survey_images.txt")
    with open(img_list_path, "w", encoding="utf-8") as f:
        for fn in selected_key["survey_image"].tolist():
            f.write(fn + "\n")

    # Print summaries
    print("Saved:")
    print(" - key csv:     ", os.path.abspath(key_path))
    print(" - blinded csv: ", os.path.abspath(blinded_path))
    print(" - image list:  ", os.path.abspath(img_list_path))
    print(" - images dir:  ", os.path.abspath(out_img_dir))

    print("\nCounts by pair_type:")
    print(selected_key["pair_type"].value_counts())

    print("\nCounts by family:")
    print(selected_key["family"].value_counts())

    # Left/right balance report
    left_counts = selected_key["left_algo"].value_counts().to_dict()
    appearances = {a: 0 for a in ALGOS}
    for _, r in selected_key.iterrows():
        appearances[str(r["left_algo"])] += 1
        appearances[str(r["right_algo"])] += 1

    print("\nLeft-side counts (A-side) by algorithm:")
    for a in ALGOS:
        print(f" - {a}: {left_counts.get(a, 0)}  (target ~{appearances[a]//2})")

    print("\nTotal appearances (left+right) by algorithm:")
    for a in ALGOS:
        print(f" - {a}: {appearances[a]}")

    med_gap = float(selected_key["score_gap"].median())
    print(f"\nMedian score_gap in selected set: {med_gap:.4f}")
    print("Tip: smaller gaps are harder; larger gaps are anchor/easy comparisons.")


if __name__ == "__main__":
    main()