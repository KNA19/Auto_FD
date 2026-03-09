Below is an updated **README.md** you can paste directly. It keeps your original demo story, and adds the **Step-3 selector results + selector_viz plots** in the best narrative order (the same order I suggested for your meeting).

---

# AUTO_FD (v1)

## Overview

This repository contains a reproducible comparison of **four force-directed graph layout methods**, all starting from the **same random initial node positions**, plus a **first AutoFD selector prototype** that learns which algorithm to choose.

### Compared layout methods (portfolio)

* **Eades** (baseline cooling + movement cap)
* **Adaptive-Eades** (temperature updated from iteration dynamics + movement cap)
* **FR** (Fruchterman–Reingold)
* **KK** (Kamada–Kawai)

---

## Why this demo matters

This project is designed to show:

1. **Cooling / step-size control strongly affects Eades outcomes**
2. **Different graph families prefer different force-directed dynamics**
3. This motivates an **algorithm portfolio + selector/controller** end goal (**AutoFD**)
4. A first selector already improves average quality, but remaining errors motivate **probe (dynamic) features**

---

## Folder structure

### `out/` (generated outputs)

* **Panel images** (2×2 comparisons): `out/<graph_name>_seed<k>.png`
* **Convergence plots**: `out/convergence_<graph_name>_seed<k>.png`
* **Long summary**: `out/summary.csv` (one row per graph×seed×algo)
* **Wide summary**: `out/summary_wide.csv` (one row per graph×seed with scores + best_algo + features)
* **Run config**: `out/run_config.json`

### `out/selector_viz/` (generated selector visualizations)

The script `visualize_selector_stats.py` produces a set of plots used to explain the selector story:

* label distributions, wins-by-family heatmaps
* confusion matrix for the selector
* feature importances
* score comparison (predicted vs best fixed vs oracle)
* regret distribution and regret-by-family
* “why probing is needed” scatter plots (Eades vs Adaptive score difference)

---

## Output Files (generated in `out/`)

### A) Panel images (main layout comparison)

Pattern: `out/<graph_name>_seed<k>.png`

Each panel is a **2×2 comparison figure**:

* **Top-left:** Eades
* **Top-right:** Adaptive-Eades
* **Bottom-left:** FR
* **Bottom-right:** KK

Each subplot prints these metrics:

* `overlap`
* `edgeCV`
* `stress`
* `aspect` *(if enabled)*
* `score`

### B) Convergence plot (diagnostic)

Pattern: `out/convergence_<graph_name>_seed<k>.png`

This plots **max displacement per iteration** for:

* Eades baseline
* Adaptive-Eades

Useful for showing:

* how quickly the algorithm stabilizes
* whether it oscillates

### C) Summary tables (CSV)

**Long format:** `out/summary.csv`
One row per **(graph, init_seed, algo)** with columns:

* metadata: `graph`, `family`, `n_nodes`, `n_edges`, `graph_seed`, `graph_params`, `init_seed`, `stress_seed`, `instance_id`
* features: `feat_*`
* metrics: `overlap`, `edge_cv`, `stress`, `aspect`, `score`

**Wide format:** `out/summary_wide.csv`
One row per **(graph, init_seed)** with:

* metadata + `feat_*`
* `score_Eades`, `score_Adaptive-Eades`, `score_FR`, `score_KK`
* `best_algo` (oracle label under current score definition)

---

## Metrics (short interpretation)

### `overlap`

Penalty for node-node overlaps after normalizing the drawing.
**Lower is better.**
Large overlap usually means the layout collapsed into a blob/line.

### `edgeCV`

Coefficient of variation of edge lengths (`std / mean`).
**Lower is better.**
Lower edgeCV often looks cleaner.

### `stress`

Sampled distance-preservation error between Euclidean distances and shortest-path distances.
**Lower is better** for distance preservation.

> Note: stress is not a pure aesthetic metric by itself (degenerate layouts can still have low stress).

### `aspect`

A simple penalty for extreme bounding-box aspect ratios (optional).
Helps discourage “long skinny” drawings.

### `score`

A combined score for quick ranking:

`score = W_OVERLAP * overlap + W_EDGECV * edgeCV + W_STRESS * stress + W_ASPECT * aspect`

**Lower is better.**
Overlap is weighted heavily (**readability first**).

---

## Results Gallery (Example Panels)

### Main 2×2 comparison panels (seed 0 examples)

#### `grid_10x10_seed0`

![grid panel](out/grid_10x10_seed0.png)

#### `tree_120_seed0`

![tree panel](out/tree_120_seed0.png)

#### `erdos_120_p004_seed0`

![erdos panel](out/erdos_120_p004_seed0.png)

#### `ba_150_m2_seed0`

![ba panel](out/ba_150_m2_seed0.png)

#### `ws_150_k4_p020_seed0`

![ws panel](out/ws_150_k4_p020_seed0.png)

---

## Step-3: AutoFD Selector (v1)

We train a lightweight selector that predicts which algorithm to run (initially trained as a **3-class problem**: Eades vs Adaptive-Eades vs KK).
FR is kept as a portfolio baseline but is not learned as a class because it rarely wins under the current score.

**Key idea:** graph families differ, and static graph features alone cannot fully resolve Eades vs Adaptive-Eades — motivating probe/dynamic features.

### Selector visualization outputs

All the following plots are saved in:

`out/selector_viz/`

---

# Recommended Meeting Story Order (best narrative)

Use this order to tell a clear “why selection + why probing” story.

---

## 1) Different graphs prefer different algorithms (selection is necessary)

### Wins by family (train)

![wins by family train](out/selector_viz/03_wins_by_family_train.png)

### Wins by family (test)

![wins by family test](out/selector_viz/04_wins_by_family_test.png)

**Message:** there is no single best algorithm across families.

---

## 2) Why FR is not learned as a class (rare winner)

### Original best_algo counts (train vs test)

![label counts original](out/selector_viz/01_label_counts_original.png)

### 3-class target counts (after FR relabel)

![label counts 3class](out/selector_viz/02_label_counts_3class.png)

**Message:** FR rarely wins under the current objective → keep as baseline, but don’t learn as a class.

---

## 3) AutoFD already improves average score vs fixed baselines

### Avg score comparison (predicted vs fixed vs oracle)

![score comparison](out/selector_viz/08_score_comparison.png)

**Message:** selector beats “best fixed” and moves toward oracle.

---

## 4) Remaining error pattern: hard case is Eades vs Adaptive-Eades

### Confusion matrix (3-class selector)

![confusion matrix](out/selector_viz/06_confusion_matrix_3class.png)

### Predicted vs true label counts

![pred vs true counts](out/selector_viz/05_predicted_vs_true_counts.png)

**Message:** KK is easy; Eades vs Adaptive is the difficult boundary.

---

## 5) Static features are not enough (motivation for probing)

### Feature importances

![feature importance](out/selector_viz/07_feature_importance.png)

**Message:** selector relies mostly on size/density/degree statistics → missing dynamics signals.

---

## 6) Strong evidence for probe features: static features cannot separate Eades vs Adaptive

These plots show the score difference:

`score_diff = score_Eades - score_Adaptive`

Negative = Eades better, Positive = Adaptive better.

### score difference vs density

![score diff vs density](out/selector_viz/11_score_diff_vs_density.png)

### score difference vs n

![score diff vs n](out/selector_viz/12_score_diff_vs_n.png)

**Message:** at similar n/density, the winner flips → need probe/dynamic features.

---

## 7) Residual regret motivates the next step (probe features reduce tail)

### Regret distribution

![regret hist](out/selector_viz/09_regret_hist.png)

### Regret by family

![regret by family](out/selector_viz/10_regret_by_family_boxplot.png)

**Message:** most cases are near oracle, but outliers remain — target for probing/dynamics.
---

## Proposed End Goal (Research Direction)

### AutoFD: Automatic Force-Directed Controller / Selector

Given a graph `G` and an initial layout, automatically choose:

* **which force-directed algorithm to run**, and/or
* the **best parameter preset / temperature schedule**

to produce an aesthetically readable layout under a fixed budget.