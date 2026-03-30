import os
import csv
import math
import random

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import demo_layouts as dl  # your existing script (must have if __name__ == "__main__": main())


# ------------------------------------------------------------
# Human-eval presets (portfolio-ready)
# ------------------------------------------------------------
ITERS = 1000

# Eades baseline
EADES_EPS = 0.005
EADES_COOL = 0.985
EADES_T0 = 2.5

# Adaptive-Eades
ADAP_EPS = 0.01
ADAP_RATE = 0.05
ADAP_T0 = 2.5
ADAP_TMIN = 1e-3

# FR
FR_ITERS = 1000

# init position range (use [-1,1] for better stability)
INIT_LOW = -1.0
INIT_HIGH = 1.0

# Stress sampling for manifest scores (keep moderate)
STRESS_PAIRS = 600


# All pairwise comparisons among your 4 algorithms (6 pairs)
PAIR_LIST = [
    ("Eades", "Adaptive-Eades"),
    ("Eades", "FR"),
    ("Eades", "KK"),
    ("Adaptive-Eades", "FR"),
    ("Adaptive-Eades", "KK"),
    ("FR", "KK"),
]


def ensure_dir(p):
    os.makedirs(p, exist_ok=True)


def mean_init_edge_length(G, init_pos):
    """Mean Euclidean edge length in the initial layout. Used as ideal spring length."""
    lens = []
    for u, v in G.edges():
        pu = init_pos[u]
        pv = init_pos[v]
        lens.append(math.hypot(pu[0] - pv[0], pu[1] - pv[1]))
    if not lens:
        return 1.0
    return float(np.mean(lens))


def run_all_algos_clean(G, init_pos, init_seed):
    """
    Runs Eades, Adaptive-Eades, FR, KK with clean presets.
    Returns:
      layouts: algo -> normalized pos
      metrics: algo -> (ov, cv, st, ar, score)
    """
    nodes = list(G.nodes())
    adj = {u: list(G.neighbors(u)) for u in nodes}

    # Make Eades scale more stable across graphs/sizes
    ideal_L = mean_init_edge_length(G, init_pos)

    # Eades baseline
    eades_pos, _ = dl.eades_baseline(
        adj, init_pos,
        iterations=ITERS,
        epsilon=EADES_EPS,
        cooling_factor=EADES_COOL,
        T0=EADES_T0,
        ideal_spring_length=ideal_L
    )

    # Adaptive-Eades
    adap_pos, _ = dl.eades_adaptive(
        adj, init_pos,
        iterations=ITERS,
        epsilon=ADAP_EPS,
        rate=ADAP_RATE,
        T0=ADAP_T0,
        Tmin=ADAP_TMIN,
        ideal_spring_length=ideal_L
    )

    # FR (make seed tied to init_seed; set k explicitly)
    n = G.number_of_nodes()
    k = 1.0 / math.sqrt(max(n, 1))
    fr_pos = nx.spring_layout(
        G, pos=init_pos, seed=init_seed, iterations=FR_ITERS, k=k
    )

    # KK
    kk_pos = nx.kamada_kawai_layout(G, pos=init_pos)

    layouts_raw = {
        "Eades": eades_pos,
        "Adaptive-Eades": adap_pos,
        "FR": fr_pos,
        "KK": kk_pos,
    }

    # Normalize once for fair visuals + fair metrics
    layouts = {a: dl.normalize_pos_dict(layouts_raw[a]) for a in layouts_raw}

    # Deterministic stress seed per (graph, init_seed)
    # Use init_seed here; if you want stronger determinism include graph id too.
    stress_seed = 9999 + init_seed

    metrics = {
        a: dl.compute_metrics(G, layouts[a], stress_pairs=STRESS_PAIRS, seed=stress_seed)
        for a in layouts
    }

    return layouts, metrics


def plot_pair_image(out_png, G, pos_left, pos_right, label_left="A", label_right="B"):
    """
    Blinded A/B side-by-side image.
    No algorithm names are shown.
    """
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), dpi=200)

    for ax, pos, lab in [(axes[0], pos_left, label_left), (axes[1], pos_right, label_right)]:
        nx.draw_networkx_edges(G, pos, ax=ax, width=0.6, alpha=0.30)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=20)

        # Fixed viewport for fairness
        ax.set_xlim(-1.05, 1.05)
        ax.set_ylim(-1.05, 1.05)
        ax.set_aspect("equal")
        ax.set_axis_off()

        # Large A/B label
        ax.text(
            0.02, 0.98, lab,
            transform=ax.transAxes,
            ha="left", va="top",
            fontsize=22, fontweight="bold"
        )

    plt.tight_layout()
    plt.savefig(out_png)
    plt.close(fig)


def select_graph_subset(graph_items, seed=7, per_family=2, max_n=160):
    """
    Pick a small balanced subset across families for a pilot human eval.
    Default: 2 graphs per family, n<=160.
    """
    rng = random.Random(seed)

    by_family = {}
    for item in graph_items:
        fam = item["family"]
        G = item["G"]
        if G.number_of_nodes() <= max_n:
            by_family.setdefault(fam, []).append(item)

    chosen = []
    for fam, items in sorted(by_family.items()):
        rng.shuffle(items)
        chosen.extend(items[:min(per_family, len(items))])

    return chosen


def main():
    out_root = os.path.join("out", "human_eval_pairs")
    ensure_dir(out_root)

    # 1) Get benchmark graphs from your existing generator
    graph_items = dl.build_benchmark_graphs(base_seed=7, mode="benchmark")

    # 2) Choose a manageable balanced subset (pilot)
    # Tune per_family and seeds_per_graph as needed.
    subset = select_graph_subset(graph_items, seed=7, per_family=2, max_n=160)
    seeds_per_graph = 1  # keep small for human eval; raise to 2 if needed

    print(f"Selected {len(subset)} graphs for human-eval pair generation.")
    print("Output folder:", os.path.abspath(out_root))

    manifest_path = os.path.join(out_root, "pairs_manifest.csv")
    fieldnames = [
        "pair_id", "image_file",
        "graph", "family", "n_nodes", "n_edges", "graph_seed", "graph_params",
        "init_seed",
        "left_label", "right_label",
        "left_algo", "right_algo",
        "left_score", "right_score",
        "left_overlap", "left_edge_cv", "left_stress", "left_aspect",
        "right_overlap", "right_edge_cv", "right_stress", "right_aspect",
    ]

    rng = random.Random(12345)
    pair_id = 0

    with open(manifest_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()

        for item_idx, item in enumerate(subset):
            gname = item["gname"]
            family = item["family"]
            graph_seed = item["graph_seed"]
            graph_params = item["graph_params"]
            G = item["G"]

            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
            nodes = list(G.nodes())

            for s in range(seeds_per_graph):
                init_seed = 10000 + 100 * item_idx + s
                init = dl.init_positions(nodes, seed=init_seed, low=INIT_LOW, high=INIT_HIGH)

                # Generate all 4 layouts + metrics
                layouts, metrics = run_all_algos_clean(G, init, init_seed)

                # Generate blinded pair images
                for (a1, a2) in PAIR_LIST:
                    # randomize left-right to prevent bias
                    if rng.random() < 0.5:
                        left_algo, right_algo = a1, a2
                    else:
                        left_algo, right_algo = a2, a1

                    left_pos = layouts[left_algo]
                    right_pos = layouts[right_algo]

                    ovL, cvL, stL, arL, scL = metrics[left_algo]
                    ovR, cvR, stR, arR, scR = metrics[right_algo]

                    img_name = f"pair_{pair_id:04d}_{gname}_init{init_seed}.png"
                    img_path = os.path.join(out_root, img_name)

                    plot_pair_image(img_path, G, left_pos, right_pos, "A", "B")

                    w.writerow({
                        "pair_id": pair_id,
                        "image_file": img_name,
                        "graph": gname,
                        "family": family,
                        "n_nodes": n_nodes,
                        "n_edges": n_edges,
                        "graph_seed": graph_seed,
                        "graph_params": graph_params,
                        "init_seed": init_seed,
                        "left_label": "A",
                        "right_label": "B",
                        "left_algo": left_algo,
                        "right_algo": right_algo,
                        "left_score": scL,
                        "right_score": scR,
                        "left_overlap": ovL,
                        "left_edge_cv": cvL,
                        "left_stress": stL,
                        "left_aspect": arL,
                        "right_overlap": ovR,
                        "right_edge_cv": cvR,
                        "right_stress": stR,
                        "right_aspect": arR,
                    })

                    pair_id += 1

    print("Done.")
    print("Pairs manifest:", os.path.abspath(manifest_path))
    print(f"Generated {pair_id} pair images in: {os.path.abspath(out_root)}")
    print("Tip: Use image_file column to embed images into a Google Form / survey.")


if __name__ == "__main__":
    main()