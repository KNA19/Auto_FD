import os, math, random, csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from itertools import combinations
import json

# -----------------------------
# Utilities
# -----------------------------
def ensure_dir(p):
    os.makedirs(p, exist_ok=True)

def normalize_pos_dict(pos):
    # center + scale into roughly [-1, 1]
    nodes = list(pos.keys())
    arr = np.array([pos[n] for n in nodes], dtype=float)
    arr = arr - arr.mean(axis=0, keepdims=True)
    mx = np.max(np.abs(arr)) + 1e-12
    arr = arr / mx
    return {n: (float(arr[i,0]), float(arr[i,1])) for i, n in enumerate(nodes)}

def run_layout(G, init, algo, iters, seed=7):
    """
    Runs one algorithm from the same init positions and returns (pos, hist).
    pos is a dict node -> (x,y).
    hist is None for FR/KK, and a dict for Eades variants.
    """
    # Build adjacency dict once (needed for Eades variants)
    adj = {u: list(G.neighbors(u)) for u in G.nodes()}

    if algo == "Eades":
        pos, hist = eades_baseline(
            adj, init,
            iterations=iters,
            epsilon=0.01,
            cooling_factor=0.95,
            T0=3.0
        )
        return pos, hist

    if algo == "Adaptive-Eades":
        pos, hist = eades_adaptive(
            adj, init,
            iterations=iters,
            epsilon=0.02,
            rate=0.05,
            T0=3.0
        )
        return pos, hist

    if algo == "FR":
        pos = nx.spring_layout(G, pos=init, seed=seed, iterations=iters)
        return pos, None

    if algo == "KK":
        pos = nx.kamada_kawai_layout(G, pos=init)
        return pos, None

    raise ValueError(f"Unknown algo: {algo}")


def run_all_algos(G, init, iters, seed=7):
    """
    Runs all 4 algorithms from the same init.
    Returns:
      layouts: dict algo_name -> normalized position dict
      histories: dict algo_name -> history (or None)
    """
    layouts = {}
    histories = {}

    for algo in ["Eades", "Adaptive-Eades", "FR", "KK"]:
        pos, hist = run_layout(G, init, algo, iters, seed=seed)

        # IMPORTANT: normalize so all layouts are comparable for metrics/plots
        layouts[algo] = normalize_pos_dict(pos)
        histories[algo] = hist

    return layouts, histories

def init_positions(nodes, seed=1, low=0.0, high=10.0):
    rng = random.Random(seed)
    return {n: (rng.uniform(low, high), rng.uniform(low, high)) for n in nodes}

def euclid(p, q):
    return math.hypot(p[0]-q[0], p[1]-q[1]) + 1e-12


# -----------------------------
# Metrics (simple + demo-friendly)
# -----------------------------
OVERLAP_RADIUS = 0.04
W_OVERLAP = 5.0
W_EDGECV  = 1.0
W_STRESS  = 0.2
W_ASPECT  = 0.5

def overlap_penalty(pos, r=OVERLAP_RADIUS):
    # nodes as circles radius r in normalized space
    nodes = list(pos.keys())
    arr = np.array([pos[n] for n in nodes], dtype=float)
    n = len(nodes)
    pen = 0.0
    for i in range(n):
        for j in range(i + 1, n):
            d = math.hypot(arr[i, 0] - arr[j, 0], arr[i, 1] - arr[j, 1])
            pen += max(0.0, 2.0 * r - d)
    return pen

def edge_length_cv(G, pos):
    lens = []
    for u, v in G.edges():
        lens.append(euclid(pos[u], pos[v]))
    if not lens:
        return 0.0
    lens = np.array(lens, dtype=float)
    return float(np.std(lens) / (np.mean(lens) + 1e-12))

def sampled_stress(G, pos, pairs=1200, seed=1):
    """
    Weighted sampled stress on the largest connected component (if disconnected).

    pos is assumed normalized.
    Uses unique unordered pairs for more stable estimates.
    """
    if G.number_of_nodes() <= 1:
        return 0.0

    # Evaluate stress on largest connected component only (simple + stable)
    if not nx.is_connected(G):
        cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(cc).copy()
        pos = {n: pos[n] for n in G.nodes()}

    nodes = list(G.nodes())
    n = len(nodes)
    if n <= 1:
        return 0.0

    # All-pairs shortest-path distances (fine for your current graph sizes)
    sp = dict(nx.all_pairs_shortest_path_length(G))

    # Scale factor L = mean edge length (so 1-hop ~ L)
    el = [euclid(pos[u], pos[v]) for u, v in G.edges()]
    L = float(np.mean(el)) if el else 1.0

    total_pairs_avail = n * (n - 1) // 2

    # If requested pairs >= all pairs, compute exact over all unordered pairs
    if pairs is None or pairs >= total_pairs_avail:
        pair_list = [(a, b) for a, b in combinations(nodes, 2)]
    else:
        rng = random.Random(seed)
        idx_map = list(range(n))
        seen = set()

        # Sample unique unordered pairs (i, j) with i < j
        # Use a bounded attempt count to avoid infinite loops if pairs is large
        max_attempts = max(10 * pairs, 1000)
        attempts = 0
        while len(seen) < pairs and attempts < max_attempts:
            i, j = rng.sample(idx_map, 2)
            if i > j:
                i, j = j, i
            seen.add((i, j))
            attempts += 1

        pair_list = [(nodes[i], nodes[j]) for (i, j) in seen]

    total = 0.0
    cnt = 0
    for a, b in pair_list:
        dg = sp[a][b]  # connected because we reduced to largest CC if needed
        if dg <= 0:
            continue
        de = euclid(pos[a], pos[b])
        diff = de - L * dg
        total += (diff * diff) / (dg * dg + 1e-12)
        cnt += 1

    return total / max(cnt, 1)

def aspect_ratio_penalty(pos):
    xs = [pos[n][0] for n in pos]
    ys = [pos[n][1] for n in pos]
    w = (max(xs) - min(xs)) + 1e-12
    h = (max(ys) - min(ys)) + 1e-12
    ar = w / h
    return abs(math.log(ar))

def compute_metrics(G, pos, stress_pairs=1200, seed=1):
    # pos is already normalized in run_all_algos()
    ov = overlap_penalty(pos)
    cv = edge_length_cv(G, pos)
    st = sampled_stress(G, pos, pairs=stress_pairs, seed=seed)
    ar = aspect_ratio_penalty(pos)

    score = W_OVERLAP * ov + W_EDGECV * cv + W_STRESS * st + W_ASPECT * ar
    return ov, cv, st, ar, score

def extract_graph_features(G):
    n = G.number_of_nodes()
    m = G.number_of_edges()
    dens = nx.density(G) if n > 1 else 0.0

    degs = [d for _, d in G.degree()]
    if len(degs) == 0:
        deg_mean = deg_std = deg_max = 0.0
    else:
        arr = np.array(degs, dtype=float)
        deg_mean = float(np.mean(arr))
        deg_std  = float(np.std(arr))
        deg_max  = float(np.max(arr))

    n_cc = nx.number_connected_components(G) if n > 0 else 0
    largest_cc = max((len(c) for c in nx.connected_components(G)), default=0) if n > 0 else 0
    lcc_ratio = (largest_cc / n) if n > 0 else 0.0

    avg_clust = float(nx.average_clustering(G)) if n > 1 else 0.0

    return {
        "feat_n": n,
        "feat_m": m,
        "feat_density": dens,
        "feat_deg_mean": deg_mean,
        "feat_deg_std": deg_std,
        "feat_deg_max": deg_max,
        "feat_n_cc": n_cc,
        "feat_lcc_ratio": lcc_ratio,
        "feat_avg_clustering": avg_clust,
    }

# -----------------------------
# Your Eades baseline (modified: accept init positions + return history)
# Same log attraction as your baseline code
# -----------------------------
def eades_baseline(adj_dict, init_pos, iterations=10000, epsilon=0.01, cooling_factor=0.95,
                  k_repulsion=2.0, k_spring=1.0, ideal_spring_length=1.0, T0=3.0):
    positions = {n: (float(init_pos[n][0]), float(init_pos[n][1])) for n in adj_dict}
    temperature = T0
    hist_max_move = []
    hist_T = []

    nodes = list(adj_dict.keys())

    for _ in range(iterations):
        forces = {node: [0.0, 0.0] for node in nodes}

        # repulsion
        for i, u in enumerate(nodes):
            ux, uy = positions[u]
            for j in range(i+1, len(nodes)):
                v = nodes[j]
                vx, vy = positions[v]
                dx, dy = ux - vx, uy - vy
                d = math.hypot(dx, dy) + 1e-12
                rep = k_repulsion / (d*d)
                fx = rep * dx / d
                fy = rep * dy / d
                forces[u][0] += fx; forces[u][1] += fy
                forces[v][0] -= fx; forces[v][1] -= fy

        # attraction (log spring like your baseline)
        # to avoid double-counting, iterate undirected edge set
        seen = set()
        for u in nodes:
            for v in adj_dict[u]:
                a, b = (u, v) if str(u) < str(v) else (v, u)
                if (a, b) in seen:
                    continue
                seen.add((a, b))

                dx = positions[u][0] - positions[v][0]
                dy = positions[u][1] - positions[v][1]
                d = math.hypot(dx, dy) + 1e-12
                attraction = k_spring * math.log(d / (ideal_spring_length + 1e-12))
                fx = attraction * dx / d
                fy = attraction * dy / d
                forces[u][0] -= fx; forces[u][1] -= fy
                forces[v][0] += fx; forces[v][1] += fy

        max_move = 0.0
        for u in nodes:
            fx, fy = forces[u][0], forces[u][1]
            fmag = math.hypot(fx, fy) + 1e-12

            # MOVEMENT CAP: move at most 'temperature' per iteration
            move = min(fmag, temperature)

            dx = (fx / fmag) * move
            dy = (fy / fmag) * move

            max_move = max(max_move, math.hypot(dx, dy))
            positions[u] = (positions[u][0] + dx, positions[u][1] + dy)

        hist_max_move.append(max_move)
        hist_T.append(temperature)

        temperature *= cooling_factor
        if max_move < epsilon:
            break

    return positions, {"max_move": hist_max_move, "T": hist_T}


# -----------------------------
# Your Adaptive Eades (modified for fairness + stability)
# Key changes for tomorrow:
# 1) accept init positions
# 2) use SAME log attraction as baseline (so the only real change is temperature rule)
# 3) clamp temperature so it never becomes negative or collapses too fast
# -----------------------------
def eades_adaptive(adj, init_pos, iterations=10000, epsilon=0.02, rate=0.05,
                  k_rep=2.0, k_spring=1.0, ideal_spring_length=1.0, T0=3.0, Tmin=1e-4):
    pos = {n: (float(init_pos[n][0]), float(init_pos[n][1])) for n in adj}
    nodes = list(adj.keys())

    T = T0
    init_max = None
    hist_max_move = []
    hist_T = []

    # undirected edge list for attraction
    seen = set()
    edges = []
    for u in nodes:
        for v in adj[u]:
            a, b = (u, v) if str(u) < str(v) else (v, u)
            if (a, b) not in seen:
                seen.add((a, b))
                edges.append((u, v))

    for _ in range(iterations):
        F = {n: [0.0, 0.0] for n in nodes}

        # repulsion
        for i, u in enumerate(nodes):
            ux, uy = pos[u]
            for j in range(i+1, len(nodes)):
                v = nodes[j]
                vx, vy = pos[v]
                dx, dy = ux - vx, uy - vy
                d = math.hypot(dx, dy) + 1e-12
                rep = k_rep / (d*d)
                fx = rep * dx / d
                fy = rep * dy / d
                F[u][0] += fx; F[u][1] += fy
                F[v][0] -= fx; F[v][1] -= fy

        # attraction (log spring like baseline)
        for u, v in edges:
            dx = pos[u][0] - pos[v][0]
            dy = pos[u][1] - pos[v][1]
            d = math.hypot(dx, dy) + 1e-12
            att = k_spring * math.log(d / (ideal_spring_length + 1e-12))
            fx = att * dx / d
            fy = att * dy / d
            F[u][0] -= fx; F[u][1] -= fy
            F[v][0] += fx; F[v][1] += fy

        max_move = 0.0
        for u in nodes:
            fx, fy = F[u][0], F[u][1]
            fmag = math.hypot(fx, fy) + 1e-12

            # MOVEMENT CAP: move at most 'T' per iteration
            move = min(fmag, T)

            dx = (fx / fmag) * move
            dy = (fy / fmag) * move

            max_move = max(max_move, math.hypot(dx, dy))
            pos[u] = (pos[u][0] + dx, pos[u][1] + dy)

        hist_max_move.append(max_move)
        hist_T.append(T)

        if init_max is None:
            init_max = max_move if max_move > 0 else 1.0

        # your rule: T *= (1 - rate*(maxF/init_max)), but clamped
        ratio = max_move / (init_max + 1e-12)
        ratio = min(ratio, 4.0)  # safety clamp
        factor = 1.0 - rate * ratio
        factor = max(0.10, min(0.999, factor))
        T = max(Tmin, T * factor)

        if max_move < epsilon:
            break

    return pos, {"max_move": hist_max_move, "T": hist_T}


# -----------------------------
# Demo graphs
# -----------------------------
def build_benchmark_graphs(base_seed=7, mode="benchmark"):
    """
    Returns a list of graph items (dicts), each containing:
      {
        "gname": str,
        "family": str,
        "n_nodes": int,
        "graph_seed": int or None,
        "graph_params": str,
        "G": nx.Graph
      }
    """
    items = []

    # -------------------------
    # Configure benchmark scale
    # -------------------------
    if mode == "demo":
        # Small version (close to your current setup)
        # One graph per family
        configs = [
            ("grid", {"sizes": [100]}),
            ("tree", {"sizes": [120], "repeats": 1}),
            ("er",   {"sizes": [120], "p_list": [0.04], "repeats": 1}),
            ("ba",   {"sizes": [150], "m_list": [2], "repeats": 1}),
            ("ws",   {"sizes": [150], "k_list": [4], "beta_list": [0.20], "repeats": 1}),
        ]
    else:
        # Step-2 benchmark mode (moderate, practical)
        configs = [
            ("grid", {"sizes": [49, 100, 144]}),  # 7x7, 10x10, 12x12 (approx)
            ("tree", {"sizes": [50, 100, 150], "repeats": 2}),
            ("er",   {"sizes": [80, 120, 160], "p_list": [0.03, 0.04], "repeats": 2}),
            ("ba",   {"sizes": [80, 120, 160], "m_list": [2, 3], "repeats": 2}),
            ("ws",   {"sizes": [80, 120, 160], "k_list": [4, 6], "beta_list": [0.20], "repeats": 2}),
        ]

    # helper: stable topology seed generator
    def topo_seed(*vals):
        # simple deterministic combiner
        h = 1469598103934665603
        for v in vals:
            s = str(v)
            for ch in s:
                h ^= ord(ch)
                h *= 1099511628211
                h &= 0xFFFFFFFFFFFFFFFF
        return (base_seed + h) % (2**31 - 1)

    # -------------------------
    # Build graph families
    # -------------------------
    for family, cfg in configs:
        if family == "grid":
            for n_target in cfg["sizes"]:
                # make a near-square grid
                side = int(round(math.sqrt(n_target)))
                side = max(2, side)
                G = nx.grid_2d_graph(side, side)
                G = nx.convert_node_labels_to_integers(G)

                n = G.number_of_nodes()
                m = G.number_of_edges()
                gname = f"grid_{side}x{side}"
                items.append({
                    "gname": gname,
                    "family": "grid",
                    "n_nodes": n,
                    "n_edges": m,
                    "graph_seed": None,
                    "graph_params": f"rows={side},cols={side}",
                    "G": G
                })

        elif family == "tree":
            repeats = cfg.get("repeats", 1)
            for n in cfg["sizes"]:
                for r in range(repeats):
                    gseed = topo_seed("tree", n, r)
                    # NetworkX 3.4+: random_tree removed; use random_labeled_tree
                    G = nx.random_labeled_tree(n, seed=gseed)
                    m = G.number_of_edges()
                    gname = f"tree_n{n}_rep{r}"
                    items.append({
                        "gname": gname,
                        "family": "tree",
                        "n_nodes": n,
                        "n_edges": m,
                        "graph_seed": gseed,
                        "graph_params": "random_labeled_tree",
                        "G": G
                    })

        elif family == "er":
            repeats = cfg.get("repeats", 1)
            for n in cfg["sizes"]:
                for p in cfg["p_list"]:
                    for r in range(repeats):
                        gseed = topo_seed("er", n, p, r)
                        G = nx.gnp_random_graph(n, p, seed=gseed)
                        # keep largest CC for consistency of graph instances (optional)
                        if G.number_of_nodes() > 0 and not nx.is_connected(G):
                            G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
                        n_eff = G.number_of_nodes()
                        m_eff = G.number_of_edges()
                        gname = f"er_n{n}_p{str(p).replace('.', '')}_rep{r}"
                        items.append({
                            "gname": gname,
                            "family": "er",
                            "n_nodes": n_eff,
                            "n_edges": m_eff,
                            "graph_seed": gseed,
                            "graph_params": f"n={n},p={p}",
                            "G": G
                        })

        elif family == "ba":
            repeats = cfg.get("repeats", 1)
            for n in cfg["sizes"]:
                for m_attach in cfg["m_list"]:
                    if m_attach >= n:
                        continue
                    for r in range(repeats):
                        gseed = topo_seed("ba", n, m_attach, r)
                        G = nx.barabasi_albert_graph(n, m_attach, seed=gseed)
                        n_eff = G.number_of_nodes()
                        m_eff = G.number_of_edges()
                        gname = f"ba_n{n}_m{m_attach}_rep{r}"
                        items.append({
                            "gname": gname,
                            "family": "ba",
                            "n_nodes": n_eff,
                            "n_edges": m_eff,
                            "graph_seed": gseed,
                            "graph_params": f"n={n},m={m_attach}",
                            "G": G
                        })

        elif family == "ws":
            repeats = cfg.get("repeats", 1)
            for n in cfg["sizes"]:
                for k in cfg["k_list"]:
                    if k >= n or k % 2 == 1:
                        continue
                    for beta in cfg["beta_list"]:
                        for r in range(repeats):
                            gseed = topo_seed("ws", n, k, beta, r)
                            G = nx.watts_strogatz_graph(n, k, beta, seed=gseed)
                            if G.number_of_nodes() > 0 and not nx.is_connected(G):
                                G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
                            n_eff = G.number_of_nodes()
                            m_eff = G.number_of_edges()
                            gname = f"ws_n{n}_k{k}_b{str(beta).replace('.', '')}_rep{r}"
                            items.append({
                                "gname": gname,
                                "family": "ws",
                                "n_nodes": n_eff,
                                "n_edges": m_eff,
                                "graph_seed": gseed,
                                "graph_params": f"n={n},k={k},beta={beta}",
                                "G": G
                            })

        else:
            raise ValueError(f"Unknown family config: {family}")

    return items


# -----------------------------
# Plotting
# -----------------------------
def plot_panel(out_png, G, layouts, metrics_map):
    # layouts: dict name -> normalized pos dict
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    order = ["Eades", "Adaptive-Eades", "FR", "KK"]
    for ax, name in zip(axes, order):
        pos = layouts[name]  # already normalized in run_all_algos()
        nx.draw_networkx_edges(G, pos, ax=ax, width=0.6, alpha=0.5)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=35)

        # Step-1 metrics now include aspect ratio penalty
        ov, cv, st, ar, sc = metrics_map[name]
        ax.set_title(
            f"{name}\n"
            f"overlap={ov:.3f}  edgeCV={cv:.3f}  stress={st:.3f}  aspect={ar:.3f}\n"
            f"score={sc:.3f}"
        )
        ax.set_axis_off()
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close(fig)

def plot_convergence(out_png, hist_base, hist_adap):
    x1 = np.arange(len(hist_base["max_move"]))
    x2 = np.arange(len(hist_adap["max_move"]))

    plt.figure(figsize=(10, 5))
    plt.plot(x1, hist_base["max_move"], label="Eades max-move")
    plt.plot(x2, hist_adap["max_move"], label="Adaptive-Eades max-move")
    plt.xlabel("Iteration")
    plt.ylabel("Max displacement per iteration")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png, dpi=160)
    plt.close()


# -----------------------------
# Run demo
# -----------------------------
def main():
    out_dir = "out"
    ensure_dir(out_dir)

    # -------------------------
    # Step-2 run mode / settings
    # -------------------------
    mode = "benchmark"   # "demo" or "benchmark"
    save_panels = False  # False for faster Step-2 dataset generation
    save_convergence_once = True

    # Budgets
    seeds_per_graph = 10 if mode == "benchmark" else 1
    iters = 1000
    stress_pairs = 600 if mode == "benchmark" else 800

    # Build benchmark graphs (with metadata)
    graph_items = build_benchmark_graphs(base_seed=7, mode=mode)

    summary_rows = []
    summary_wide_rows = []
    conv_done = False

    # -------------------------
    # Main benchmark loop
    # -------------------------
    for g_idx, item in enumerate(graph_items):
        gname = item["gname"]
        family = item["family"]
        graph_seed = item["graph_seed"]
        graph_params = item["graph_params"]
        G = item["G"]

        nodes = list(G.nodes())
        n_nodes = G.number_of_nodes()
        n_edges = G.number_of_edges()

        feats = extract_graph_features(G)

        # group id for leakage-free splitting later
        # use graph_seed if available, else gname is fine
        instance_id = f"{gname}|gseed={graph_seed}"

        for s in range(seeds_per_graph):
            init_seed = 10000 + 100 * g_idx + s  # deterministic + distinct seed per (graph instance, seed index)
            init = init_positions(nodes, seed=init_seed, low=0.0, high=10.0)

            # run all 4 algorithms consistently (same init, same iters)
            layouts, histories = run_all_algos(G, init, iters=iters, seed=7)

            # Deterministic + fair stress sampling seed per (graph instance, init seed)
            stress_seed = 10000 + 100 * g_idx + s

            metrics_map = {
                name: compute_metrics(G, layouts[name], stress_pairs=stress_pairs, seed=stress_seed)
                for name in layouts
            }

            # best algorithm label for this (graph, init seed)
            best_algo = min(metrics_map, key=lambda a: metrics_map[a][-1])  # last element = score

            # -------------------------
            # Wide summary row (one row per graph+seed)
            # -------------------------
            summary_wide_rows.append({
                "graph": gname,
                "family": family,
                "n_nodes": n_nodes,
                "n_edges": n_edges,
                "graph_seed": graph_seed,
                "graph_params": graph_params,
                "init_seed": init_seed,
                "stress_seed": stress_seed,
                "score_Eades": metrics_map["Eades"][-1],
                "score_Adaptive-Eades": metrics_map["Adaptive-Eades"][-1],
                "score_FR": metrics_map["FR"][-1],
                "score_KK": metrics_map["KK"][-1],
                "best_algo": best_algo,
                "instance_id": instance_id,
                **feats
            })

            # -------------------------
            # Optional panel image saving (Step-2 usually off for speed)
            # -------------------------
            if save_panels:
                out_png = os.path.join(out_dir, f"{gname}_seed{s}.png")
                plot_panel(out_png, G, layouts, metrics_map)

            # save one convergence plot only (optional)
            if save_convergence_once and (not conv_done):
                conv_path = os.path.join(out_dir, f"convergence_{gname}_seed{s}.png")
                plot_convergence(conv_path, histories["Eades"], histories["Adaptive-Eades"])
                conv_done = True

            # -------------------------
            # Long summary rows (one row per algorithm)
            # metrics tuple = (ov, cv, st, ar, score)
            # -------------------------
            for name, (ov, cv, st, ar, sc) in metrics_map.items():
                summary_rows.append({
                    "graph": gname,
                    "family": family,
                    "n_nodes": n_nodes,
                    "n_edges": n_edges,
                    "graph_seed": graph_seed,
                    "graph_params": graph_params,
                    "seed": s,          # legacy seed index
                    "init_seed": init_seed,
                    "stress_seed": stress_seed,
                    "algo": name,
                    "overlap": ov,
                    "edge_cv": cv,
                    "stress": st,
                    "aspect": ar,
                    "score": sc,
                    "instance_id": instance_id,
                    **feats
                })

    # -------------------------
    # Write long summary.csv
    # -------------------------
    
    feature_fields = [
    "feat_n","feat_m","feat_density","feat_deg_mean","feat_deg_std","feat_deg_max",
    "feat_n_cc","feat_lcc_ratio","feat_avg_clustering"
    ]

    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames = [
                "graph","family","n_nodes","n_edges","graph_seed","graph_params",
                "seed","init_seed","stress_seed",
                "instance_id",
                *feature_fields,
                "algo","overlap","edge_cv","stress","aspect","score"
            ]
        )
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    # -------------------------
    # Write wide summary_wide.csv
    # -------------------------
    wide_path = os.path.join(out_dir, "summary_wide.csv")
    with open(wide_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(
            f,
            fieldnames = [
                "graph","family","n_nodes","n_edges","graph_seed","graph_params",
                "init_seed","stress_seed",
                "instance_id",
                *feature_fields,
                "score_Eades","score_Adaptive-Eades","score_FR","score_KK",
                "best_algo"
            ]
        )
        w.writeheader()
        for r in summary_wide_rows:
            w.writerow(r)

    # -------------------------
    # Write run_config.json (reproducibility manifest)
    # -------------------------
    cfg_path = os.path.join(out_dir, "run_config.json")
    run_cfg = {
        "mode": mode,
        "save_panels": save_panels,
        "save_convergence_once": save_convergence_once,
        "seeds_per_graph": seeds_per_graph,
        "iterations": iters,
        "stress_pairs": stress_pairs,
        "algorithms": ["Eades", "Adaptive-Eades", "FR", "KK"],
        "score_weights": {
            "W_OVERLAP": W_OVERLAP,
            "W_EDGECV": W_EDGECV,
            "W_STRESS": W_STRESS,
            "W_ASPECT": W_ASPECT
        },
        "overlap_radius": OVERLAP_RADIUS,
        "notes": "Step-2 benchmark generation for AutoFD (4-algorithm portfolio)"
    }
    with open(cfg_path, "w", encoding="utf-8") as f:
        json.dump(run_cfg, f, indent=2)

    # -------------------------
    # Aggregate summaries (quick sanity checks)
    # -------------------------
    # Win counts from wide rows
    win_counts = {}
    for r in summary_wide_rows:
        win_counts[r["best_algo"]] = win_counts.get(r["best_algo"], 0) + 1

    # Average score by algorithm from long rows
    algo_to_scores = {}
    for r in summary_rows:
        algo_to_scores.setdefault(r["algo"], []).append(r["score"])

    print("Done.")
    print("See outputs in:", os.path.abspath(out_dir))
    print("Key files:")
    print(" - summary csv:", os.path.abspath(csv_path))
    print(" - summary wide csv:", os.path.abspath(wide_path))
    print(" - run config:", os.path.abspath(cfg_path))
    if save_panels:
        print(" - panel images: *.png")
    if save_convergence_once:
        print(" - convergence plot: convergence_*.png")

    print("\nBenchmark summary:")
    print(f" - graph instances: {len(graph_items)}")
    print(f" - init seeds per graph: {seeds_per_graph}")
    print(f" - labeled rows (wide): {len(summary_wide_rows)}")
    print(f" - rows (long): {len(summary_rows)}")

    print("\nWin counts (best_algo):")
    for algo in ["Eades", "Adaptive-Eades", "FR", "KK"]:
        print(f" - {algo}: {win_counts.get(algo, 0)}")

    print("\nAverage score by algorithm:")
    for algo in ["Eades", "Adaptive-Eades", "FR", "KK"]:
        vals = algo_to_scores.get(algo, [])
        if vals:
            print(f" - {algo}: {sum(vals)/len(vals):.4f} (n={len(vals)})")
        else:
            print(f" - {algo}: n=0")

if __name__ == "__main__":
    main()