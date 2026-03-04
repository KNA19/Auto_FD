import os, math, random, csv
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


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
def overlap_penalty(pos, r=0.04):
    # nodes as circles radius r in normalized space
    nodes = list(pos.keys())
    arr = np.array([pos[n] for n in nodes], dtype=float)
    n = len(nodes)
    pen = 0.0
    for i in range(n):
        for j in range(i+1, n):
            d = math.hypot(arr[i,0]-arr[j,0], arr[i,1]-arr[j,1])
            pen += max(0.0, 2.0*r - d)
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
    # stress ~ mean over sampled pairs of (||xi-xj|| - L*dG)^2 / dG^2
    # L = mean edge length (so 1 hop ~ L)
    if G.number_of_nodes() <= 1:
        return 0.0
    if not nx.is_connected(G):
        # keep it simple for tomorrow: evaluate on largest CC
        cc = max(nx.connected_components(G), key=len)
        G = G.subgraph(cc).copy()
        pos = {n: pos[n] for n in G.nodes()}

    nodes = list(G.nodes())
    sp = dict(nx.all_pairs_shortest_path_length(G))

    # scale L from mean edge length
    el = []
    for u, v in G.edges():
        el.append(euclid(pos[u], pos[v]))
    L = float(np.mean(el)) if el else 1.0

    rng = random.Random(seed)
    total = 0.0
    cnt = 0
    n = len(nodes)
    for _ in range(pairs):
        a = nodes[rng.randrange(n)]
        b = nodes[rng.randrange(n)]
        if a == b:
            continue
        if b not in sp.get(a, {}):
            continue
        dg = sp[a][b]
        if dg <= 0:
            continue
        de = euclid(pos[a], pos[b])
        diff = de - L * dg
        total += (diff*diff) / (dg*dg + 1e-12)
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
    ov = overlap_penalty(pos)
    cv = edge_length_cv(G, pos)
    st = sampled_stress(G, pos, pairs=stress_pairs, seed=seed)
    ar = aspect_ratio_penalty(pos)

    score = 5.0*ov + 1.0*cv + 0.2*st + 0.5*ar
    return ov, cv, st, ar, score


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
def build_graphs(seed=7):
    graphs = []

    G = nx.grid_2d_graph(10, 10)
    G = nx.convert_node_labels_to_integers(G)
    graphs.append(("grid_10x10", G))

    # FIX: random_tree removed in NX 3.4+
    G = nx.random_labeled_tree(120, seed=seed)
    graphs.append(("tree_120", G))

    G = nx.gnp_random_graph(120, 0.04, seed=seed)
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    graphs.append(("erdos_120_p004", G))

    G = nx.barabasi_albert_graph(150, 2, seed=seed)
    graphs.append(("ba_150_m2", G))

    G = nx.watts_strogatz_graph(150, 4, 0.20, seed=seed)
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    graphs.append(("ws_150_k4_p020", G))

    return graphs


# -----------------------------
# Plotting
# -----------------------------
def plot_panel(out_png, G, layouts, metrics_map):
    # layouts: dict name -> pos dict
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    order = ["Eades", "Adaptive-Eades", "FR", "KK"]
    for ax, name in zip(axes, order):
        pos = normalize_pos_dict(layouts[name])
        nx.draw_networkx_edges(G, pos, ax=ax, width=0.6, alpha=0.5)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=35)

        ov, cv, st, sc = metrics_map[name]
        ax.set_title(f"{name}\noverlap={ov:.3f}  edgeCV={cv:.3f}  stress={st:.3f}\nscore={sc:.3f}")
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

    graphs = build_graphs(seed=7)

    summary_rows = []
    summary_wide_rows = []
    conv_done = False

    seeds_per_graph = 1
    iters = 400

    for gname, G in graphs:
        nodes = list(G.nodes())

        for s in range(seeds_per_graph):
            init = init_positions(nodes, seed=100 + s, low=0.0, high=10.0)

            # run all 4 algorithms consistently
            layouts, histories = run_all_algos(G, init, iters=iters, seed=7)

            metrics_map = {
                name: compute_metrics(G, layouts[name], stress_pairs=800, seed=7 + s)
                for name in layouts
            }

            best_algo = min(metrics_map, key=lambda a: metrics_map[a][-1])  # last element is score

            summary_wide_rows.append({
                "graph": gname,
                "seed": s,
                "score_Eades": metrics_map["Eades"][-1],
                "score_Adaptive-Eades": metrics_map["Adaptive-Eades"][-1],
                "score_FR": metrics_map["FR"][-1],
                "score_KK": metrics_map["KK"][-1],
                "best_algo": best_algo
            })

            out_png = os.path.join(out_dir, f"{gname}_seed{s}.png")
            plot_panel(out_png, G, layouts, metrics_map)

            if not conv_done:
                conv_path = os.path.join(out_dir, f"convergence_{gname}_seed{s}.png")
                plot_convergence(conv_path, histories["Eades"], histories["Adaptive-Eades"])
                conv_done = True

            for name, (ov, cv, st, sc) in metrics_map.items():
                summary_rows.append({
                    "graph": gname,
                    "seed": s,
                    "algo": name,
                    "overlap": ov,
                    "edge_cv": cv,
                    "stress": st,
                    "score": sc
                })

    # write summary.csv
    csv_path = os.path.join(out_dir, "summary.csv")
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["graph","seed","algo","overlap","edge_cv","stress","score"])
        w.writeheader()
        for r in summary_rows:
            w.writerow(r)

    print("Done.")
    print("See outputs in:", os.path.abspath(out_dir))
    print("Key files:")
    print(" - panel images: *.png")
    print(" - convergence plot: convergence_*.png")
    print(" - summary csv:", os.path.abspath(csv_path))


if __name__ == "__main__":
    main()