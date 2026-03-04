# initial_layouts_panel.py
# Creates ONE image with the INITIAL (random) layouts for 4 graph categories
# and saves it to: out_initial/initial_layouts_4cats.png

import os
import random
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def ensure_dir(p: str) -> None:
    os.makedirs(p, exist_ok=True)


def init_positions(nodes, seed=100, low=0.0, high=10.0):
    rng = random.Random(seed)
    return {n: (rng.uniform(low, high), rng.uniform(low, high)) for n in nodes}


def normalize_pos_dict(pos):
    # center + scale into roughly [-1, 1]
    nodes = list(pos.keys())
    arr = np.array([pos[n] for n in nodes], dtype=float)
    arr = arr - arr.mean(axis=0, keepdims=True)
    mx = np.max(np.abs(arr)) + 1e-12
    arr = arr / mx
    return {n: (float(arr[i, 0]), float(arr[i, 1])) for i, n in enumerate(nodes)}


def build_4_graph_categories(seed=7):
    graphs = []

    # 1) Grid
    G = nx.grid_2d_graph(10, 10)
    G = nx.convert_node_labels_to_integers(G)
    graphs.append(("Grid (10x10)", G))

    # 2) Tree
    G = nx.random_labeled_tree(120, seed=seed)
    graphs.append(("Tree (n=120)", G))

    # 3) Erdos-Renyi (keep largest connected component if needed)
    G = nx.gnp_random_graph(120, 0.04, seed=seed)
    if not nx.is_connected(G):
        G = G.subgraph(max(nx.connected_components(G), key=len)).copy()
    graphs.append(("Erdos–Renyi (n≈120, p=0.04)", G))

    # 4) Barabasi-Albert
    G = nx.barabasi_albert_graph(150, 2, seed=seed)
    graphs.append(("Barabasi–Albert (n=150, m=2)", G))

    return graphs


def plot_initial_panel(out_png, graphs, base_init_seed=100):
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()

    for i, ((title, G), ax) in enumerate(zip(graphs, axes)):
        init = init_positions(list(G.nodes()), seed=base_init_seed + i, low=0.0, high=10.0)
        pos = normalize_pos_dict(init)

        n = G.number_of_nodes()
        node_size = max(10, int(3000 / max(n, 1)))  # auto-scale a bit across sizes

        nx.draw_networkx_edges(G, pos, ax=ax, width=0.6, alpha=0.5)
        nx.draw_networkx_nodes(G, pos, ax=ax, node_size=node_size)

        ax.set_title(f"{title}\n(n={G.number_of_nodes()}, m={G.number_of_edges()})")
        ax.set_axis_off()
        ax.set_aspect("equal")

    plt.tight_layout()
    plt.savefig(out_png, dpi=200)
    plt.close(fig)


def main():
    out_dir = "out_initial"
    ensure_dir(out_dir)

    graphs = build_4_graph_categories(seed=7)

    out_png = os.path.join(out_dir, "initial_layouts_4cats.png")
    plot_initial_panel(out_png, graphs, base_init_seed=100)

    print("Done.")
    print("Saved:", os.path.abspath(out_png))


if __name__ == "__main__":
    main()