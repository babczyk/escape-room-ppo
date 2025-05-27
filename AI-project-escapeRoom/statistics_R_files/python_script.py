import json
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

def build_network_graph(weight_layers, bias_layers):
    G = nx.DiGraph()
    pos = {}
    layer_x = 0

    for l, (W, b) in enumerate(zip(weight_layers, bias_layers)):
        W = np.array(W)
        b = np.array(b)

        input_size, output_size = W.shape

        # Input layer nodes
        for i in range(input_size):
            node = f"L{l}_N{i}"
            G.add_node(node)
            pos[node] = (layer_x, -i)

        # Output layer nodes
        for j in range(output_size):
            node = f"L{l+1}_N{j}"
            G.add_node(node)
            pos[node] = (layer_x + 1, -j)

        # Edges (weights)
        for i in range(input_size):
            for j in range(output_size):
                weight = W[i][j]
                G.add_edge(f"L{l}_N{i}", f"L{l+1}_N{j}", weight=weight)

        layer_x += 1

    return G, pos

def visualize_all_networks_from_json(network_data):
    """
    Visualizes multiple neural networks stored in a JSON-like dictionary.

    Args:
        network_data: Dictionary with keys like 'in_actor_weight', 'in_actor_biase',
                      'in_critic_weight', 'in_critic_biase'.
    """
    networks = {
        "Actor 1": (network_data['IN_ACTOR_WEIGHTS'], network_data['IN_ACTOR_BIASES']),
        "Critic 1": (network_data['IN_CRITIC_WEIGHTS'], network_data['IN_CRITIC_BIASES']),
    }

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    for ax, (name, (weights, biases)) in zip(axes, networks.items()):
        G, pos = build_network_graph(weights, biases)

        edge_weights = [G[u][v]['weight'] for u, v in G.edges()]
        max_w = max(abs(w) for w in edge_weights) if edge_weights else 1
        edge_colors = ['red' if w < 0 else 'green' for w in edge_weights]
        edge_widths = [abs(w) / max_w * 2 for w in edge_weights]

        nx.draw(G, pos, with_labels=False, arrows=False,
                node_size=300, node_color='lightblue',
                edge_color=edge_colors, width=edge_widths, ax=ax)

        ax.set_title(name)
        ax.axis('off')

    plt.suptitle("PPO Neural Network Weights Visualization", fontsize=16)
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = 'PG_MAIN/AI-project-escapeRoom/statistics_R_files/ppo_model_episode.json'
    with open(file_path) as f:
        model_data = json.load(f)

    visualize_all_networks_from_json(model_data)
